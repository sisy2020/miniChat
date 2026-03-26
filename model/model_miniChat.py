import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from .config import miniChatConfig

###Qwen3\Llama3 Dense结构
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):#dim需要是int
        super().__init__()
        self.eps = eps
        #γ初始化为1.0，刚开始不起作用，直接标准化，随着训练进行，γ会逐渐起作用
        #nn.Parameter会一起跟着求导更新参数
        self.weight = nn.Parameter(torch.ones(dim))

    #(bs,seq_len,hidden_size)只会在hidden_size维度作用，所以x是一个向量
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        #RMSNorm=(ai)*γ(γ是可训练的参数)
        #(ai)=x/sqrt(mean(x^2))
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6):
    """
    预计算 RoPE (Rotary Position Embedding) 的 cos 和 sin 频率
    
    Args:
        dim: 注意力头的维度 (head_dim)
        end: 最大序列长度
        rope_base: RoPE 的基础频率，默认 1e6
    
    Returns:
        freqs_cos: cos 频率张量 (end, dim)
        freqs_sin: sin 频率张量 (end, dim)
    """
    # 计算频率：θ_i = base^(-2i/d), i ∈ [0, d/2),base=rope_base=1000000
    #torch.arange(0,dim,2)生成一个一维整数张量:从0开始，到dim结束，步长为2,即[0,2,4,...,dim-2]
    #[: (dim // 2)]表示取前dim // 2个元素
    #即i=0 0 1 1 2 2 ... dim/2-1 dim/2-1,两个元素一组，i表示组数
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 生成位置索引 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    
    # 计算每个位置的频率：pos * θ_i
    #结果形状是(end, dim/2)，构造RoPE中“位置×频率”的二维网格，后面再对这个freqs做cos、sin得到freqs_cos和freqs_sin。
    #512*256
    freqs = torch.outer(t, freqs).float()
    
    # 计算 cos 和 sin，并复制一次（用于 rotate_half）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用 RoPE (Rotary Position Embedding) 到 query 和 key
    
    Args:
        q: Query 张量 (batch, seq_len, num_heads, head_dim)
        k: Key 张量 (batch, seq_len, num_kv_heads, head_dim)
        cos: 预计算的 cos 频率
        sin: 预计算的 sin 频率
        unsqueeze_dim: 用于广播的维度
    
    Returns:
        q_embed: 应用 RoPE 后的 query
        k_embed: 应用 RoPE 后的 key
    """
    def rotate_half(x):
        """将特征维度分成两半并交换位置（用于 RoPE 旋转）"""
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    #把Q、K做旋转（位置编码），Q、K做旋转后，Q、K的每个位置都对应一个不同的频率
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    对 KV 进行重复以匹配 Query 的头数（用于 Grouped Query Attention）
    等价于 torch.repeat_interleave(x, dim=2, repeats=n_rep)，但更高效
    
    Args:
        x: KV 张量 (batch, seq_len, num_kv_heads, head_dim)
        n_rep: 重复次数 (num_heads // num_kv_heads)
    
    Returns:
        重复后的张量 (batch, seq_len, num_heads, head_dim)
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力机制（支持 Grouped Query Attention 和 Flash Attention）
    """
    def __init__(self, args: miniChatConfig):
        super().__init__()
        # GQA: 允许 KV 头数少于 Query 头数
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        #assert确保Query头数能被KV头数整除，否则会报错，确保GQA能正常工作
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        self.num_heads = args.num_attention_heads  # Query 头数
        self.num_kv_heads = self.num_key_value_heads  # KV 头数z'x
        #因为KV只存一份，运算时要临时生成备份
        self.n_rep = self.num_heads // self.num_kv_heads  # KV 重复次数
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        # QKV 投影层，即全连接层linear
        #Wq,Wk,Wv(hidden_size, num_attention_heads * head_dim)
        #Wq 512->512 Wk 512->128,Wk\v要小一些，k-v头少一些，所以映射出来向量短一些
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)#残差连接的dropout
        self.dropout = args.dropout
        
        # Flash Attention 支持检测
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        Args:
            x: 输入张量tensor (batch, seq_len, hidden_size)
            position_embeddings: (cos, sin) RoPE 位置编码
            past_key_value: KV cache，用于推理加速
            use_cache: 是否返回新的 KV cache，区别推理还是训练，推理就用past_k_v
            attention_mask: 注意力掩码 (batch, seq_len)，1=有效位置，0=padding（训练时）
        
        Returns:
            output: 注意力输出 (batch, seq_len, hidden_size)
            past_kv: 新的 KV cache（如果 use_cache=True）
        """
        bsz, seq_len, _ = x.shape
        
        # QKV 投影，生成Q、K、V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        #分头hidden_size->num_head*head_dim
        # Reshape 为多头格式：(batch, seq_len, num_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # 应用 RoPE 位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # KV cache 实现（用于推理加速）
        if past_key_value is not None:
            #每步只需算当前token的K/V，把之前步的key和当前步的key在seq_len维上拼在一起
            #(batch, past_len, num_kv_heads, head_dim)
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 转置并重复 KV（用于 GQA）
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        xq, xk, xv = (
            #转置，为了和k v做点积
            xq.transpose(1, 2),
            #重复KV，不然GQA维度不匹配
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 使用 Flash Attention (PyTorch >= 2.0) - 仅在训练时且无 KV cache 时启用
        if self.flash and (seq_len > 1) and (past_key_value is None):
            if attention_mask is None or torch.all(attention_mask == 1):
                # 无 padding，使用 is_causal=True（最快路径）
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True
                )
            else:
                # 有 padding，构造 Boolean mask (True=可见, False=屏蔽)
                # Causal mask: 下三角为 True（可见），上三角为 False（屏蔽）
                causal_mask = torch.tril(
                    torch.ones((seq_len, seq_len), device=xq.device, dtype=torch.bool),
                    diagonal=0
                )
                # Padding mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
                # 1 -> True (可见), 0 -> False (padding，屏蔽)
                padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)
                
                # 组合 mask: 两个都为 True 才能参与运算（逻辑与）
                # causal_mask: (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                # padding_mask: (batch, 1, 1, seq_len) -> broadcast 到每个 query 位置
                combined_mask = causal_mask.unsqueeze(0) & padding_mask
                
                output = F.scaled_dot_product_attention(
                    xq, xk, xv,
                    attn_mask=combined_mask,
                    dropout_p=self.dropout if self.training else 0.0
                )
        else:
            # 传统 Attention 实现（用于推理时的 KV cache 或 PyTorch < 2.0）
            #Q@K.T
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 应用 causal mask（上三角设为 -inf）
            #scores+causal mask
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            # 应用 padding mask
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                #scores+causal mask+padding mask
                scores = scores + extended_attention_mask

            #softmax
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 恢复形状并输出投影,乘Wo
        ##output:bs,num_heads,seq_len,head_dim
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    前馈神经网络（SwiGLU 激活函数）
    结构: Gate(x) * Up(x) -> Down
    """
    def __init__(self, config: miniChatConfig):
        super().__init__()
        # 计算中间层大小：默认为 hidden_size * 8/3，向上取整到 64 的倍数
        intermediate_size = config.intermediate_size
        if intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # 向上取整到 64 的倍数
        
        #门 降维 升维
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """SwiGLU: act(gate_proj(x)) * up_proj(x) -> down_proj"""
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class miniChatBlock(nn.Module):
    """
    Transformer 块：Self-Attention + FeedForward
    采用 Pre-Norm 结构（Norm before attention/mlp）
    """
    def __init__(self, layer_id: int, config: miniChatConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        #block组成元件
        self.self_attn = Attention(config)
        #标识每一层
        self.layer_id = layer_id
        #一层有两个Norm，定义2个Norm，因为Norm有可学习参数
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """
        前向传播：Pre-Norm Transformer Block
        
        结构：
            x = x + Attention(Norm(x))#做残差
            x = x + MLP(Norm(x))
        """
        # Self-Attention with residual connection
        #因为需要残差连接，所以需要保存上一个x的状态
        residual = hidden_states
        #Attention(Norm(x))
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        #x = x + Attention(Norm(x))
        hidden_states += residual
        
        # FeedForward with residual connection
        #x = x + MLP(Norm(x))
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class miniChatModel(nn.Module):
    """
    miniChat 模型主体（Decoder-only Transformer）
    """
    def __init__(self, config: miniChatConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        
        # Token Embedding
        # nn.Embedding相当于Linear，只不过能直接查表，更快
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer Blocks
        # l:layer_id
        self.layers = nn.ModuleList([miniChatBlock(l, config) for l in range(self.num_hidden_layers)])
        
        # 最终的 LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 频率（注册为 buffer，不参与训练但会保存在模型中）
        # 在模型中计算，而不是在block中计算频率，防止反复计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            attention_mask: 注意力掩码 (batch, seq_len)，1=有效位置，0=padding
            past_key_values: KV cache 列表，用于推理加速
            use_cache: 是否返回新的 KV cache
        
        Returns:
            hidden_states: 最后一层的隐藏状态 (batch, seq_len, hidden_size)，Output中的Linear层，先不做分类
            presents: 新的 KV cache 列表
        """
        batch_size, seq_length = input_ids.shape
        
        # 处理 past_key_values
        if hasattr(past_key_values, 'layers'): 
            past_key_values = None
        #初始值为none
        past_key_values = past_key_values or [None] * len(self.layers)
        
        # 计算起始位置（用于 RoPE）
        #第一层 Decoder 的 Key 矩阵中注意力头的数量（num_heads）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        #####开始！！！输入input_ids（batchsize,seq_len）
        # Token Embedding + Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 获取当前序列的位置编码（从 start_pos 开始）
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层前向传播
        presents = []
        #layer:ModelList的一个层，返回隐藏层和k-v，k-v一直存着
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终 LayerNorm
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class miniChatForCausalLM(PreTrainedModel, GenerationMixin):
    """
    PreTrainedModel
    GenerationMixin 做下一个token的生成，推理任务
    miniChat 因果语言模型（用于文本生成）
    在 miniChatModel 基础上添加 Language Modeling Head
    """
    config_class = miniChatConfig

    def __init__(self, config: miniChatConfig = None):
        self.config = config or miniChatConfig()
        super().__init__(self.config)
        
        # Transformer 主体
        self.model = miniChatModel(self.config)
        
        # Language Modeling Head（与 embed_tokens 权重共享）
        # embedding的权重矩阵:vocab_size->hidden_size,lm_head(输出过的线性层)hidden_size->vocab_size
        # 转置的关系，所以共用节约模型参数
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # 权重绑定

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        前向传播（用于训练和推理）
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            attention_mask: 注意力掩码 (batch, seq_len)
            labels: 标签 (batch, seq_len)，用于计算 loss
            past_key_values: KV cache
            use_cache: 是否返回 KV cache
            logits_to_keep: 保留最后多少个 token 的 logits（节省内存）
        
        Returns:
            CausalLMOutputWithPast: 包含 loss, logits, past_key_values, hidden_states
        """
        # Transformer 前向传播
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        #预测下一个token
        # 计算 logits（可选择只保留最后几个 token的logits）
        # 因为其实预测下一个token只需要当前token的logits向量softmax就够了
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # 计算交叉熵损失（如果提供了 labels，训练阶段）
        loss = None
        if labels is not None:
            # 标准的自回归语言模型 loss 计算：
            # 预测 token[i+1]，使用 token[0:i] 的信息
            # shift_logits: [0, 1, ..., n-2] 位置的预测
            # shift_labels: [1, 2, ..., n-1] 位置的真实标签
            # 输入左移，就是label（label是预测的下一个token）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                #输出
                shift_logits.view(-1, shift_logits.size(-1)), 
                #label
                shift_labels.view(-1), 
                #padding的token_id=-100，不计算padding的loss
                #每个token的loss取均值
                ignore_index=-100  # 忽略 padding 和 mask 的位置
            )

        #记录器，把这些返回值都记录下来
        output = CausalLMOutputWithPast(
            loss=loss, 
            logits=logits, 
            past_key_values=past_key_values, 
            hidden_states=hidden_states
        )
        return output