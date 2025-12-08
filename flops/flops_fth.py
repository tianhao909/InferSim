from config.model_config import ModelConfig


def gemm_flops(m, n, k):
    # 计算矩阵乘法的浮点运算次数: 2 * m * n * k
    return 2.0 * m * n * k


def get_mha_gflops(config, bs, avg_context_len):
    # 计算多头注意力机制中Q矩阵投影的浮点运算次数
    q_proj = gemm_flops(
        bs, config.hidden_size, config.num_attention_heads * config.head_dim
    )
    # 计算多头注意力机制中K矩阵投影的浮点运算次数
    k_proj = gemm_flops(
        bs, config.hidden_size, config.num_key_value_heads * config.head_dim
    )
    # 计算多头注意力机制中V矩阵投影的浮点运算次数
    v_proj = gemm_flops(
        bs, config.hidden_size, config.num_key_value_heads * config.head_dim
    )
    # 计算多头注意力机制中输出投影的浮点运算次数
    o_proj = gemm_flops(
        bs, config.num_attention_heads * config.head_dim, config.hidden_size
    )
    # 计算注意力核心计算的浮点运算次数（Q*K^T 和 A*V）
    attn_core = gemm_flops(
        bs, config.num_attention_heads * config.head_dim, avg_context_len
    ) + gemm_flops(bs, avg_context_len, config.num_attention_heads * config.head_dim)
    # 返回注意力核心计算GFLOPs和投影层GFLOPs
    return attn_core / 1e9, (q_proj + k_proj + v_proj + o_proj) / 1e9


def get_mla_absorb_gflops(config, bs, avg_context_len):
    # 计算MLA中Q矩阵降维投影的浮点运算次数
    q_down_proj = gemm_flops(bs, config.hidden_size, config.q_lora_rank)
    # 计算MLA中Q矩阵升维投影的浮点运算次数
    q_up_proj = gemm_flops(
        bs, config.q_lora_rank, config.num_attention_heads * config.qk_head_dim
    )

    # 计算MLA中KV矩阵降维投影的浮点运算次数
    kv_down_proj = gemm_flops(
        bs, config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim
    )

    # 计算MLA中Q与WK矩阵相乘的浮点运算次数
    bmm_q_wk = config.num_attention_heads * gemm_flops(
        bs, config.qk_nope_head_dim, config.kv_lora_rank
    )
    # 计算MLA中O与WV矩阵相乘的浮点运算次数
    bmm_o_wv = config.num_attention_heads * gemm_flops(
        bs, config.kv_lora_rank, config.v_head_dim
    )

    # 计算MLA中输出投影的浮点运算次数
    o_proj = gemm_flops(
        bs, config.num_attention_heads * config.v_head_dim, config.hidden_size
    )

    # 计算MLA注意力核心计算的浮点运算次数
    attn_core = gemm_flops(
        bs,
        config.num_attention_heads * (config.kv_lora_rank + config.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(
        bs, avg_context_len, config.num_attention_heads * config.kv_lora_rank
    )

    # 返回MLA注意力核心计算GFLOPs和投影层GFLOPs
    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj) / 1e9,
    )


def get_gqla_absorb_gflops(config, bs, avg_context_len):
    # 计算GQLA中Q矩阵降维投影的浮点运算次数
    q_down_proj = gemm_flops(bs, config.hidden_size, config.q_lora_rank)
    # 计算GQLA中Q矩阵升维投影的浮点运算次数
    q_up_proj = gemm_flops(
        bs, config.q_lora_rank, config.num_attention_heads * config.qk_head_dim
    )

    # 计算GQLA中KV矩阵降维投影的浮点运算次数
    kv_down_proj = gemm_flops(
        bs, config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim
    )

    # 计算GQLA中Q与WK矩阵相乘的浮点运算次数（分组处理）
    bmm_q_wk = (
        2
        * config.num_attention_heads
        / 2
        * gemm_flops(bs, config.qk_nope_head_dim, config.kv_lora_rank / 2)
    )
    # 计算GQLA中O与WV矩阵相乘的浮点运算次数（分组处理）
    bmm_o_wv = (
        2
        * config.num_attention_heads
        / 2
        * gemm_flops(bs, config.kv_lora_rank / 2, config.v_head_dim)
    )

    # 计算GQLA中输出投影的浮点运算次数
    o_proj = gemm_flops(
        bs, config.num_attention_heads * config.v_head_dim, config.hidden_size
    )

    # 计算GQLA注意力核心计算的浮点运算次数
    attn_core = gemm_flops(
        bs,
        config.num_attention_heads
        * (config.kv_lora_rank / 2 + config.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(
        bs, avg_context_len, config.num_attention_heads * config.kv_lora_rank / 2
    )

    # 返回GQLA注意力核心计算GFLOPs和投影层GFLOPs
    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj) / 1e9,
    )


def get_mla_noabsorb_gflops(config, bs, avg_context_len):
    # 计算MLA(非吸收版本)中Q矩阵降维投影的浮点运算次数
    q_down_proj = gemm_flops(bs, config.hidden_size, config.q_lora_rank)
    # 计算MLA(非吸收版本)中Q矩阵升维投影的浮点运算次数
    q_up_proj = gemm_flops(
        bs, config.q_lora_rank, config.num_attention_heads * config.qk_head_dim
    )

    # 计算MLA(非吸收版本)中KV矩阵降维投影的浮点运算次数
    kv_down_proj = gemm_flops(
        bs, config.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim
    )
    # 计算MLA(非吸收版本)中KV矩阵升维投影的浮点运算次数
    kv_up_proj = gemm_flops(
        bs,
        config.kv_lora_rank,
        config.num_attention_heads * (config.v_head_dim + config.qk_nope_head_dim),
    )

    # 计算MLA(非吸收版本)中输出投影的浮点运算次数
    o_proj = gemm_flops(
        bs, config.num_attention_heads * config.v_head_dim, config.hidden_size
    )

    # 计算MLA(非吸收版本)注意力核心计算的浮点运算次数
    attn_core = gemm_flops(
        bs,
        config.num_attention_heads
        * (config.qk_nope_head_dim + config.qk_rope_head_dim),
        avg_context_len,
    ) + gemm_flops(bs, avg_context_len, config.num_attention_heads * config.v_head_dim)

    # 返回MLA(非吸收版本)注意力核心计算GFLOPs和投影层GFLOPs
    return (
        attn_core / 1e9,
        (q_down_proj + q_up_proj + kv_down_proj + kv_up_proj + o_proj) / 1e9,
    )


def get_attn_gflops(config: ModelConfig, avg_context_len: int, absorb=True):
    # 根据注意力类型选择相应的FLOPs计算方法
    if config.attn_type == "MHA/GQA":
        # 多头注意力/GQA的FLOPs计算
        return get_mha_gflops(config, bs=1, avg_context_len=avg_context_len)
    elif config.attn_type == "MLA":
        # MLA注意力的FLOPs计算，根据是否吸收KV来选择不同函数
        if absorb:
            return get_mla_absorb_gflops(config, bs=1, avg_context_len=avg_context_len)
        return get_mla_noabsorb_gflops(config, bs=1, avg_context_len=avg_context_len)


def get_moe_gflops(config: ModelConfig):
    # 计算MoE层的激活专家数（共享专家数+每token专家数）
    act = config.num_shared_experts + config.num_experts_per_tok
    # 计算MoE层的GFLOPs（每个激活专家有3个矩阵乘法操作）
    return act * 3.0 * gemm_flops(1, config.hidden_size, config.intermediate_size) / 1e9