from config.model_config import ModelConfig  # 导入模型配置类
from hardware.gpu import GPU  # 导入GPU硬件类


def get_mha_params_size(config: ModelConfig, use_fp8: bool):
    wq = config.hidden_size * config.num_attention_heads * config.head_dim  # Q权重参数量：隐藏层大小 × 注意力头数 × 头维度
    wk = config.hidden_size * config.num_key_value_heads * config.head_dim  # K权重参数量：隐藏层大小 × KV头数 × 头维度
    wv = config.hidden_size * config.num_key_value_heads * config.head_dim  # V权重参数量：隐藏层大小 × KV头数 × 头维度
    wo = config.hidden_size * config.num_attention_heads * config.head_dim  # 输出权重参数量：隐藏层大小 × 注意力头数 × 头维度
    if use_fp8:  # 如果使用FP8量化
        return wq + wk + wv + wo  # 则只计算一次权重（单精度存储）
    return 2 * (wq + wk + wv + wo)  # 否则按全精度计算，每个参数占两个字节（如FP16）

# MLA
    # dpsk v3 参数推导参考：
        # https://zhuanlan.zhihu.com/p/21455638257 
        # https://yangwenbo.com/articles/deepseek-v3-parameter-size.html
    # "hidden_size": 7168,
    # "num_key_value_heads": 128,
    # "v_head_dim": 128,
    # "kv_lora_rank": 512,

    # "num_attention_heads": 128,
    # "q_lora_rank": 1536,

    # "qk_nope_head_dim": 128,
    # "qk_rope_head_dim": 64,

    # "num_hidden_layers": 61,
def get_mla_params_size(config: ModelConfig, use_fp8: bool):
    # 单层 MLA 中 Q 的 LoRA 参数量是：
        # = 7168 * 1536 + 1536 + 1536 * 128 * (128 + 64) = 48,760,320
        # = wq_down + wq_up
        # = (config.hidden_size * config.q_lora_rank) + (config.q_lora_rank * config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim))
        # = (config.hidden_size * config.q_lora_rank) + (config.q_lora_rank * config.num_attention_heads * (config.qk_head_dim))
    wq_down = config.hidden_size * config.q_lora_rank  # Q的LoRA下投影矩阵参数量
    wq_up = config.q_lora_rank * config.num_attention_heads * config.qk_head_dim  # Q的LoRA上投影矩阵参数量
    # 单层 MLA 中 KV 的 LoRA 参数量是：
        # = 7168 * (512 + 64) + 512 + 512 * 128 * (128 + 128) = 20,906,496
        # = wkv_down + 512 + wkv_up (TODO fth: 512是啥，不太懂)
        # = config.hidden_size *（config.kv_lora_rank + config.qk_rope_head_dim) + 512 + config.kv_lora_rank * config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)
        # = (config.hidden_size * config.kv_lora_rank) + (config.kv_lora_rank * config.num_key_value_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim))
    wkv_down = config.hidden_size * config.kv_lora_rank  # KV的LoRA下投影矩阵参数量
    wkv_up = (  # KV的LoRA上投影矩阵参数量
        config.kv_lora_rank
        * config.num_attention_heads
        * (config.qk_nope_head_dim + config.v_head_dim)
    )
    # 单层 MLA 中 WO 的参数量是
        # 128 * 128 * 7168 = 117,440,512
        # config.num_attention_heads * config.v_head_dim * config.hidden_size
    wo = config.hidden_size * config.num_attention_heads * config.v_head_dim  # 输出权重参数量
    if use_fp8:  # 如果使用FP8量化
        return wq_down + wq_up + wkv_down + wkv_up + wo  # 返回所有参数之和（单精度）
    return 2 * (wq_down + wq_up + wkv_down + wkv_up + wo)  # 否则乘以2（如FP16）

    # 另外：
        # pre 和 post attention layernorm 的参数量是：
        # 7168 * 2 = 14336
        # 所以 DeepSeek V3 的 MLA 部分共 61 层的总参数量是：
        # (48,760,320 + 20,906,496 + 117,440,512 + 14336) * 61 = 11,414,421,504 (~11B)


def get_gdn_params_size(config: ModelConfig, use_fp8: bool):
    wq = config.hidden_size * config.linear_num_key_heads * config.linear_key_head_dim  # Q线性注意力权重参数量
    wk = wq  # K权重与Q相同
    wv = (  # V权重参数量
        config.hidden_size
        * config.linear_num_value_heads
        * config.linear_value_head_dim
    )
    wz = wv  # Z权重与V相同
    wa = config.hidden_size * config.linear_num_value_heads  # A门控参数量
    wb = wa  # B门控参数量与A相同
    s = wq + wk + wv + wz + wa + wb  # 所有主要权重参数总和
    wconv = (  # 卷积核权重参数量（第一部分）
        config.linear_num_key_heads
        * config.linear_key_head_dim
        * config.linear_conv_kernel_dim
    )
    wconv += (  # 卷积核权重参数量（第二部分）
        config.linear_num_key_heads
        * config.linear_key_head_dim
        * config.linear_conv_kernel_dim
    )
    wconv += (  # 卷积核权重参数量（第三部分）
        config.linear_num_value_heads
        * config.linear_value_head_dim
        * config.linear_conv_kernel_dim
    )
    if use_fp8:  # 如果使用FP8量化
        return s + wconv  # 返回主参数加卷积参数（单精度）
    return 2 * s + wconv  # 否则主参数×2，卷积仍为单精度（假设卷积不加倍）


def get_attn_params_size(config: ModelConfig, use_fp8: bool):
    if config.attn_type == "MHA/GQA":  # 如果注意力类型是MHA或GQA
        return get_mha_params_size(config, use_fp8)  # 调用MHA参数计算函数
    elif config.attn_type == "MLA":  # 如果是MLA结构
        return get_mla_params_size(config, use_fp8)  # 调用MLA参数计算函数


def get_linear_attn_params_size(config: ModelConfig, use_fp8: bool):
    return get_gdn_params_size(config, use_fp8)  # 获取线性注意力（GD-Nets风格）参数总量

# Deepseek MoE
    # "num_hidden_layers": 61,
    # "hidden_size": 7168,
    # "moe_intermediate_size": 2048,  // 路由专家 MLP 的中间维度
    # "n_shared_experts": 1,          // 共享专家数量
    # "n_routed_experts": 256,        // 路由专家数量
    # "first_k_dense_replace": 3,     // 前几层使用dense替换MoE
    # "intermediate_size": 18432,     // 前3层 (9*moe_intermediate_size)
    
    # 每个专家的参数量是：
        # 7168 * 2048 * 3 = 44,040,192
        # config.hidden_size * config.moe_intermediate_size * 3
    # 路由 Gate 的参数量是：
        # 256 * 7168 + 256 = 1,835,264
    # 前 3 层 dense（固定激活 8 路由专家），前 3 层参数量是：
        # 44,040,192 * 9 * 3 = 1,189,085,184
    # 后 58 层稀疏（动态激活 8 路由专家），后 58 层参数量是：
        # (44,040,192 * 257 + 1,835,264) * 58 = 656,569,547,264
    # 所以 DeepSeek V3 的 MoE 部分的总参数量是：
        # 1,189,085,184 + 656,569,547,264 = 657,758,632,448 (~657B)
    # 每次计算激活 1 个共享专家，8 个路由专家，所以 DeepSeek V3 MoE 部分的激活参数量是：
        # 44,040,192 * 9 * 61 + 1,835,264 * 58 = 24,284,510,720 (~24B)
def get_expert_params_size(config: ModelConfig, use_fp8: bool):
    w = 3 * config.hidden_size * config.intermediate_size  # MoE专家前馈网络参数量（通常为W1, W2, W3三组权重）
    if not use_fp8:  # 如果不使用FP8量化
        w *= 2  # 参数量翻倍（例如从FP8到FP16）
    return w  # 返回专家参数总量


def load_attn_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU):
    size = get_attn_params_size(config, use_fp8)  # 获取注意力模块权重总大小（字节数）
    return size / 1024 / 1024 / 1024 / gpu.mem_bw  # 转换为GB并除以GPU内存带宽，得到加载时间（秒）


def load_moe_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU, num_gpus):
    size = get_expert_params_size(config, use_fp8)  # 获取单个专家权重大小
    size *= config.num_routed_experts / num_gpus  # 总专家数分配到多个GPU上，每卡需加载的部分
    return size / 1024 / 1024 / 1024 / gpu.mem_bw  # 转为GB后除以带宽，得加载时间（秒）
