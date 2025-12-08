from config.model_config import ModelConfig  # 从 config.model_config 模块导入 ModelConfig 类
from hardware.gpu import GPU  # 从 hardware.gpu 模块导入 GPU 类


def get_mha_params_size(config: ModelConfig, use_fp8: bool):  # 定义函数，计算多头注意力（MHA/GQA）的参数大小
    wq = config.hidden_size * config.num_attention_heads * config.head_dim  # 计算查询权重矩阵大小
    wk = config.hidden_size * config.num_key_value_heads * config.head_dim  # 计算键权重矩阵大小
    wv = config.hidden_size * config.num_key_value_heads * config.head_dim  # 计算值权重矩阵大小
    wo = config.hidden_size * config.num_attention_heads * config.head_dim  # 计算输出权重矩阵大小
    if use_fp8:  # 如果使用 FP8 精度
        return wq + wk + wv + wo  # 返回权重之和（FP8 精度下不需要乘 2）
    return 2 * (wq + wk + wv + wo)  # 如果不是 FP8，则参数存储大小乘 2（通常表示 FP16 或 FP32）


def get_mla_params_size(config: ModelConfig, use_fp8: bool):  # 定义函数，计算 MLA（Mixed Lora Attention）的参数大小
    wq_down = config.hidden_size * config.q_lora_rank  # 查询权重下采样部分大小
    wq_up = config.q_lora_rank * config.num_attention_heads * config.qk_head_dim  # 查询权重上采样部分大小
    wkv_down = config.hidden_size * config.kv_lora_rank  # 键值权重下采样部分大小
    wkv_up = (  # 计算键值权重上采样部分大小
        config.kv_lora_rank
        * config.num_attention_heads
        * (config.qk_nope_head_dim + config.v_head_dim)  # 注意这里是 QK 不带位置编码的维度加 V 维度
    )
    wo = config.hidden_size * config.num_attention_heads * config.v_head_dim  # 输出权重矩阵大小
    if use_fp8:  # 如果使用 FP8 精度
        return wq_down + wq_up + wkv_down + wkv_up + wo  # 返回总大小（FP8 精度下不需要乘 2）
    return 2 * (wq_down + wq_up + wkv_down + wkv_up + wo)  # 否则存储大小乘 2


def get_attn_params_size(config: ModelConfig, use_fp8: bool):  # 获取注意力层的参数大小
    if config.attn_type == "MHA/GQA":  # 如果是多头注意力或分组查询注意力
        return get_mha_params_size(config, use_fp8)  # 调用 MHA 参数计算
    elif config.attn_type == "MLA":  # 如果是 MLA
        return get_mla_params_size(config, use_fp8)  # 调用 MLA 参数计算


def get_expert_params_size(config: ModelConfig, use_fp8: bool):  # 获取 MoE 专家层参数大小
    w = 3 * config.hidden_size * config.intermediate_size  # 一个专家的权重大小（3 个矩阵）
    if not use_fp8:  # 如果不是 FP8
        w *= 2  # 存储大小乘 2
    return w  # 返回权重大小


def load_attn_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU):  # 计算加载注意力权重所需时间
    size = get_attn_params_size(config, use_fp8)  # 获取注意力参数大小
    return size / 1024 / 1024 / 1024 / gpu.mem_bw  # 转为 GB，然后除以显存带宽得到加载时间


def load_moe_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU, num_gpus):  # 计算加载 MoE 权重所需时间
    size = get_expert_params_size(config, use_fp8)  # 获取单专家参数大小
    size *= config.num_routed_experts / num_gpus  # 按路由的专家数量和 GPU 数量分摊
    return size / 1024 / 1024 / 1024 / gpu.mem_bw  # 转为 GB，再除以显存带宽得到加载时间
