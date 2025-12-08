# 从config.model_config模块导入ModelConfig类
from config.model_config import ModelConfig


# 定义计算MHA（Multi-Head Attention）KV缓存大小的函数，接受模型配置和是否使用fp8精度作为参数
def get_mha_kvcache_size(config: ModelConfig, use_fp8):  # 获取MHA类型的KV缓存大小
    # 计算KV缓存大小：2表示K和V两个缓存，乘以层数、键值头数和头维度
    kvcache_size = (  # KV缓存大小计算
        2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim  # 2*(层数)*(键值头数)*(头维度)
    )  # 计算基本的KV缓存大小
    # 如果不使用fp8精度，则缓存大小需要乘以2（因为fp8是8位浮点，而默认是16位浮点）
    if not use_fp8:  # 如果不使用fp8精度
        kvcache_size *= 2  # 缓存大小翻倍（因为默认精度占用更多空间）
    return kvcache_size  # 返回计算得到的KV缓存大小


# 定义计算MLA（Multi-Layer Attention）KV缓存大小的函数，接受模型配置和是否使用fp8精度作为参数
def get_mla_kvcache_size(config: ModelConfig, use_fp8):  # 获取MLA类型的KV缓存大小
    # 计算KV缓存大小：层数乘以(kv_lora_rank + qk_rope_head_dim)的和
    kvcache_size = config.num_hidden_layers * (  # KV缓存大小 = 层数 * (kv_lora秩 + qk_rope头维度)
        config.kv_lora_rank + config.qk_rope_head_dim  # kv_lora秩 + qk_rope头维度
    )  # 计算MLA的KV缓存大小
    # 如果不使用fp8精度，则缓存大小需要乘以2
    if not use_fp8:  # 如果不使用fp8精度
        kvcache_size *= 2  # 缓存大小翻倍（因为默认精度占用更多空间）
    return kvcache_size  # 返回计算得到的KV缓存大小


# 定义获取KV缓存大小的通用函数，根据注意力类型选择相应的计算方法
def get_kvcache_size(config: ModelConfig, use_fp8):  # 根据注意力机制类型获取对应的KV缓存大小
    # 如果注意力类型是MHA/GQA，则调用get_mha_kvcache_size函数计算
    if config.attn_type == "MHA/GQA":  # 当注意力类型为MHA/GQA时
        return get_mha_kvcache_size(config, use_fp8)  # 调用MHA KV缓存大小计算函数
    # 如果注意力类型是MLA，则调用get_mla_kvcache_size函数计算
    elif config.attn_type == "MLA":  # 当注意力类型为MLA时
        return get_mla_kvcache_size(config, use_fp8)  # 调用MLA KV缓存大小计算函数