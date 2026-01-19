import json


class ModelConfig:
    def __init__(
        self,
        config_path,
    ):
        d = dict()  # 初始化一个空字典用于存储配置
        with open(config_path, "r") as f:  # 打开配置文件路径，以只读模式读取
            d = json.load(f)  # 将JSON格式的配置文件内容加载到字典d中

        self.hidden_size = d["hidden_size"]  # 模型隐藏层维度
        self.num_hidden_layers = d["num_hidden_layers"]  # 模型的隐藏层数量

        self.is_hybrid_linear = d.get("full_attention_interval") is not None  # 判断是否使用混合注意力机制（全注意力与线性注意力交替）
        if self.is_hybrid_linear:  # 如果是混合线性注意力结构
            self.num_full_attn_layers = (
                self.num_hidden_layers // d["full_attention_interval"]  # 全注意力层的数量：每隔 full_attention_interval 层插入一次全注意力
            )
            self.num_linear_attn_layers = (
                self.num_hidden_layers - self.num_full_attn_layers  # 线性注意力层的数量：总层数减去全注意力层数
            )
            self.linear_conv_kernel_dim = d["linear_conv_kernel_dim"]  # 线性注意力中卷积核的维度
            self.linear_key_head_dim = d["linear_key_head_dim"]  # 线性注意力中键向量每个头的维度
            self.linear_num_key_heads = d["linear_num_key_heads"]  # 线性注意力中键向量的头数
            self.linear_value_head_dim = d["linear_value_head_dim"]  # 线性注意力中值向量每个头的维度
            self.linear_num_value_heads = d["linear_num_value_heads"]  # 线性注意力中值向量的头数

        self.attn_type = "MHA/GQA"  # 默认注意力类型为多头注意力（MHA）或分组查询注意力（GQA）
        if "kv_lora_rank" in d:  # 如果配置中包含 kv_lora_rank，则使用 MLA（Multi-Head Latent Attention）注意力机制
            self.attn_type = "MLA"

        # 注意力机制相关参数设置
        if self.attn_type == "MHA/GQA":  # 如果是 MHA/GQA 类型
            self.num_attention_heads = d["num_attention_heads"]  # 注意力头的数量
            self.num_key_value_heads = d["num_key_value_heads"]  # 键和值的头数量（用于GQA）
            if "head_dim" in d:  # 如果配置中指定了 head_dim
                self.head_dim = d["head_dim"]  # 使用指定的 head_dim
            else:
                self.head_dim = self.hidden_size // self.num_attention_heads  # 否则通过 hidden_size 和 头数计算得到
        elif self.attn_type == "MLA":  # 如果是 MLA 类型
            self.q_lora_rank = d["q_lora_rank"]  # 查询向量LoRA的秩
            self.qk_nope_head_dim = d["qk_nope_head_dim"]  # 无位置编码的QK头维度
            self.qk_rope_head_dim = d["qk_rope_head_dim"]  # 使用RoPE编码的QK头维度
            self.kv_lora_rank = d["kv_lora_rank"]  # 键值对LoRA的秩
            self.num_attention_heads = d["num_attention_heads"]  # 注意力头总数
            self.v_head_dim = d["v_head_dim"]  # 值向量每个头的维度
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # QK总头维度为两部分之和

        # FFN/MoE（前馈网络/专家混合模型）配置
        self.is_moe = True  # 默认启用MoE（Mixture of Experts）
        if "num_routed_experts" in d:  # 如果存在路由专家数量
            self.num_routed_experts = d["num_routed_experts"]  # 设置路由专家数量
        elif "num_experts" in d:  # 否则尝试从 num_experts 字段读取
            self.num_routed_experts = d["num_experts"]  # 设置专家总数
        else:
            self.is_moe = False  # 如果没有相关字段，则不使用MoE
            self.num_routed_experts = 1  # 单一专家（即普通FFN）

        if self.is_moe:  # 如果启用了MoE
            self.num_experts_per_tok = d["num_experts_per_tok"]  # 每个token激活的专家数量
            self.intermediate_size = d["moe_intermediate_size"]  # MoE中每个专家的中间层大小
            self.num_shared_experts = d.get("num_shared_experts", 0)  # 共享专家的数量，默认为0
        else:  # 如果未启用MoE（普通FFN）
            self.num_experts_per_tok = 1  # 每个token只使用一个“专家”（即标准FFN）
            self.intermediate_size = d["intermediate_size"]  # 标准FFN的中间层大小
            self.num_shared_experts = 0  # 无共享专家
