import json


class ModelConfig:
    def __init__(
        self,
        config_path,  # 配置文件路径
    ):
        d = dict()  # 初始化一个空字典用于存储配置
        with open(config_path, "r") as f:  # 打开配置文件进行读取
            d = json.load(f)  # 将 JSON 配置加载到字典 d 中
        self.attn_type = "MHA/GQA"  # 默认注意力类型为多头注意力（MHA）或分组查询注意力（GQA）
        if "kv_lora_rank" in d:  # 如果配置中包含 kv_lora_rank 字段
            self.attn_type = "MLA"  # 则注意力类型为 MLA（多头低秩注意力）

        self.hidden_size = d["hidden_size"]  # 从配置中读取隐藏层维度
        self.num_hidden_layers = d["num_hidden_layers"]  # 从配置中读取隐藏层数量

        # attn（注意力相关配置）
        if self.attn_type == "MHA/GQA":  # 如果是 MHA 或 GQA 类型
            self.num_attention_heads = d["num_attention_heads"]  # 注意力头数量
            self.num_key_value_heads = d["num_key_value_heads"]  # Key/Value 的头数量（用于 GQA）
            if "head_dim" in d:  # 如果配置中显式指定了每个头的维度
                self.head_dim = d["head_dim"]  # 直接使用配置中的 head_dim
            else:  # 否则根据 hidden_size 和注意力头数计算
                self.head_dim = self.hidden_size // self.num_attention_heads  # 每个头的维度 = 隐藏层维度 / 注意力头数
        elif self.attn_type == "MLA":  # 如果是 MLA 类型
            self.q_lora_rank = d["q_lora_rank"]  # Q 投影的 LoRA 秩
            self.qk_nope_head_dim = d["qk_nope_head_dim"]  # QK 中不带 RoPE 的部分维度
            self.qk_rope_head_dim = d["qk_rope_head_dim"]  # QK 中带 RoPE 的部分维度
            self.kv_lora_rank = d["kv_lora_rank"]  # KV 投影的 LoRA 秩
            self.num_attention_heads = d["num_attention_heads"]  # 注意力头数量
            self.v_head_dim = d["v_head_dim"]  # Value 头的维度
            self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # QK 总头维度 = nope + rope 部分

        # FFN/MoE（前馈网络或混合专家配置）
        self.is_moe = True  # 默认假设是 MoE（混合专家）结构
        if "num_routed_experts" in d:  # 如果配置中有 num_routed_experts 字段
            self.num_routed_experts = d["num_routed_experts"]  # 使用该字段作为路由专家数
        elif "num_experts" in d:  # 否则尝试使用 num_experts 字段
            self.num_routed_experts = d["num_experts"]  # 兼容旧配置
        else:  # 如果都没有，则不是 MoE
            self.is_moe = False  # 标记为非 MoE
            self.num_routed_experts = 1  # 路由专家数设为 1（即普通 FFN）

        if self.is_moe:  # 如果是 MoE 结构
            self.num_experts_per_tok = d["num_experts_per_tok"]  # 每个 token 激活的专家数量
            self.intermediate_size = d["moe_intermediate_size"]  # MoE 中每个专家的中间层维度
            self.num_shared_experts = d.get("num_shared_experts", 0)  # 共享专家数量，若未指定则默认为 0
        else:  # 如果不是 MoE（即普通 FFN）
            self.num_experts_per_tok = 1  # 每个 token 只使用 1 个“专家”（即普通 FFN）
            self.intermediate_size = d["intermediate_size"]  # 普通 FFN 的中间层维度
            self.num_shared_experts = 0  # 无共享专家
