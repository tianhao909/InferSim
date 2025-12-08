from flops.flops import gemm_flops  # 从 flops.flops 导入用于计算 GEMM 浮点运算量的函数
from hardware.gpu import gpu_map  # 从 hardware.gpu 导入 GPU 参数映射表
from mfu.mfu import get_attn_decode_mfu, get_attn_prefill_mfu, get_gemm_mfu  # 从 mfu.mfu 导入用于估算 MFU 的函数


def get_gemm_mfu_and_latency(m, k, n, device_type, use_fp8_gemm):  # 计算一次 GEMM 的 MFU 和估算延迟（秒）
    gpu = gpu_map[device_type]  # 根据设备类型获取对应 GPU 参数
    gflops = gemm_flops(m, k, n) / 1e9  # 计算本次 GEMM 的 GFLOPs（将 FLOPs 转为 GFLOPs）
    mfu = get_gemm_mfu(device_type, m, k, n)  # 根据设备和矩阵形状估算 GEMM 的 MFU
    latency = gflops / (gpu.fp16_tflops * 1024 * mfu)  # 使用 FP16 理论算力估算延迟：GFLOPs / (TFLOPs*1024*MFU)
    if use_fp8_gemm:  # 如果使用 FP8 GEMM，则用 FP8 理论算力替换计算
        latency = gflops / (gpu.fp8_tflops * 1024 * mfu)  # 使用 FP8 理论算力估算延迟
    # print(f"Debug: gemm m:{m} k:{k} n:{n}")  # 调试：打印 GEMM 的 m,k,n 形状
    return latency  # 返回估算的延迟（秒）


class MHA:  # 多头注意力/组查询注意力实现类
    def __init__(self, config, use_fp8_gemm, use_fp8_kv):  # 初始化，保存配置和精度开关
        self.use_fp8_gemm = use_fp8_gemm  # 是否使用 FP8 的 GEMM
        self.use_fp8_kv = use_fp8_kv  # 是否使用 FP8 的 KV 缓存
        self.config = config  # 模型配置

    def get_attn_core_gflops(self, bs, kv_len):  # 计算注意力核心部分（QK^T 与 AV）的 GFLOPs
        attn_core = (
            gemm_flops(
                bs, self.config.num_attention_heads * self.config.head_dim, kv_len
            )  # 计算 QK^T 的 FLOPs（简化为 GEMM 形式）
            * 2  # 乘以 2 以同时计入 AV 的 FLOPs
        )  #
        return attn_core / 1e9  # 转为 GFLOPs 并返回

    def decode_attn_core(self, bs, kv_len, kvcache_bytes, device_type):  # 解码阶段的注意力核心计算与 KV 读取延迟估算
        gpu = gpu_map[device_type]  # 获取 GPU 参数
        attn_core_gflops = self.get_attn_core_gflops(1, kv_len)  # 以 bs=1 的 head 计算单位 token 的核心 GFLOPs
        attn_core_mfu = get_attn_decode_mfu(
            self.config, bs, kv_len, device_type, self.use_fp8_kv
        )  # 估算解码阶段注意力核心的 MFU（考虑 KV 精度）
        attn_core_time = (
            bs * attn_core_gflops / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )  # 计算注意力核心计算耗时（秒）
        kv_load_time = (
            kvcache_bytes
            * kv_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )  # 估算从显存读取 KV cache 的耗时（秒），按层均摊

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))  # 打印注意力核心 MFU
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )  # 打印注意力核心计算延迟（微秒）
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))  # 打印 KV 读取延迟（微秒）

        return max(attn_core_time, kv_load_time)  # 返回计算耗时与加载耗时中的较大值（瓶颈）

    def decode_attn_others(self, bs, device_type):  # 解码阶段注意力模块中除核心外的 GEMM 延迟估算
        qkv_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.hidden_size,
            n=(self.config.num_attention_heads + self.config.num_key_value_heads * 2)
            * self.config.head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )  # QKV 线性投影的 GEMM 延迟（秒）
        print("{:<40} {:<10.2f}".format("QKV_proj latency (us):", qkv_proj * 1e6))  # 打印 QKV 投影延迟（微秒）

        o_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.num_attention_heads * self.config.head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )  # 输出投影 O 的 GEMM 延迟（秒）
        print("{:<40} {:<10.2f}".format("O_proj latency (us):", o_proj * 1e6))  # 打印 O 投影延迟（微秒）
        return qkv_proj + o_proj  # 返回两部分延迟之和（秒）

    def prefill_attn_core(self, seq_len, kvcache_bytes, device_type):  # 预填充阶段的注意力核心计算与 KV 读取延迟估算
        gpu = gpu_map[device_type]  # 获取 GPU 参数
        attn_core_gflops = self.get_attn_core_gflops(1, seq_len)  # 计算序列长度下的单位 GFLOPs
        attn_core_mfu = get_attn_prefill_mfu(self.config, seq_len, device_type)  # 估算预填充阶段注意力核心 MFU
        attn_core_time = (
            seq_len * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )  # 计算注意力核心耗时（秒），1.8 为经验并行/重叠因子
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )  # 估算 KV 写入/读取带宽开销（秒），按层均摊

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))  # 打印注意力核心 MFU
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )  # 打印注意力核心延迟（微秒）
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))  # 打印 KV 加载延迟（微秒）

        return max(attn_core_time, kv_load_time)  # 返回计算耗时与加载耗时的较大值

    def prefill_attn_others(self, seq_len, device_type):  # 预填充阶段其它 GEMM 延迟（与解码阶段相同形状）
        return self.decode_attn_others(seq_len, device_type)  # 直接复用解码阶段的计算


class MLA(MHA):  # 多头长连接注意力（MLA）实现，继承 MHA
    def __init__(self, config, use_fp8_gemm, use_fp8_kv):  # 初始化 MLA
        self.use_fp8_gemm = use_fp8_gemm  # 是否使用 FP8 GEMM
        self.use_fp8_kv = use_fp8_kv  # 是否使用 FP8 KV
        self.config = config  # 模型配置

    def get_attn_core_gflops_absorb(self, bs, kv_len):  # 计算吸收（absorb）权重路径下的注意力核心 GFLOPs
        attn_core = gemm_flops(
            bs,
            self.config.num_attention_heads
            * (self.config.kv_lora_rank + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(
            bs, kv_len, self.config.num_attention_heads * self.config.kv_lora_rank
        )  # 两个 GEMM：Q*(W_k_absorb) 与 O*(W_v_absorb) 的 FLOPs 求和
        return attn_core / 1e9  # 转为 GFLOPs 并返回

    def get_attn_core_gflops_noabsorb(self, bs, kv_len):  # 计算非吸收（no-absorb）路径下的注意力核心 GFLOPs
        attn_core = gemm_flops(
            bs,
            self.config.num_attention_heads
            * (self.config.qk_nope_head_dim + self.config.qk_rope_head_dim),
            kv_len,
        ) + gemm_flops(
            bs, kv_len, self.config.num_attention_heads * self.config.v_head_dim
        )  # 两个 GEMM：Q*(W_k_noabsorb) 与 O*(W_v_noabsorb) 的 FLOPs 求和
        return attn_core / 1e9  # 转为 GFLOPs 并返回

    def decode_attn_core(self, bs, kv_len, kvcache_bytes, device_type):  # MLA 解码阶段注意力核心耗时估算
        gpu = gpu_map[device_type]  # 获取 GPU 参数
        attn_core_gflops = self.get_attn_core_gflops_absorb(1, kv_len)  # 使用吸收路径的 GFLOPs
        attn_core_mfu = get_attn_decode_mfu(
            self.config, bs, kv_len, device_type, self.use_fp8_kv
        )  # 估算解码阶段的注意力核心 MFU
        attn_core_time = (
            bs * attn_core_gflops / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )  # 计算注意力核心计算耗时（秒）
        kv_load_time = (
            kvcache_bytes
            * kv_len
            * bs
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )  # 估算从显存加载 KV 的耗时（秒），按层均摊

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))  # 打印注意力核心 MFU
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )  # 打印注意力核心延迟（微秒）
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))  # 打印 KV 加载延迟（微秒）

        return max(attn_core_time, kv_load_time)  # 返回瓶颈耗时

    def decode_attn_others(self, bs, device_type):  # MLA 解码阶段其余 GEMM 延迟估算
        q_down_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.q_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )  # Q 的下投影（LoRA 降维）GEMM 延迟（秒）
        print("{:<40} {:<10.2f}".format("Q_down_proj latency (us):", q_down_proj * 1e6))  # 打印 Q 下投影延迟（微秒）

        q_up_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.q_lora_rank,
            n=self.config.num_attention_heads * self.config.qk_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )  # Q 的上投影（LoRA 升维）GEMM 延迟（秒）
        print("{:<40} {:<10.2f}".format("Q_up_proj latency (us):", q_up_proj * 1e6))  # 打印 Q 上投影延迟（微秒）

        kv_down_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.hidden_size,
            n=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )  # KV 的下投影（含 RoPE 相关维度）GEMM 延迟（秒）
        print(
            "{:<40} {:<10.2f}".format("KV_down_proj latency (us):", kv_down_proj * 1e6)
        )  # 打印 KV 下投影延迟（微秒）

        bmm_q_wk = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.num_attention_heads * self.config.qk_nope_head_dim,
            n=self.config.kv_lora_rank,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )  # Q 与 W_k 的批量矩阵乘（无位置编码部分）延迟（秒）
        print("{:<40} {:<10.2f}".format("bmm_q_wk latency (us):", bmm_q_wk * 1e6))  # 打印 bmm_q_wk 延迟（微秒）

        bmm_o_wv = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.num_attention_heads * self.config.kv_lora_rank,
            n=self.config.v_head_dim,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )  # O 与 W_v 的批量矩阵乘延迟（秒）
        print("{:<40} {:<10.2f}".format("bmm_o_wv latency (us):", bmm_o_wv * 1e6))  # 打印 bmm_o_wv 延迟（微秒）

        o_proj = get_gemm_mfu_and_latency(
            m=bs,
            k=self.config.num_attention_heads * self.config.v_head_dim,
            n=self.config.hidden_size,
            device_type=device_type,
            use_fp8_gemm=self.use_fp8_gemm,
        )  # 输出投影 O 的 GEMM 延迟（秒）
        print("{:<40} {:<10.2f}".format("O_proj latency (us):", o_proj * 1e6))  # 打印 O 投影延迟（微秒）
        return q_down_proj + q_up_proj + kv_down_proj + bmm_q_wk + bmm_o_wv + o_proj  # 返回所有子项延迟之和（秒）

    def prefill_attn_core(self, seq_len, kvcache_bytes, device_type):  # MLA 预填充阶段注意力核心耗时估算
        gpu = gpu_map[device_type]  # 获取 GPU 参数
        attn_core_gflops = self.get_attn_core_gflops_noabsorb(1, seq_len)  # 使用非吸收路径的 GFLOPs
        attn_core_mfu = get_attn_prefill_mfu(self.config, seq_len, device_type)  # 估算预填充阶段的注意力核心 MFU
        attn_core_time = (
            seq_len * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * attn_core_mfu)
        )  # 计算注意力核心耗时（秒），含 1.8 并行/重叠因子
        kv_load_time = (
            kvcache_bytes
            * seq_len
            / self.config.num_hidden_layers
            / 1024
            / 1024
            / 1024
            / gpu.mem_bw
        )  # 估算 KV 带宽耗时（秒），按层均摊

        print("{:<40} {:<10.2f}".format("Attn core MFU:", attn_core_mfu))  # 打印注意力核心 MFU
        print(
            "{:<40} {:<10.2f}".format("Attn core latency (us):", attn_core_time * 1e6)
        )  # 打印注意力核心延迟（微秒）
        print("{:<40} {:<10.2f}".format("KV loading latency (us):", kv_load_time * 1e6))  # 打印 KV 加载延迟（微秒）

        return max(attn_core_time, kv_load_time)  # 返回瓶颈耗时


def create_attention(config, use_fp8_gemm, use_fp8_kv):  # 根据配置创建对应的注意力实现
    if config.attn_type == "MHA/GQA":  # 如果是标准多头注意力或组查询注意力
        return MHA(config, use_fp8_gemm, use_fp8_kv)  # 返回 MHA 实例
    elif config.attn_type == "MLA":  # 如果是 MLA
        return MLA(config, use_fp8_gemm, use_fp8_kv)  # 返回 MLA 实例