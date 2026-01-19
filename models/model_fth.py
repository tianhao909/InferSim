import math

from comm.comm import Comm
from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import gpu_map
from kvcache.kvcache import get_kvcache_size
from layers.attn import create_attention
from layers.moe import MoE
from params.params import get_attn_params_size, get_expert_params_size


class Model:
    def __init__(self, args, config):
        self.gpu = gpu_map[args.device_type]  # 根据设备类型获取GPU配置
        self.args = args  # 保存命令行参数
        self.config = config  # 保存模型配置

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))  # 打印标题：模型权重
        attn_params_bytes = get_attn_params_size(self.config, self.args.use_fp8_gemm)  # 计算注意力层参数字节数
        expert_params_bytes = get_expert_params_size(  # 计算单个专家参数字节数
            self.config, self.args.use_fp8_gemm
        )
        print(  # 打印单个注意力层参数大小（MB）
            "{:<40} {:<10.2f}".format(
                "One attn params size (MB):", attn_params_bytes / 1024 / 1024
            )
        )
        print(  # 打印单个专家参数大小（MB）
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", expert_params_bytes / 1024 / 1024
            )
        )
        # 计算每张GPU上的模型参数总量（包括共享专家和路由专家）
        params_per_gpu = attn_params_bytes + expert_params_bytes * (
            self.config.num_shared_experts
            + self.config.num_routed_experts / self.args.world_size
        )
        params_per_gpu = params_per_gpu / 1024 / 1024 / 1024  # 转换为GB
        params_per_gpu *= self.config.num_hidden_layers  # 乘以层数得到总参数量
        # 计算可用KV缓存内存（总显存减去模型参数、运行时开销和编码器预留）
        self.kvcache_mem = (
            self.gpu.mem - params_per_gpu - 15 - 5
        )  # 15GB for runtime, 5GB for encoder（15GB用于运行时，5GB用于编码器）
        print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))  # 打印每GPU参数大小（GB）

    def print_kvcache_info(self):
        print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))  # 打印标题：KV缓存
        print("{:<40} {:<10.2f}".format("KV cache space (GB):", self.kvcache_mem))  # 打印KV缓存空间（GB）
        context_len = self.args.target_isl + self.args.target_osl  # 计算上下文长度（输入+输出）

        if self.args.decode_bs is None:  # 如果未指定解码batch size
            target_bs = math.ceil(self.args.target_tgs * self.args.target_tpot / 1000)  # 根据目标吞吐和延迟估算batch size
        else:
            target_bs = self.args.decode_bs  # 使用指定的解码batch size
        print("{:<40} {:<10}".format("Input seq len:", self.args.target_isl))  # 打印输入序列长度
        print("{:<40} {:<10}".format("Output seq len:", self.args.target_osl))  # 打印输出序列长度
        print("{:<40} {:<10}".format("Target decode batchsize:", target_bs))  # 打印目标解码batch size
        # 计算目标每token的KV缓存字节数
        target_kvcache_bytes = (
            self.kvcache_mem * 1024 * 1024 * 1024 / target_bs / context_len
        )
        kvcache_bytes = get_kvcache_size(self.config, self.args.use_fp8_kv)  # 获取当前每token KV缓存大小
        print(  # 打印目标每token KV缓存大小（KB）
            "{:<40} {:<10.2f}".format(
                "Target per-token KV cache size (KB):", target_kvcache_bytes / 1024
            )
        )
        print(  # 打印当前每token KV缓存大小（KB）
            "{:<40} {:<10.2f}".format(
                "Current per-token KV cache size (KB):", kvcache_bytes / 1024
            )
        )
        if kvcache_bytes > target_kvcache_bytes:  # 如果当前KV缓存超过目标
            print("!Error: need smaller kvcache")  # 报错：需要更小的KV缓存
        self.kvcache_bytes = kvcache_bytes  # 保存当前每token KV缓存大小
        self.target_bs = target_bs  # 保存目标batch size

    def print_flops_info(self):
        print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))  # 打印标题：FLOPs计算
        print("{:<40} {:<10}".format("Num hidden layers:", self.config.num_hidden_layers))  # 打印隐藏层数量
        # 计算平均上下文长度（输入长度 + 一半输出长度）
        self.avg_context_len = int(self.args.target_isl + self.args.target_osl / 2)
        # 获取注意力核心和其他部分的GFLOPs
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self.avg_context_len, absorb=True
        )
        moe_gflops = get_moe_gflops(self.config)  # 获取MoE部分的GFLOPs
        print("{:<40} {:<10.2f}".format("Per-token per-layer attn core (GFLOPs):", attn_core_gflops))  # 每token每层注意力核心计算量
        print("{:<40} {:<10.2f}".format("Per-token per-layer MoE/FFN (GFLOPs):", moe_gflops))  # 每token每层MoE/FFN计算量
        print("{:<40} {:<10.2f}".format("Per-token per-layer others (GFLOPs):", other_gflops))  # 每token每层其他部分计算量
        print("{:<40} {:<10.2f}".format("Per-token attn core (GFLOPs):", attn_core_gflops * self.config.num_hidden_layers))  # 总注意力核心计算量
        print("{:<40} {:<10.2f}".format("Per-token MoE (GFLOPs):", moe_gflops * self.config.num_hidden_layers))  # 总MoE计算量
        print("{:<40} {:<10.2f}".format("Per-token others (GFLOPs):", other_gflops * self.config.num_hidden_layers))  # 其他部分总计算量
        print("{:<40} {:<10.2f}".format("Per-token total (GFLOPs):", (attn_core_gflops + moe_gflops + other_gflops) * self.config.num_hidden_layers))  # 每token总计算量

    def prefill(self):
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))  # 打印标题：Prefill阶段
        print("{:<40} {:<10}".format("Max prefill tokens:", self.args.max_prefill_tokens))  # 打印最大prefill token数
        # 创建注意力模块
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )
        # 计算注意力核心部分的prefill时间
        attn_core_time = attn.prefill_attn_core(
            self.args.target_isl, self.kvcache_bytes, self.args.device_type
        )
        # 计算注意力其他部分的prefill时间
        attn_other_time = attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )
        # 若最大prefill tokens超过目标isl，则按比例扩展核心计算时间
        attn_core_time *= math.ceil(self.args.max_prefill_tokens / self.args.target_isl)

        # 创建MoE模块
        moe = MoE(self.config, self.args.use_fp8_gemm)
        # 计算MoE在prefill阶段的时间
        moe_time = moe.prefill_moe(
            self.args.max_prefill_tokens, self.args.device_type, self.args.world_size
        )

        # 创建通信模块
        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.prefill_comm(self.args.max_prefill_tokens)  # 获取prefill阶段两次通信耗时
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))  # 打印MoE前通信时间（微秒）
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))  # 打印MoE后通信时间（微秒）

        num_tokens = self.args.max_prefill_tokens  # 当前处理的token数量
        if self.args.enable_tbo:  # 如果启用TBO（Tensor Batch Optimization）
            num_tokens *= 2  # token数翻倍
            # 计算TTFT：各阶段最大时间取并行，考虑SM利用率
            ttft = max(
                (attn_core_time + attn_other_time) / self.args.sm_ratio, comm_time1
            )
            ttft += max(
                (attn_core_time + attn_other_time) / self.args.sm_ratio, comm_time2
            )
            ttft += max(moe_time / self.args.sm_ratio, comm_time1)
            ttft += max(moe_time / self.args.sm_ratio, comm_time2)
        else:  # 未启用TBO，串行叠加各阶段时间
            ttft = attn_core_time
            ttft += moe_time
            ttft += attn_other_time
            ttft += comm_time1 + comm_time2
        ttft *= self.config.num_hidden_layers  # 乘以层数
        ttft *= 1000  # 转换为毫秒
        ttft += 30  # 加上调度器开销（约30ms）

        print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))  # 打印首token延迟（毫秒）
        print("{:<40} {:<10.0f}".format("Throughput (TGS:tok/GPU/s):", num_tokens / (ttft / 1000)))  # 打印吞吐量（token/GPU/秒）

    def decoding(self):
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))  # 打印标题：解码阶段
        # 创建注意力模块
        attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )
        # 计算解码阶段注意力核心耗时
        attn_core_time = attn.decode_attn_core(
            self.target_bs,
            self.avg_context_len,
            self.kvcache_bytes,
            self.args.device_type,
        )
        # 计算解码阶段注意力其他部分耗时
        attn_other_time = attn.decode_attn_others(self.target_bs, self.args.device_type)

        # 创建MoE模块
        moe = MoE(self.config, self.args.use_fp8_gemm)
        # 计算解码阶段MoE耗时
        moe_time = moe.decode_moe(
            self.target_bs, self.args.device_type, self.args.world_size
        )

        # 创建通信模块
        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )
        comm_time1, comm_time2 = comm.decode_comm(self.target_bs)  # 获取解码阶段两次通信耗时
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))  # 打印MoE前通信时间（微秒）
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))  # 打印MoE后通信时间（微秒）

        num_tokens = self.target_bs  # 当前处理的token数（等于batch size）
        if self.args.enable_tbo:  # 如果启用TBO
            num_tokens *= 2  # token数翻倍
            # TPOT（每token延迟）取各阶段最大并行时间，并乘以2
            tpot = max(
                attn_core_time + attn_other_time, moe_time + comm_time1 + comm_time2
            )
            tpot *= 2
        else:  # 未启用TBO，串行叠加
            tpot = attn_core_time
            tpot += attn_other_time
            tpot += moe_time
            tpot += comm_time1 + comm_time2
        tpot *= self.config.num_hidden_layers  # 乘以层数
        tpot *= 1000  # 转换为毫秒
        tpot += 5  # 加上调度器开销（约5ms）

        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))  # 打印每token延迟（毫秒）
        print("{:<40} {:<10.0f}".format("Throughput (TGS):", num_tokens / tpot * 1000))  # 打印吞吐量（token/GPU/秒）
        if tpot > self.args.target_tpot:  # 如果实际延迟超过目标
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")  # 报错：延迟超限，需降低计算量加速
