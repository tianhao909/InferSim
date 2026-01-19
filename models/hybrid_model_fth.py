import math

from comm.comm import Comm
from flops.flops import get_attn_gflops, get_moe_gflops
from hardware.gpu import gpu_map
from kvcache.kvcache import get_kvcache_size, get_states_size
from layers.attn import create_attention
from layers.linear_attn import create_linear_attn
from layers.moe import MoE
from params.params import (get_attn_params_size, get_expert_params_size,
                           get_linear_attn_params_size)


class HybridModel:
    def __init__(self, args, config):
        self.gpu = gpu_map[args.device_type]  # 根据设备类型选择对应的GPU配置
        self.args = args  # 保存命令行参数
        self.config = config  # 保存模型结构配置

    def print_weights_info(self):
        print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))  # 打印标题：模型权重信息
        full_attn_params_bytes = get_attn_params_size(
            self.config, self.args.use_fp8_gemm
        )  # 计算完整注意力层的参数大小（字节）
        linear_attn_params_bytes = get_linear_attn_params_size(
            self.config, self.args.use_fp8_gemm
        )  # 计算线性注意力层的参数大小（字节）
        expert_params_bytes = get_expert_params_size(
            self.config, self.args.use_fp8_gemm
        )  # 计算单个专家网络的参数大小（字节）
        print(
            "{:<40} {:<10.2f}".format(
                "One full attn params size (MB):", full_attn_params_bytes / 1024 / 1024
            )
        )  # 输出单个完整注意力层参数大小（MB）
        print(
            "{:<40} {:<10.2f}".format(
                "One linear attn params size (MB):",
                linear_attn_params_bytes / 1024 / 1024,
            )
        )  # 输出单个线性注意力层参数大小（MB）
        print(
            "{:<40} {:<10.2f}".format(
                "One expert params size (MB):", expert_params_bytes / 1024 / 1024
            )
        )  # 输出单个专家参数大小（MB）
        # 每张GPU上存放的专家参数总量（共享专家 + 路由专家按并行度划分）
        params_per_gpu = expert_params_bytes * (
            self.config.num_shared_experts
            + self.config.num_routed_experts / self.args.world_size
        )
        params_per_gpu *= self.config.num_hidden_layers  # 乘以层数得到所有层的专家参数
        params_per_gpu += self.config.num_full_attn_layers * full_attn_params_bytes  # 加上完整注意力层参数
        params_per_gpu += self.config.num_linear_attn_layers * full_attn_params_bytes  # 注意：此处应为linear_attn_params_bytes，疑似bug

        params_per_gpu = params_per_gpu / 1024 / 1024 / 1024  # 转换为GB
        self.kvcache_mem = (
            self.gpu.mem - params_per_gpu - 15 - 5
        )  # 可用KV缓存内存 = 总显存 - 模型参数 - 运行时开销(15GB) - 编码器预留(5GB)
        print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))  # 输出每GPU参数占用（GB）

    def print_kvcache_info(self):
        print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))  # 打印标题：KV缓存信息
        print("{:<40} {:<10.2f}".format("KV cache space (GB):", self.kvcache_mem))  # 输出可用KV缓存空间（GB）
        context_len = self.args.target_isl + self.args.target_osl  # 上下文长度 = 输入长度 + 输出长度

        if self.args.decode_bs is None:  # 如果未指定解码batch size
            target_bs = math.ceil(self.args.target_tgs * self.args.target_tpot / 1000)  # 根据目标吞吐和TPOT估算batch size
        else:
            target_bs = self.args.decode_bs  # 使用指定的解码batch size
        print("{:<40} {:<10}".format("Input seq len:", self.args.target_isl))  # 输出输入序列长度
        print("{:<40} {:<10}".format("Output seq len:", self.args.target_osl))  # 输出输出序列长度
        print("{:<40} {:<10}".format("Target decode batchsize:", target_bs))  # 输出目标解码batch size
        target_kvcache_bytes = self.kvcache_mem * 1024 * 1024 * 1024 / target_bs  # 每请求可分配的KV缓存字节数
        kvcache_bytes = (
            get_kvcache_size(self.config, self.args.use_fp8_kv)
            / self.config.num_hidden_layers
        )  # 单层KV缓存大小
        kvcache_bytes *= self.config.num_full_attn_layers * context_len  # 完整注意力层 × 上下文长度
        print(
            "{:<40} {:<10.2f}".format(
                "Target per-req KV cache size (MB):", target_kvcache_bytes / 1024 / 1024
            )
        )  # 输出每请求目标KV缓存大小（MB）
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-req full attn KV cache size (MB):",
                kvcache_bytes / 1024 / 1024,
            )
        )  # 输出当前每请求完整注意力KV缓存大小（MB）
        states_bytes = get_states_size(self.config)  # 获取状态缓存大小（如线性注意力中的递归状态）
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-req states size (MB):", states_bytes / 1024 / 1024
            )
        )  # 输出每请求状态缓存大小（MB）
        print(
            "{:<40} {:<10.2f}".format(
                "Current per-req cache size (MB):",
                (kvcache_bytes + states_bytes) / 1024 / 1024,
            )
        )  # 输出每请求总缓存大小（KV + 状态）
        if kvcache_bytes + states_bytes > target_kvcache_bytes:
            print("!Error: need smaller kvcache")  # 错误提示：KV缓存超出容量
        self.kvcache_bytes = kvcache_bytes / context_len  # 存储每token的KV缓存大小（用于后续计算）
        self.states_bytes = states_bytes  # 存储每请求的状态缓存大小
        self.target_bs = target_bs  # 存储目标batch size

    def print_flops_info(self):
        print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))  # 打印标题：FLOPs信息
        print(
            "{:<40} {:<10}".format("Num hidden layers:", self.config.num_hidden_layers)
        )  # 输出隐藏层数量
        # 每token每层GFLOPs计算
        self.avg_context_len = int(self.args.target_isl + self.args.target_osl / 2)  # 平均上下文长度
        attn_core_gflops, other_gflops = get_attn_gflops(
            self.config, self.avg_context_len, absorb=True
        )  # 获取注意力核心和其他部分的GFLOPs
        moe_gflops = get_moe_gflops(self.config)  # 获取MoE部分的GFLOPs
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer full attn core (GFLOPs):", attn_core_gflops
            )
        )  # 输出每token每层完整注意力核心计算量
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer MoE/FFN (GFLOPs):", moe_gflops
            )
        )  # 输出每token每层MoE/FFN计算量
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token per-layer others (GFLOPs):", other_gflops
            )
        )  # 输出每token每层其他部分计算量
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token full attn core (GFLOPs):",
                attn_core_gflops * self.config.num_full_attn_layers,
            )
        )  # 输出每token完整注意力总计算量
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token MoE (GFLOPs):", moe_gflops * self.config.num_hidden_layers
            )
        )  # 输出每token MoE总计算量
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token others (GFLOPs):",
                other_gflops * self.config.num_hidden_layers,
            )
        )  # 输出每token其他部分总计算量
        print(
            "{:<40} {:<10.2f}".format(
                "Per-token total (GFLOPs):",
                (attn_core_gflops + moe_gflops + other_gflops)
                * self.config.num_hidden_layers,
            )
        )  # 输出每token总计算量

    def prefill(self):
        print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))  # 打印标题：预填充阶段
        print(
            "{:<40} {:<10}".format("Max prefill tokens:", self.args.max_prefill_tokens)
        )  # 输出最大预填充token数
        # 完整注意力模块
        full_attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )  # 创建完整注意力实例
        t_full_attn_core = full_attn.prefill_attn_core(
            self.args.target_isl, self.kvcache_bytes, self.args.device_type
        )  # 预填充阶段注意力核心耗时
        t_full_attn_others = full_attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )  # 预填充阶段注意力其他部分耗时
        t_full_attn_core *= self.args.max_prefill_tokens / self.args.target_isl  # 按实际token数缩放

        # 线性注意力模块
        linear_attn = create_linear_attn(self.config, self.args.use_fp8_gemm)  # 创建线性注意力实例
        t_linear_attn_core = linear_attn.prefill_attn_core(
            self.args.target_isl, self.states_bytes, self.args.device_type
        )  # 线性注意力核心耗时
        t_linear_attn_core *= self.args.max_prefill_tokens / self.args.target_isl  # 按token数缩放
        t_linear_attn_others = linear_attn.prefill_attn_others(
            self.args.max_prefill_tokens, self.args.device_type
        )  # 线性注意力其他部分耗时

        # MoE模块
        moe = MoE(self.config, self.args.use_fp8_gemm)  # 创建MoE实例
        t_moe = moe.prefill_moe(
            self.args.max_prefill_tokens, self.args.device_type, self.args.world_size
        )  # MoE预填充耗时

        # 通信模块
        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )  # 创建通信实例
        comm_t1, comm_t2 = comm.prefill_comm(self.args.max_prefill_tokens)  # 获取预填充阶段通信耗时
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_t1 * 1e6))  # 输出MoE前通信耗时（微秒）
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_t2 * 1e6))  # 输出MoE后通信耗时（微秒）

        num_tokens = self.args.max_prefill_tokens  # 当前处理的token数量
        ttft = (
            t_full_attn_core + t_full_attn_others
        ) * self.config.num_full_attn_layers  # 所有完整注意力层总耗时
        ttft += (
            t_linear_attn_core + t_linear_attn_others
        ) * self.config.num_linear_attn_layers  # 所有线性注意力层总耗时
        ttft += (t_moe + comm_t1 + comm_t2) * self.config.num_hidden_layers  # 所有MoE和通信总耗时
        ttft *= 1000  # 转换为毫秒
        ttft += 30  # 加上调度器开销（30ms）

        print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))  # 输出首token延迟（TTFT）
        print(
            "{:<40} {:<10.0f}".format(
                "Throughput (TGS:tok/GPU/s):", num_tokens / (ttft / 1000)
            )
        )  # 输出吞吐量（每GPU每秒处理token数）

    def decoding(self):
        print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))  # 打印标题：解码阶段
        # 完整注意力
        full_attn = create_attention(
            self.config, self.args.use_fp8_gemm, self.args.use_fp8_kv
        )  # 创建完整注意力实例
        t_full_attn_core = full_attn.decode_attn_core(
            self.target_bs,
            self.avg_context_len,
            self.kvcache_bytes,
            self.args.device_type,
        )  # 解码阶段注意力核心耗时
        t_full_attn_others = full_attn.decode_attn_others(
            self.target_bs, self.args.device_type
        )  # 解码阶段注意力其他部分耗时

        # 线性注意力
        linear_attn = create_linear_attn(self.config, self.args.use_fp8_gemm)  # 创建线性注意力实例
        t_linear_attn_core = linear_attn.decode_attn_core(
            self.target_bs, self.states_bytes, self.args.device_type
        )  # 解码阶段线性注意力核心耗时
        t_linear_attn_others = linear_attn.decode_attn_others(
            self.target_bs, self.args.device_type
        )  # 解码阶段线性注意力其他部分耗时

        # MoE模块
        moe = MoE(self.config, self.args.use_fp8_gemm)  # 创建MoE实例
        t_moe = moe.decode_moe(
            self.target_bs, self.args.device_type, self.args.world_size
        )  # 解码阶段MoE耗时

        # 通信模块
        comm = Comm(
            self.config,
            self.gpu,
            self.args.world_size,
            self.args.num_nodes,
            self.args.enable_deepep,
        )  # 创建通信实例
        comm_t1, comm_t2 = comm.decode_comm(self.target_bs)  # 获取解码阶段通信耗时
        print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_t1 * 1e6))  # 输出MoE前通信耗时（微秒）
        print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_t2 * 1e6))  # 输出MoE后通信耗时（微秒）

        num_tokens = self.target_bs  # 当前处理的token数量（等于batch size）
        tpot = (
            t_full_attn_core + t_full_attn_others
        ) * self.config.num_full_attn_layers  # 所有完整注意力层总耗时
        tpot += (
            t_linear_attn_core + t_linear_attn_others
        ) * self.config.num_linear_attn_layers  # 所有线性注意力层总耗时
        tpot += (t_moe + comm_t1 + comm_t2) * self.config.num_hidden_layers  # 所有MoE和通信总耗时
        tpot *= 1000  # 转换为毫秒
        tpot += 5  # 加上调度器开销（5ms）

        print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))  # 输出每个token处理时间（TPOT）
        print("{:<40} {:<10.0f}".format("Throughput (TGS):", num_tokens / tpot * 1000))  # 输出吞吐量（TGS）
        if tpot > self.args.target_tpot:
            print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")  # 错误提示：TPOT超过目标，需优化性能