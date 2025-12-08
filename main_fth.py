import argparse  # 导入命令行参数解析库
import math  # 导入数学库

from comm.comm import Comm  # 从通信模块导入 Comm 类
from config.model_config import ModelConfig  # 从配置模块导入模型配置类
from flops.flops import get_attn_gflops, get_moe_gflops  # 导入计算 FLOPs 的函数
from hardware.gpu import gpu_map  # 导入 GPU 设备映射表
from kvcache.kvcache import get_kvcache_size  # 导入计算 KV cache 大小的函数
from layers.attn import create_attention  # 导入创建注意力层的函数
from layers.moe import MoE  # 导入 MoE（专家混合）层
from params.params import get_attn_params_size, get_expert_params_size  # 导入参数大小计算函数


def prefill(args, config, gpu, kvcache_bytes):  # 定义预填充（prefill）阶段的模拟函数
    print("{s:{c}^{n}}".format(s="Prefilling", n=50, c="-"))  # 打印标题，居中，用“-”填充
    print("{:<40} {:<10}".format("Max prefill tokens:", args.max_prefill_tokens))  # 打印最大预填充 token 数
    attn = create_attention(config, args.use_fp8_gemm, args.use_fp8_kv)  # 创建注意力对象，考虑是否使用 FP8
    attn_core_time = attn.prefill_attn_core(  # 计算预填充阶段注意力核心计算耗时（单位：秒）
        args.target_isl, kvcache_bytes, args.device_type  # 传入输入序列长度、KV 大小和设备类型
    )
    attn_other_time = attn.prefill_attn_others(  # 计算预填充阶段注意力非核心部分耗时（单位：秒）
        args.max_prefill_tokens, args.device_type  # 传入最大预填充 token 数和设备类型
    )
    attn_core_time *= math.ceil(args.max_prefill_tokens / args.target_isl)  # 将核心时间乘以需要的块数（ceil 向上取整）

    moe = MoE(config, args.use_fp8_gemm)  # 创建 MoE 层对象，考虑是否使用 FP8 GEMM
    moe_time = moe.prefill_moe(  # 计算预填充阶段 MoE/FFN 的耗时（单位：秒）
        args.max_prefill_tokens, args.device_type, args.world_size  # 传入最大 token 数、设备类型和世界大小
    )

    comm = Comm(config, gpu, args.world_size, args.num_nodes, args.enable_deepep)  # 创建通信对象，包含是否启用 DeepEP
    comm_time1, comm_time2 = comm.prefill_comm(args.max_prefill_tokens)  # 计算预填充阶段两次通信的耗时（单位：秒）
    print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))  # 打印 MoE/FFN 前通信耗时（转换为微秒）
    print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))  # 打印 MoE/FFN 后通信耗时（转换为微秒）

    num_tokens = args.max_prefill_tokens  # 预填充处理的 token 数
    if args.enable_tbo:  # 如果启用 TBO（双批次重叠）
        num_tokens *= 2  # Throughput 统计按双批次计数
        ttft = max((attn_core_time + attn_other_time) / args.sm_ratio, comm_time1)  # 第一段：计算与通信重叠取最大值
        ttft += max((attn_core_time + attn_other_time) / args.sm_ratio, comm_time2)  # 第二段：同理
        ttft += max(moe_time / args.sm_ratio, comm_time1)  # 第三段：MoE 计算与通信重叠
        ttft += max(moe_time / args.sm_ratio, comm_time2)  # 第四段：同理
    else:  # 未启用 TBO
        ttft = attn_core_time  # 先计算注意力核心时间
        ttft += moe_time  # 加上 MoE 时间
        ttft += attn_other_time  # 加上注意力其他部分时间
        ttft += comm_time1 + comm_time2  # 加上两次通信时间
    ttft *= config.num_hidden_layers  # 乘以层数，得到总延迟
    ttft *= 1000  # 转换为毫秒  # 转换为毫秒
    ttft += 30  # 加上调度器开销（估计值，毫秒）  # 为调度器预留时间

    print("{:<40} {:<10.2f}".format("TTFT (ms):", ttft))  # 打印首 token 延迟（毫秒）
    print(  # 打印吞吐（TGS：每 GPU 每秒 token），按预填充 token 数除以时间
        "{:<40} {:<10.0f}".format(
            "Throughput (TGS:tok/GPU/s):", num_tokens / (ttft / 1000)
        )
    )


def decoding(args, config, gpu, target_bs, kvcache_bytes, avg_context_len):  # 定义解码阶段模拟函数
    print("{s:{c}^{n}}".format(s="Decoding", n=50, c="-"))  # 打印标题
    attn = create_attention(config, args.use_fp8_gemm, args.use_fp8_kv)  # 创建注意力对象
    attn_core_time = attn.decode_attn_core(  # 计算解码阶段注意力核心时间（单位：秒）
        target_bs, avg_context_len, kvcache_bytes, args.device_type  # 传入批大小、平均上下文长度、KV大小和设备类型
    )
    attn_other_time = attn.decode_attn_others(target_bs, args.device_type)  # 计算解码阶段注意力其他部分时间（单位：秒）

    moe = MoE(config, args.use_fp8_gemm)  # 创建 MoE 对象
    moe_time = moe.decode_moe(target_bs, args.device_type, args.world_size)  # 计算解码阶段 MoE/FFN 时间（单位：秒）

    comm = Comm(config, gpu, args.world_size, args.num_nodes, args.enable_deepep)  # 创建通信对象
    comm_time1, comm_time2 = comm.decode_comm(target_bs)  # 计算解码阶段两次通信时间（单位：秒）
    print("{:<40} {:<10.2f}".format("Comm before MoE/FFN (us):", comm_time1 * 1e6))  # 打印 MoE 前通信时间（微秒）
    print("{:<40} {:<10.2f}".format("Comm after MoE/FFN (us):", comm_time2 * 1e6))  # 打印 MoE 后通信时间（微秒）

    num_tokens = target_bs  # 解码时的 token 数（每步处理的样本数）
    if args.enable_tbo:  # 如果启用 TBO
        num_tokens *= 2  # 吞吐按双批次计
        tpot = max(attn_core_time + attn_other_time, moe_time + comm_time1 + comm_time2)  # 计算每步时间取两部分最大值（计算与通信重叠）
        tpot *= 2  # 双批次重叠乘 2
    else:  # 未启用 TBO
        tpot = attn_core_time  # 注意力核心时间
        tpot += attn_other_time  # 注意力其他部分时间
        tpot += moe_time  # MoE/FFN 时间
        tpot += comm_time1 + comm_time2  # 两次通信时间
    tpot *= config.num_hidden_layers  # 乘以层数
    tpot *= 1000  # 转换为毫秒  # 转换为毫秒
    tpot += 5  # 加上调度器开销（估计值，毫秒）  # 为调度器预留时间

    print("{:<40} {:<10.2f}".format("TPOT (ms):", tpot))  # 打印每步解码时间（毫秒）
    print("{:<40} {:<10.0f}".format("Throughput (TGS):", num_tokens / tpot * 1000))  # 打印解码吞吐（TGS）
    if tpot > args.target_tpot:  # 如果每步时间超过设定的 SLO（目标时间）
        print("!Error: TPOT > SLO, need smaller GFLOPs to speedup")  # 提示需要降低 GFLOPs 以提升速度


def main(args):  # 主函数，负责整体流程
    config = ModelConfig(args.config_path)  # 加载模型配置
    gpu = gpu_map[args.device_type]  # 根据设备类型 获取 GPU 参数

    print("\n{s:{c}^{n}}".format(s=" Simulator Result ", n=50, c="="))  # 打印总标题
    print("{:<40} {:<10}".format("Device type:", args.device_type))  # 打印设备类型
    print("{:<40} {:<10}".format("World size:", args.world_size))  # 打印世界大小（GPU 数量）
    print("{:<40} {:<10}".format("Attn type:", config.attn_type))  # 打印注意力类型
    print("{:<40} {:<10}".format("Use FP8 GEMM:", args.use_fp8_gemm))  # 是否使用 FP8 GEMM
    print("{:<40} {:<10}".format("Use FP8 KV:", args.use_fp8_kv))  # 是否使用 FP8 KV 缓存

    print("{s:{c}^{n}}".format(s="Model Weights", n=50, c="-"))  # 打印模型权重信息标题
    attn_params_bytes = get_attn_params_size(config, args.use_fp8_gemm)  # 计算单个注意力层参数大小（字节）
    expert_params_bytes = get_expert_params_size(config, args.use_fp8_gemm)  # 计算单个专家（FFN/MoE）参数大小（字节）
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
    params_per_gpu = attn_params_bytes + expert_params_bytes * (  # 计算每个 GPU 的参数大小（字节）
        config.num_shared_experts + config.num_routed_experts / args.world_size  # 共享专家全部在每 GPU，上游路由专家平均分配
    )
    params_per_gpu = params_per_gpu / 1024 / 1024 / 1024  # 转换为 GB
    params_per_gpu *= config.num_hidden_layers  # 乘以隐藏层数得到总参数
    kvcache_mem = gpu.mem - params_per_gpu - 15 - 5  # 计算可用 KV cache 内存（GB），扣除参数与运行时开销  # 15GB 用于运行时，5GB 用于编码器
    print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))  # 打印每 GPU 参数大小（GB）

    print("{s:{c}^{n}}".format(s="KV Cache", n=50, c="-"))  # 打印 KV cache 信息标题
    print("{:<40} {:<10.2f}".format("KV cache space (GB):", kvcache_mem))  # 打印可用 KV cache 空间（GB）
    context_len = args.target_isl + args.target_osl  # 计算总上下文长度（输入+输出）

    if args.decode_bs is None:  # 如果未指定解码批大小
        target_bs = math.ceil(args.target_tgs * args.target_tpot / 1000)  # 按目标 TGS 和 TPOT 估算批大小（向上取整）
    else:  # 如果指定了
        target_bs = args.decode_bs  # 使用用户指定批大小
    print("{:<40} {:<10}".format("Input seq len:", args.target_isl))  # 打印输入序列长度
    print("{:<40} {:<10}".format("Output seq len:", args.target_osl))  # 打印输出序列长度
    print("{:<40} {:<10}".format("Target decode batchsize:", target_bs))  # 打印目标解码批大小
    target_kvcache_bytes = kvcache_mem * 1024 * 1024 * 1024 / target_bs / context_len  # 计算目标每 token 可用 KV 空间（字节）
    kvcache_bytes = get_kvcache_size(config, args.use_fp8_kv)  # 计算当前配置下每 token KV 大小（字节）
    print(  # 打印目标每 token KV 大小（KB）
        "{:<40} {:<10.2f}".format(
            "Target per-token KV cache size (KB):", target_kvcache_bytes / 1024
        )
    )
    print(  # 打印当前每 token KV 大小（KB）
        "{:<40} {:<10.2f}".format(
            "Current per-token KV cache size (KB):", kvcache_bytes / 1024
        )
    )
    if kvcache_bytes > target_kvcache_bytes:  # 如果当前 KV 大小超过目标可用空间
        print("!Error: need smaller kvcache")  # 提示需要更小的 KV cache

    print("{s:{c}^{n}}".format(s="FLOPs", n=50, c="-"))  # 打印 FLOPs 信息标题
    print("{:<40} {:<10}".format("Num hidden layers:", config.num_hidden_layers))  # 打印隐藏层数
    # per-token per-layer gflops  # 每 token 每层的 GFLOPs
    avg_context_len = int(args.target_isl + args.target_osl / 2)  # 估算平均上下文长度（输入 + 输出的一半）  # 平均上下文长度
    attn_core_gflops, other_gflops = get_attn_gflops(  # 计算注意力核心与其他部分的 GFLOPs
        config, avg_context_len, absorb=True  # 吸收一些常量项到核心计算中
    )
    moe_gflops = get_moe_gflops(config)  # 计算 MoE/FFN 的 GFLOPs
    print(  # 打印每 token 每层注意力核心 GFLOPs
        "{:<40} {:<10.2f}".format(
            "Per-token per-layer attn core (GFLOPs):", attn_core_gflops
        )
    )
    print(  # 打印每 token 每层 MoE/FFN GFLOPs
        "{:<40} {:<10.2f}".format("Per-token per-layer MoE/FFN (GFLOPs):", moe_gflops)
    )
    print(  # 打印每 token 每层其他部分 GFLOPs
        "{:<40} {:<10.2f}".format("Per-token per-layer others (GFLOPs):", other_gflops)
    )
    print(  # 打印每 token 注意力核心总 GFLOPs（乘层数）
        "{:<40} {:<10.2f}".format(
            "Per-token attn core (GFLOPs):", attn_core_gflops * config.num_hidden_layers
        )
    )
    print(  # 打印每 token MoE 总 GFLOPs（乘层数）
        "{:<40} {:<10.2f}".format(
            "Per-token MoE (GFLOPs):", moe_gflops * config.num_hidden_layers
        )
    )
    print(  # 打印每 token 其他部分总 GFLOPs（乘层数）
        "{:<40} {:<10.2f}".format(
            "Per-token others (GFLOPs):", other_gflops * config.num_hidden_layers
        )
    )
    print(  # 打印每 token 总 GFLOPs（核心+MoE+其他，乘层数）
        "{:<40} {:<10.2f}".format(
            "Per-token total (GFLOPs):",
            (attn_core_gflops + moe_gflops + other_gflops) * config.num_hidden_layers,
        )
    )

    if not args.decode_only:  # 如果不是仅解码模式
        prefill(args, config, gpu, kvcache_bytes)  # 执行预填充阶段模拟

    if not args.prefill_only:  # 如果不是仅预填充模式
        decoding(args, config, gpu, target_bs, kvcache_bytes, avg_context_len)  # 执行解码阶段模拟


if __name__ == "__main__":  # 程序入口
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(  # 添加配置路径参数
        "--config-path",
        type=str,
        help="The path of the hf model config.json",  # 帮助信息：HuggingFace 模型 config.json 的路径
        required=True,
    )
    parser.add_argument(  # 添加设备类型参数
        "--device-type",
        type=str,
        default="H20",
        choices=["H20", "H800"],
        help="Device type",  # 帮助信息：设备类型
    )
    parser.add_argument("--world-size", type=int, default=1, help="Num of GPUs")  # 添加世界大小（GPU 数量）
    parser.add_argument("--num-nodes", type=int, default=1, help="Num of nodes")  # 添加节点数量
    parser.add_argument(  # 添加最大预填充 token 参数
        "--max-prefill-tokens", type=int, default=4096, help="Max prefill tokens"
    )
    parser.add_argument(  # 添加解码批大小参数（可选）
        "--decode-bs",
        type=int,
        help="Decoding batchsize. If not specified, bs = tgs * tpot.",  # 帮助信息：未指定时按 tgs*tpot 估算
    )
    parser.add_argument(  # 添加目标 TGS 参数（tokens/s per GPU）
        "--target-tgs", type=float, default=2560, help="Target tokens/s per GPU"
    )
    parser.add_argument("--target-tpot", type=float, default=50, help="TPOT in ms")  # 添加目标 TPOT 参数（毫秒）
    parser.add_argument(  # 添加目标输入序列长度参数
        "--target-isl", type=int, default=4096, help="Input sequence length, in tokens"
    )
    parser.add_argument(  # 添加目标输出序列长度参数
        "--target-osl", type=int, default=2048, help="Output sequence length, in tokens"
    )
    parser.add_argument("--use-fp8-gemm", action="store_true", help="Use fp8 gemm")  # 添加是否使用 FP8 GEMM 的开关
    parser.add_argument("--use-fp8-kv", action="store_true", help="Use fp8 kvcache")  # 添加是否使用 FP8 KV 缓存的开关
    parser.add_argument("--enable-deepep", action="store_true", help="Enable DeepEP")  # 添加是否启用 DeepEP 的开关
    parser.add_argument(  # 添加是否启用 TBO 的开关
        "--enable-tbo", action="store_true", help="Enable two batch overlap"
    )
    parser.add_argument(  # 添加 SM 比例参数，用于 TBO 模式估算计算资源占用
        "--sm-ratio",
        type=float,
        default=108 / 132,
        help="In TBO DeepEP normal mode, the SM ratio used for computation",  # 帮助信息：TBO DeepEP 模式下用到的 SM 比例
    )
    parser.add_argument(  # 添加仅预填充模式开关
        "--prefill-only", action="store_true", help="Only simulate prefill"
    )
    parser.add_argument(  # 添加仅解码模式开关
        "--decode-only", action="store_true", help="Only simulate decoding"
    )
    args = parser.parse_args()  # 解析命令行参数
    main(args)  # 运行主函数