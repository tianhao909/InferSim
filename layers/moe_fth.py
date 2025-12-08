from flops.flops import gemm_flops  # 从 flops.flops 模块导入 gemm_flops，用于计算矩阵乘法的 FLOPs
from hardware.gpu import gpu_map  # 从 hardware.gpu 模块导入 gpu_map，用于映射设备类型到 GPU 规格
from layers.attn import get_gemm_mfu_and_latency  # 从 layers.attn 模块导入函数，用于获取 GEMM 的 MFU 和延迟
from mfu.mfu import (get_gemm_mfu, get_groupedgemm_decode_mfu,  # 从 mfu.mfu 模块导入计算 MFU 的相关函数
                     get_groupedgemm_prefill_mfu)
from params.params import load_moe_weights_time  # 从 params.params 模块导入加载 MoE 权重的时间计算函数


class MoE:  # MoE 类，表示专家混合层
    """
    MoE/FFN 层，密集 FFN 被视作特殊的 1 专家 MoE
    """  # 原英文注释翻译：MoE/FFN 层，密集型 FFN 当作只有 1 个专家的 MoE

    def __init__(self, config, use_fp8_gemm):  # 初始化方法，接收配置和是否使用 FP8 GEMM
        self.use_fp8_gemm = use_fp8_gemm  # 保存是否使用 FP8 GEMM 的标志
        self.config = config  # 保存模型配置

    def decode_moe(self, bs, device_type, num_gpus):  # decode 阶段的 MoE 计算方法
        gpu = gpu_map[device_type]  # 根据设备类型获取对应 GPU 参数

        routed_experts_gflops = gemm_flops(  # 计算一次 GEMM 操作的 FLOPs
            1, self.config.hidden_size, self.config.intermediate_size  # GEMM 的 m, k, n 参数
        )
        routed_experts_gflops *= bs * self.config.num_experts_per_tok * 3.0 / 1e9  # 按 batch size、每 token 的专家数和系数计算总的 GFLOPs，并转为单位 G

        if self.config.is_moe:  # 如果配置中启用 MoE
            routed_experts_mfu = max(  # 获取 grouped GEMM decode 阶段的最大 MFU 值
                get_groupedgemm_decode_mfu(
                    self.config, bs, device_type, num_gpus, self.use_fp8_gemm
                )
            )
        else:  # 密集型 FFN 被作为特殊的 1 专家 MoE 处理
            routed_experts_mfu = get_gemm_mfu(  # 获取普通 GEMM 的 MFU 值
                device_type,
                bs,
                self.config.hidden_size,
                self.config.intermediate_size * 2 // num_gpus,  # 对中间层维度乘 2 再按 GPU 数量分配
            )

        routed_experts_latency = routed_experts_gflops / (  # 计算延迟
            gpu.fp16_tflops * 1024 * routed_experts_mfu  # 使用 FP16 性能值和 MFU 计算
        )
        if self.use_fp8_gemm:  # 如果使用 FP8 GEMM
            routed_experts_latency = routed_experts_gflops / (  # 按 FP8 性能计算延迟
                gpu.fp8_tflops * 1024 * routed_experts_mfu
            )

        moe_load_time = load_moe_weights_time(  # 计算加载 MoE 权重的时间
            self.config, self.use_fp8_gemm, gpu, num_gpus
        )
        print("{:<40} {:<10.2f}".format("Routed experts/FFN MFU:", routed_experts_mfu))  # 打印 MFU 值
        print(
            "{:<40} {:<10.2f}".format(
                "Routed experts/FFN latency (us):", routed_experts_latency * 1e6  # 将延迟转换为微秒打印
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Experts loading latency (us):", moe_load_time * 1e6  # 打印加载权重延迟，单位微秒
            )
        )
        t = max(routed_experts_latency, moe_load_time)  # 初始总时间取这两个延迟的最大值

        if self.config.num_shared_experts > 0:  # 如果存在共享专家
            shared_expert_up_proj = get_gemm_mfu_and_latency(  # 计算共享专家上投影的延迟
                m=bs,
                k=self.config.hidden_size,
                n=self.config.intermediate_size * 2 * self.config.num_shared_experts,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )

            shared_expert_down_proj = get_gemm_mfu_and_latency(  # 计算共享专家下投影的延迟
                m=bs,
                k=self.config.intermediate_size * self.config.num_shared_experts,
                n=self.config.hidden_size,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Shared expert latency (us):",
                    (shared_expert_up_proj + shared_expert_down_proj) * 1e6,  # 打印共享专家总延迟
                )
            )
            t += shared_expert_up_proj + shared_expert_down_proj  # 总时间加上共享专家延迟
        return t  # 返回总时间

    def prefill_moe(self, seq_len, device_type, num_gpus):  # prefill 阶段的 MoE 计算方法
        gpu = gpu_map[device_type]  # 根据设备类型获取 GPU 参数

        routed_experts_gflops = gemm_flops(  # 计算 GEMM FLOPs
            1, self.config.hidden_size, self.config.intermediate_size
        )
        routed_experts_gflops *= seq_len * self.config.num_experts_per_tok * 3.0 / 1e9  # 按序列长度、每 token 专家数和系数计算总 GFLOPs

        if self.config.is_moe:  # 如果是 MoE
            routed_experts_mfu = max(  # 获取 grouped GEMM prefill 阶段 MFU
                get_groupedgemm_prefill_mfu(
                    self.config, seq_len, device_type, num_gpus, self.use_fp8_gemm
                )
            )
        else:  # 密集型 FFN 当作 1 专家 MoE
            routed_experts_mfu = get_gemm_mfu(  # 获取普通 GEMM 的 MFU
                device_type,
                seq_len,
                self.config.hidden_size,
                self.config.intermediate_size * 2 // num_gpus,
            )

        routed_experts_latency = routed_experts_gflops / (  # 计算延迟
            gpu.fp16_tflops * 1024 * routed_experts_mfu
        )
        if self.use_fp8_gemm:  # 如果用 FP8 GEMM
            routed_experts_latency = routed_experts_gflops / (  # 按 FP8 性能计算延迟
                gpu.fp8_tflops * 1024 * routed_experts_mfu
            )

        moe_load_time = load_moe_weights_time(  # 加载 MoE 权重时间
            self.config, self.use_fp8_gemm, gpu, num_gpus
        )
        print("{:<40} {:<10.2f}".format("Routed experts MFU:", routed_experts_mfu))  # 打印 MFU
        print(
            "{:<40} {:<10.2f}".format(
                "Routed experts latency (us):", routed_experts_latency * 1e6  # 打印路由专家延迟
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                "Experts loading latency (us):", moe_load_time * 1e6  # 打印权重加载延迟
            )
        )
        t = max(routed_experts_latency, moe_load_time)  # 取两者最大值作为初始时间

        if self.config.num_shared_experts > 0:  # 如果有共享专家
            shared_expert_up_proj = get_gemm_mfu_and_latency(  # 上投影延迟
                m=seq_len,
                k=self.config.hidden_size,
                n=self.config.intermediate_size * 2 * self.config.num_shared_experts,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )

            shared_expert_down_proj = get_gemm_mfu_and_latency(  # 下投影延迟
                m=seq_len,
                k=self.config.intermediate_size * self.config.num_shared_experts,
                n=self.config.hidden_size,
                device_type=device_type,
                use_fp8_gemm=self.use_fp8_gemm,
            )
            print(
                "{:<40} {:<10.2f}".format(
                    "Shared expert latency (us):",
                    (shared_expert_up_proj + shared_expert_down_proj) * 1e6,  # 打印共享专家延迟
                )
            )
            t += shared_expert_up_proj + shared_expert_down_proj  # 总时间加上共享专家延迟
        return t  # 返回总时间
