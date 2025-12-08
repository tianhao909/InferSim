from config.model_config import ModelConfig  # 从 config.model_config 导入模型配置类
from hardware.gpu import GPU  # 从 hardware.gpu 导入 GPU 类


class Comm:  # 定义通信类 Comm
    def __init__(  # 初始化函数
        self,
        config: ModelConfig,  # 模型配置对象
        gpu: GPU,  # GPU 对象
        world_size: int,  # 进程总数（世界大小）
        num_nodes=1,  # 节点数量，默认为 1
        enable_deepep=False,  # 是否启用 DeepEP 优化，默认为 False
    ):
        self.config = config  # 保存模型配置
        self.gpu = gpu  # 保存 GPU 信息
        self.world_size = world_size  # 保存世界大小
        self.num_nodes = num_nodes  # 保存节点数量
        self.enable_deepep = enable_deepep  # 保存 DeepEP 是否启用

    def size_bw_model(self, tensor_shape, use_fp8=False, inter_node=False):  
        # 计算张量通信所需的时间（单位：秒），根据带宽和大小
        if self.world_size <= 1:  # 如果只有一个进程则不需要通信
            return 0  # 返回 0
        size = 1 if use_fp8 else 2  # 如果是 FP8 精度则每元素占 1 字节，否则占 2 字节
        for v in tensor_shape:  # 遍历张量形状的每个维度
            size *= v  # 计算总元素数
        if inter_node:  # 如果是跨节点通信
            return size / (1024**3) / self.gpu.rdma_bw  # 使用 RDMA 带宽计算时间（GB 转换）
        return size / (1024**3) / self.gpu.nvlink_bw  # 否则使用 NVLink 带宽计算时间

    def all_reduce(self, num_tokens):  
        # 计算 All-Reduce 通信时间
        tensor_shape = [num_tokens * self.world_size, self.config.hidden_size]  
        # 张量形状：所有进程的 token 数 × 隐藏层大小
        return self.size_bw_model(
            tensor_shape, use_fp8=False, inter_node=(self.num_nodes > 1)  
            # 不使用 FP8，是否跨节点由节点数判断
        )

    def dispatch(self, num_tokens, mode="normal"):  
        # 分发 tokens 的通信时间
        if mode == "normal":  # 普通模式
            send_tokens = num_tokens * (self.num_nodes - 1)  
            # 要发送的 token 数为本节点发往其他节点总数
            tensor_shape1 = [send_tokens, self.config.hidden_size]  
            # 第一部分：跨节点发送形状
            t1 = self.size_bw_model(tensor_shape1, use_fp8=True, inter_node=True)  
            # 跨节点，使用 FP8

            tensor_shape2 = [num_tokens, self.config.hidden_size]  
            # 第二部分：节点内发送形状
            t2 = self.size_bw_model(tensor_shape2, use_fp8=True, inter_node=False)  
            # 节点内，使用 FP8
            return t1 + t2  # 返回总时间
        else:  # 低延迟模式（用于专家模型）
            send_tokens = num_tokens * self.config.num_experts_per_tok  
            # 要发送的 token 数为每 token 对应专家数倍
            tensor_shape = [send_tokens, self.config.hidden_size]  
            # 形状：发送 token 数 × 隐藏层大小
            return self.size_bw_model(
                tensor_shape, use_fp8=True, inter_node=(self.num_nodes > 1)  
                # 使用 FP8，跨节点由节点数判断
            )

    def combine(self, num_tokens, mode="normal"):  
        # 接收 tokens 的通信时间
        if mode == "normal":  # 普通模式
            rcv_tokens = num_tokens * (self.num_nodes - 1)  
            # 接收的 token 数为来自其他节点的总数
            tensor_shape1 = [rcv_tokens, self.config.hidden_size]  
            # 第一部分：跨节点接收形状
            t1 = self.size_bw_model(tensor_shape1, use_fp8=False, inter_node=True)  
            # 跨节点，使用 FP16/FP32（每元素 2 字节）

            tensor_shape2 = [num_tokens, self.config.hidden_size]  
            # 第二部分：节点内接收形状
            t2 = self.size_bw_model(tensor_shape2, use_fp8=False, inter_node=False)  
            # 节点内，使用 FP16/FP32
            return t1 + t2  # 返回总时间
        else:  # 低延迟模式（用于专家模型）
            rcv_tokens = num_tokens * self.config.num_experts_per_tok  
            # 接收 token 数为每 token 对应专家数倍
            tensor_shape = [rcv_tokens, self.config.hidden_size]  
            # 接收张量形状
            return self.size_bw_model(
                tensor_shape, use_fp8=False, inter_node=(self.num_nodes > 1)  
                # 使用 FP16/FP32，跨节点由节点数判断
            )

    def a2f(self, num_tokens):  
        # 从激活到前馈（跨节点，使用 FP8）
        tensor_shape = [num_tokens, self.config.hidden_size]  
        return self.size_bw_model(tensor_shape, use_fp8=True, inter_node=True)  

    def f2a(self, num_tokens):  
        # 从前馈到激活（跨节点，使用 FP16/FP32）
        tensor_shape = [num_tokens, self.config.hidden_size]  
        return self.size_bw_model(tensor_shape, use_fp8=False, inter_node=True)  

    def prefill_comm(self, num_tokens: int):  
        # 预填充阶段的通信时间
        if self.enable_deepep:  # 如果启用 DeepEP
            return self.dispatch(num_tokens, "normal"), self.combine(
                num_tokens, "normal"
            )
        return self.all_reduce(num_tokens), self.all_reduce(num_tokens)  
        # 如果未启用 DeepEP，则用 All-Reduce

    def decode_comm(self, num_tokens: int):  
        # 解码阶段的通信时间
        if self.enable_deepep:  # 如果启用 DeepEP
            return self.dispatch(num_tokens, "low_latency"), self.combine(
                num_tokens, "low_latency"
            )
        return self.all_reduce(num_tokens), self.all_reduce(num_tokens)  
        # 如果未启用 DeepEP，则用 All-Reduce
