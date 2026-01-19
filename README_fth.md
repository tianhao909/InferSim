# InferSim: A Lightweight LLM Inference Performance Simulator

InferSim is a lightweight simulator for LLM inference, written in pure Python without any 3rd-party depenencies. It calculates the TTFT, TPOT and throughput TGS (tokens/GPU/second) based on computation complexity FLOPs (Floating-Point Operations), GPU computing power FLOPS (Floating-Point Operations per Second), GPU memory bandwidth and MFU (Model FLOPs Utilization) obtained by benchmarking the state-of-the-art LLM kernels. For multi-GPU, multi-node deployment, InferSim also estimates the communication latency according to data volume and bandwidth.

The main use cases of InferSim include:
- **Model-Sys co-design**: predicting inference performance given the hyperparameters of a model.
- **Inference performance analysis**: quantifying performance bottlenecks, such as compute-bound or IO-bound, and supporting optimization efforts.

For more details, please check [InferSim Technical Report](https://github.com/user-attachments/files/23016438/infersim_tech_report.pdf).

## Simulation Result

| Model | GPU | Prefill TGS(Actual) | Prefill TGS(Sim) | Decode TGS(Actual) | Decode TGS(Sim) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| DeepSeek-V3 | H800 | 7839 | 9034 | 2324 | 2675 | Actual data from [deepseek/profile-data](https://github.com/deepseek-ai/profile-data/). Simulated with same setup: [example/deepseek-v3/](./example/deepseek-v3/). |
| Qwen3-30B-A3B-BF16 | H20 | 16594 | 17350 | 2749 | 2632 | Actual data tested with SGLang, simulation example: [example/qwen3-30B-A3B/](./example/qwen3-30B-A3B/). |
| Qwen3-8B-FP8 | H20 | 15061 | 16328 | 2682 | 2581 | Actual data tested with SGLang, simulation example: [example/qwen3-8B/](./example/qwen3-8B/). |

## Supported Features

- **Attention**: MHA/GQA, MLA. Benchmarked on FlashInfer, FlashAttention-3, FlashMLA.
- **MoE**: GroupedGEMM. Benchmarked on DeepGEMM.
- **Linear**: GEMM. Benchmarked on DeepGEMM.
- **Parallelization**: DP Attn, EP MoE.
- **Large EP**: DeepEP dispatch and combine, with normal and low_latency mode.

## Help

```
$ python3 main.py --help
usage: main.py [-h] --config-path CONFIG_PATH [--device-type {H20,H800}] [--world-size WORLD_SIZE] [--num-nodes NUM_NODES]
               [--max-prefill-tokens MAX_PREFILL_TOKENS] [--decode-bs DECODE_BS] [--target-tgs TARGET_TGS]
               [--target-tpot TARGET_TPOT] [--target-isl TARGET_ISL] [--target-osl TARGET_OSL] [--use-fp8-gemm]
               [--use-fp8-kv] [--enable-deepep] [--enable-tbo] [--sm-ratio SM_RATIO] [--prefill-only] [--decode-only]

optional arguments:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH
                        The path of the hf model config.json
  --device-type {H20,H800}
                        Device type
  --world-size WORLD_SIZE
                        Num of GPUs
  --num-nodes NUM_NODES
                        Num of nodes
  --max-prefill-tokens MAX_PREFILL_TOKENS
                        Max prefill tokens
  --decode-bs DECODE_BS
                        Decoding batchsize. If not specified, bs = tgs * tpot.
  --target-tgs TARGET_TGS
                        Target tokens/s per GPU
  --target-tpot TARGET_TPOT
                        TPOT in ms
  --target-isl TARGET_ISL
                        Input sequence length, in tokens
  --target-osl TARGET_OSL
                        Output sequence length, in tokens
  --use-fp8-gemm        Use fp8 gemm
  --use-fp8-kv          Use fp8 kvcache
  --enable-deepep       Enable DeepEP
  --enable-tbo          Enable two batch overlap
  --sm-ratio SM_RATIO   In TBO DeepEP normal mode, the SM ratio used for computation
  --prefill-only        Only simulate prefill
  --decode-only         Only simulate decoding
```

## Example

```
$ bash example/qwen3-30B-A3B/decode.sh

================ Simulator Result ================
Device type:                             H20
World size:                              4
Attn type:                               MHA/GQA
Use FP8 GEMM:                            0
Use FP8 KV:                              0
------------------Model Weights-------------------
One attn params size (MB):               36.00
One expert params size (MB):             9.00
Per GPU params size (GB):                15.19
---------------------KV Cache---------------------
KV cache space (GB):                     60.81
Input seq len:                           4096
Output seq len:                          2048
Target decode batchsize:                 100
Target per-token KV cache size (KB):     103.79
Current per-token KV cache size (KB):    96.00
----------------------FLOPs-----------------------
Num hidden layers:                       48
Per-token per-layer attn core (GFLOPs):  0.08
Per-token per-layer MoE/FFN (GFLOPs):    0.08
Per-token per-layer others (GFLOPs):     0.04
Per-token attn core (GFLOPs):            4.03
Per-token MoE (GFLOPs):                  3.62
Per-token others (GFLOPs):               1.81
Per-token total (GFLOPs):                9.46
---------------------Decoding---------------------
Attn core MFU:                           0.15
Attn core latency (us):                  361.77
KV loading latency (us):                 298.02
QKV_proj latency (us):                   31.03
O_proj latency (us):                     16.95
Routed experts/FFN MFU:                  0.18
Routed experts/FFN latency (us):         269.28
Experts loading latency (us):            85.83
Comm before MoE/FFN (us):                4.24
Comm after MoE/FFN (us):                 4.24
TPOT (ms):                               38.00
Throughput (TGS):                        2632
```

## Acknowledgement

This work is developed and maintained by Alimama AI Infra Team & Future Living Lab, Alibaba Group.


# InferSim：一个轻量级的大语言模型推理性能模拟器

InferSim 是一个用纯 Python 编写的轻量级大语言模型（LLM）推理性能模拟器，无需任何第三方依赖。它基于计算复杂度 FLOPs（浮点运算次数）、GPU 计算能力 FLOPS（每秒浮点运算次数）、GPU 内存带宽以及通过基准测试主流 LLM 核心所获得的 MFU（模型浮点利用率），来计算 TTFT（首次生成时间）、TPOT（每令牌处理时间）和吞吐量 TGS（每 GPU 每秒生成的令牌数）。对于多 GPU、多节点部署场景，InferSim 还可根据数据量和带宽估算通信延迟。

InferSim 的主要使用场景包括：
- **模型-系统协同设计**：根据模型超参数预测推理性能。
- **推理性能分析**：量化性能瓶颈（如计算瓶颈或 IO 瓶颈），支持优化工作。

更多详情请参见 [InferSim 技术报告](https://github.com/user-attachments/files/23016438/infersim_tech_report.pdf)。

## 模拟结果

| 模型 | GPU | Prefill TGS（实际） | Prefill TGS（模拟） | Decode TGS（实际） | Decode TGS（模拟） | 备注 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| DeepSeek-V3 | H800 | 7839 | 9034 | 2324 | 2675 | 实际数据来自 [deepseek/profile-data](https://github.com/deepseek-ai/profile-data/)。使用相同设置进行模拟：[example/deepseek-v3/](./example/deepseek-v3/)。 |
| Qwen3-30B-A3B-BF16 | H20 | 16594 | 17350 | 2749 | 2632 | 实际数据使用 SGLang 测试，模拟示例：[example/qwen3-30B-A3B/](./example/qwen3-30B-A3B/)。 |
| Qwen3-8B-FP8 | H20 | 15061 | 16328 | 2682 | 2581 | 实际数据使用 SGLang 测试，模拟示例：[example/qwen3-8B/](./example/qwen3-8B/)。 |

## 支持的功能

- **注意力机制**：MHA/GQA、MLA。基于 FlashInfer、FlashAttention-3、FlashMLA 基准测试。
- **MoE（专家混合）**：GroupedGEMM。基于 DeepGEMM 基准测试。
- **线性层**：GEMM。基于 DeepGEMM 基准测试。
- **并行化策略**：DP Attn（数据并行注意力）、EP MoE（专家并行 MoE）。
- **大规模 EP**：支持 DeepEP 的 dispatch 和 combine 阶段，包含正常模式和低延迟模式。

## 帮助信息

```
$ python3 main.py --help
usage: main.py [-h] --config-path CONFIG_PATH [--device-type {H20,H800}] [--world-size WORLD_SIZE] [--num-nodes NUM_NODES]
               [--max-prefill-tokens MAX_PREFILL_TOKENS] [--decode-bs DECODE_BS] [--target-tgs TARGET_TGS]
               [--target-tpot TARGET_TPOT] [--target-isl TARGET_ISL] [--target-osl TARGET_OSL] [--use-fp8-gemm]
               [--use-fp8-kv] [--enable-deepep] [--enable-tbo] [--sm-ratio SM_RATIO] [--prefill-only] [--decode-only]

可选参数：
  -h, --help            显示帮助信息并退出
  --config-path CONFIG_PATH
                        Hugging Face 模型 config.json 文件路径
  --device-type {H20,H800}
                        设备类型
  --world-size WORLD_SIZE
                        GPU 数量
  --num-nodes NUM_NODES
                        节点数量
  --max-prefill-tokens MAX_PREFILL_TOKENS
                        最大预填充令牌数
  --decode-bs DECODE_BS
                        解码批次大小。若未指定，则 bs = tgs * tpot。
  --target-tgs TARGET_TGS
                        目标每 GPU 每秒生成令牌数（TGS）
  --target-tpot TARGET_TPOT
                        TPOT（每令牌处理时间），单位为毫秒
  --target-isl TARGET_ISL
                        输入序列长度，单位为令牌数
  --target-osl TARGET_OSL
                        输出序列长度，单位为令牌数
  --use-fp8-gemm        使用 FP8 GEMM 计算
  --use-fp8-kv          使用 FP8 KV Cache
  --enable-deepep       启用 DeepEP
  --enable-tbo          启用双批次重叠（Two Batch Overlap）
  --sm-ratio SM_RATIO   在 TBO DeepEP 正常模式下用于计算的 SM 占比
  --prefill-only        仅模拟预填充阶段
  --decode-only         仅模拟解码阶段
```

## 示例

```
$ bash example/qwen3-30B-A3B/decode.sh

================ 模拟器结果 ================
设备类型：                              H20
GPU 数量：                               4
注意力机制类型：                         MHA/GQA
使用 FP8 GEMM：                          0
使用 FP8 KV Cache：                      0
------------------模型权重-------------------
单个注意力层参数大小（MB）：             36.00
单个专家参数大小（MB）：                 9.00
每 GPU 参数大小（GB）：                  15.19
---------------------KV Cache---------------
KV Cache 空间（GB）：                   60.81
输入序列长度：                           4096
输出序列长度：                           2048
目标解码批次大小：                       100
目标每令牌 KV Cache 大小（KB）：         103.79
当前每令牌 KV Cache 大小（KB）：         96.00
----------------------FLOPs-----------------------
隐藏层数量：                             48
每令牌每层注意力核心（GFLOPs）：         0.08
每令牌每层 MoE/FFN（GFLOPs）：           0.08
每令牌每层其他部分（GFLOPs）：           0.04
每令牌注意力核心总（GFLOPs）：           4.03
每令牌 MoE 总（GFLOPs）：                3.62
每令牌其他部分总（GFLOPs）：             1.81
每令牌总计算量（GFLOPs）：               9.46
---------------------解码阶段---------------------
注意力核心 MFU：                         0.15
注意力核心延迟（微秒）：                 361.77
KV 加载延迟（微秒）：                    298.02
QKV_proj 延迟（微秒）：                  31.03
O_proj 延迟（微秒）：                    16.95
路由专家/FFN MFU：                      0.18
路由专家/FFN 延迟（微秒）：              269.28
专家加载延迟（微秒）：                   85.83
MoE/FFN 前通信延迟（微秒）：             4.24
MoE/FFN 后通信延迟（微秒）：             4.24
TPOT（毫秒）：                           38.00
吞吐量（TGS）：                          2632
```

## 致谢

本项目由阿里巴巴集团 Alimama AI Infra 团队与 Future Living Lab 共同开发与维护。
