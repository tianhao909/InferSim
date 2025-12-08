import csv  # 导入csv模块，用于读取CSV文件
import os  # 导入os模块，用于文件路径操作

from hardware.gpu import gpu_map  # 从hardware.gpu模块导入gpu_map，用于根据设备类型获取GPU信息


def get_attn_decode_mfu(config, target_bs, kv_len, device_type, use_fp8_kv):
    gpu = gpu_map[device_type]  # 根据设备类型获取对应的GPU对象
    if config.attn_type == "MHA/GQA":  # 如果注意力类型是MHA/GQA
        head_dim = config.head_dim  # 获取头维度
        file_name = f"bench_data/mha/decode/{device_type.lower()}/{config.num_attention_heads}-{config.num_key_value_heads}-{head_dim}.csv"  # 构建MHA/GQA解码阶段的基准数据文件路径
    elif config.attn_type == "MLA":  # 如果注意力类型是MLA
        head_dim = f"{config.kv_lora_rank}-{config.qk_rope_head_dim}"  # MLA的头维度由kv_lora_rank和qk_rope_head_dim组合表示
        file_name = f"bench_data/mla/decode/{device_type.lower()}/{config.num_attention_heads}-{head_dim}.csv"  # 构建MLA解码阶段的基准数据文件路径
    if not os.path.exists(file_name):  # 如果文件不存在
        print(f"warning: {file_name} not exists")  # 打印警告信息
        return gpu.mfu  # 返回该GPU的默认MFU（模型浮点运算利用率）

    # CSV文件格式：dtype,kv_dtype,batch_size,kv_len,latency,mfu
    kv_dtype = "fp8" if use_fp8_kv else "bf16"  # 根据use_fp8_kv决定KV缓存的数据类型
    rows = list()  # 初始化存储匹配行的列表
    with open(file_name, "r") as f:  # 打开CSV文件
        reader = csv.reader(f)  # 创建CSV读取器
        next(reader)  # 跳过表头
        for row in reader:  # 遍历每一行
            if row[1] != kv_dtype:  # 如果KV数据类型不匹配
                continue  # 跳过该行
            rows.append(row)  # 将匹配的行加入列表

    mfu_bs = 1  # 初始化最接近但不超过目标batch size的batch size
    for row in rows:  # 遍历所有匹配的行
        bs = int(row[2])  # 获取当前行的batch size
        if bs <= target_bs:  # 如果当前batch size不超过目标值
            mfu_bs = bs  # 更新为当前batch size
        else:  # 由于数据按batch size升序排列，一旦超过即可跳出
            break

    mfu_kv_len = 1  # 初始化最接近但不超过目标kv_len的kv长度
    for row in rows:  # 遍历所有匹配的行
        kv_l = int(row[3])  # 获取当前行的kv长度
        if kv_l <= kv_len:  # 如果当前kv长度不超过目标值
            mfu_kv_len = kv_l  # 更新为当前kv长度
        else:  # 同样假设数据按kv_len升序排列
            break

    mfu = gpu.mfu  # 默认MFU为GPU的默认值
    for row in rows:  # 再次遍历所有匹配的行
        bs = int(row[2])  # 获取batch size
        kv_l = int(row[3])  # 获取kv长度
        if bs == mfu_bs and kv_l == mfu_kv_len:  # 找到同时匹配batch size和kv长度的行
            mfu = float(row[5])  # 读取对应的MFU值

    return round(mfu, 3)  # 返回四舍五入到小数点后3位的MFU


def get_attn_prefill_mfu(config, seq_len, device_type):
    gpu = gpu_map[device_type]  # 根据设备类型获取GPU对象
    if config.attn_type == "MHA/GQA":  # 如果注意力类型是MHA/GQA
        head_dim = config.head_dim  # 获取头维度
        file_name = f"bench_data/mha/prefill/{device_type.lower()}/{config.num_attention_heads}-{config.num_key_value_heads}-{head_dim}.csv"  # 构建MHA/GQA预填充阶段的基准数据文件路径
    elif config.attn_type == "MLA":  # 如果注意力类型是MLA
        head_dim = f"{config.qk_nope_head_dim}-{config.qk_rope_head_dim}"  # MLA预填充阶段的头维度由qk_nope_head_dim和qk_rope_head_dim组合
        file_name = f"bench_data/mla/prefill/{device_type.lower()}/{config.num_attention_heads}-{head_dim}.csv"  # 构建MLA预填充阶段的基准数据文件路径
    if not os.path.exists(file_name):  # 如果文件不存在
        print(f"warning: {file_name} not exist.")  # 打印警告信息
        return 0.9  # 返回默认MFU值0.9（注意：此处与decode不同，使用固定值而非gpu.mfu）

    # CSV文件格式：dtype,seq_len,latency_us,mfu
    rows = list()  # 初始化存储行的列表
    with open(file_name, "r") as f:  # 打开CSV文件
        reader = csv.reader(f)  # 创建CSV读取器
        next(reader)  # 跳过表头
        for row in reader:  # 遍历每一行
            rows.append(row)  # 将所有行加入列表（未按seq_len过滤，后续处理）

    mfu = gpu.mfu  # 默认MFU为GPU的默认值
    # mfu_seq_len = 1  # 注释掉的变量，不再使用
    for row in rows:  # 遍历所有行（假设按seq_len升序排列）
        sql = int(row[1])  # 获取当前行的序列长度
        if sql <= seq_len:  # 如果当前序列长度不超过目标值
            # mfu_seq_len = sql  # 更新匹配的序列长度（注释掉）
            mfu = float(row[3])  # 更新MFU为当前行的值（取最后一个不超过seq_len的值）
        else:  # 一旦超过目标seq_len，跳出循环
            break

    return round(mfu, 3)  # 返回四舍五入到小数点后3位的MFU


def get_groupedgemm_decode_mfu(config, target_bs, device_type, num_gpus, use_fp8):
    gpu = gpu_map[device_type]  # 根据设备类型获取GPU对象
    file_name = f"bench_data/grouped_gemm/decode/{device_type.lower()}/data.csv"  # 构建GroupedGEMM解码阶段的基准数据文件路径
    if not os.path.exists(file_name):  # 如果文件不存在
        print(f"warning: {file_name} not exists")  # 打印警告信息
        return gpu.mfu, gpu.mfu  # 返回两个默认MFU值（up和down）

    # CSV文件格式：num_experts,num_gpus,num_local_experts,topk,hidden_size,intermediate_size,batch_size_per_gpu,tokens_per_expert,up_proj_us,up_mfu,down_proj_us,down_mfu
    rows = list()  # 初始化存储匹配行的列表
    with open(file_name, "r") as f:  # 打开CSV文件
        reader = csv.reader(f)  # 创建CSV读取器
        next(reader)  # 跳过表头
        for row in reader:  # 遍历每一行
            if int(row[0]) != config.num_routed_experts:  # 专家总数不匹配则跳过
                continue
            if int(row[1]) != num_gpus:  # GPU数量不匹配则跳过
                continue
            if int(row[3]) != config.num_experts_per_tok:  # 每token选择的专家数不匹配则跳过
                continue
            if int(row[4]) != config.hidden_size:  # 隐藏层维度不匹配则跳过
                continue
            if int(row[5]) != config.intermediate_size:  # 中间层维度不匹配则跳过
                continue
            rows.append(row)  # 将匹配的行加入列表

    mfu1 = gpu.mfu  # up_proj的默认MFU
    mfu2 = gpu.mfu  # down_proj的默认MFU
    for row in rows:  # 遍历所有匹配的行（假设按batch_size_per_gpu升序排列）
        bs = int(row[6])  # 获取每GPU的batch size
        if bs <= target_bs:  # 如果当前batch size不超过目标值
            mfu1 = float(row[9])  # 更新up_proj的MFU
            mfu2 = float(row[11])  # 更新down_proj的MFU
        else:  # 一旦超过目标batch size，跳出循环
            break

    return round(mfu1, 3), round(mfu2, 3)  # 返回两个四舍五入到小数点后3位的MFU值


def get_groupedgemm_prefill_mfu(config, seq_len, device_type, num_gpus, use_fp8):
    gpu = gpu_map[device_type]  # 根据设备类型获取GPU对象
    file_name = f"bench_data/grouped_gemm/prefill/{device_type.lower()}/data.csv"  # 构建GroupedGEMM预填充阶段的基准数据文件路径
    if not os.path.exists(file_name):  # 如果文件不存在
        print(f"warning: {file_name} not exists")  # 打印警告信息
        return gpu.mfu, gpu.mfu  # 返回两个默认MFU值

    # CSV文件格式：num_experts,num_gpus,num_local_experts,topk,hidden_size,intermediate_size,seq_len_per_gpu,tokens_per_expert,up_proj_us,up_mfu,down_proj_us,down_mfu
    rows = list()  # 初始化存储匹配行的列表
    with open(file_name, "r") as f:  # 打开CSV文件
        reader = csv.reader(f)  # 创建CSV读取器
        next(reader)  # 跳过表头
        for row in reader:  # 遍历每一行
            if int(row[0]) != config.num_routed_experts:  # 专家总数不匹配则跳过
                continue
            if int(row[1]) != num_gpus:  # GPU数量不匹配则跳过
                continue
            if int(row[3]) != config.num_experts_per_tok:  # 每token选择的专家数不匹配则跳过
                continue
            if int(row[4]) != config.hidden_size:  # 隐藏层维度不匹配则跳过
                continue
            if int(row[5]) != config.intermediate_size:  # 中间层维度不匹配则跳过
                continue
            rows.append(row)  # 将匹配的行加入列表

    mfu1 = gpu.mfu  # up_proj的默认MFU
    mfu2 = gpu.mfu  # down_proj的默认MFU
    for row in rows:  # 遍历所有匹配的行（假设按seq_len_per_gpu升序排列）
        sql = int(row[6])  # 获取每GPU的序列长度
        if sql <= seq_len:  # 如果当前序列长度不超过目标值
            mfu1 = float(row[9])  # 更新up_proj的MFU
            mfu2 = float(row[11])  # 更新down_proj的MFU
        else:  # 一旦超过目标序列长度，跳出循环
            break

    return round(mfu1, 3), round(mfu2, 3)  # 返回两个四舍五入到小数点后3位的MFU值


def get_gemm_mfu(device_type, m, k, n):
    gpu = gpu_map[device_type]  # 根据设备类型获取GPU对象
    file_name = f"bench_data/gemm/{device_type.lower()}/data.csv"  # 构建GEMM基准数据文件路径
    if not os.path.exists(file_name):  # 如果文件不存在
        print(f"warning: {file_name} not exists")  # 打印警告信息
        return gpu.mfu  # 返回默认MFU

    mfu_k = 0  # 初始化最接近的k维度
    mfu_n = 0  # 初始化最接近的n维度
    dist = 1e9  # 初始化最小距离（用于寻找最接近的k,n组合）
    # CSV文件格式：m,k,n,latency_us,mfu
    rows = list()  # 初始化存储行的列表
    with open(file_name, "r") as f:  # 打开CSV文件
        reader = csv.reader(f)  # 创建CSV读取器
        next(reader)  # 跳过表头
        for row in reader:  # 遍历每一行
            k_ = int(row[1])  # 获取当前行的k维度
            n_ = int(row[2])  # 获取当前行的n维度
            if k_ < k or n_ < n:  # 只考虑k和n都不小于目标值的行（确保能覆盖目标尺寸）
                continue
            if (k - k_) ** 2 + (n - n_) ** 2 < dist:  # 计算欧氏距离平方，寻找最接近的(k_, n_)
                dist = (k - k_) ** 2 + (n - n_) ** 2  # 更新最小距离
                mfu_k = k_  # 更新最接近的k
                mfu_n = n_  # 更新最接近的n
            rows.append(row)  # 将所有行加入列表（包括不满足k>=k或n>=n的，但后续会过滤）

    mfu = gpu.mfu  # 默认MFU为GPU的默认值
    for row in rows:  # 遍历所有行
        m_ = int(row[0])  # 获取当前行的m维度
        k_ = int(row[1])  # 获取当前行的k维度
        n_ = int(row[2])  # 获取当前行的n维度
        if k_ == mfu_k and n_ == mfu_n and m_ <= m:  # 找到匹配最接近的k,n且m不超过目标m的行
            mfu = float(row[4])  # 读取对应的MFU值

    return round(mfu, 3)  # 返回四舍五入到小数点后3位的MFU