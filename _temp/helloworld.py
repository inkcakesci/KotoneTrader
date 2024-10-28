# 用于测试Torch是否能正常调用Cuda资源的测试脚本


import torch
import time

# 设置矩阵的维度
N = 10000

# 生成随机矩阵
A = torch.rand(N, N)
B = torch.rand(N, N)

print(torch.cuda.is_available())

# CPU 计算
start_time_cpu = time.time()
C_cpu = torch.mm(A, B)
end_time_cpu = time.time()
cpu_time = end_time_cpu - start_time_cpu
print(f"CPU 计算时间: {cpu_time:.6f} 秒")

# 将矩阵转移到 GPU
A_gpu = A.to('cuda')
B_gpu = B.to('cuda')

# GPU 计算
start_time_gpu = time.time()
C_gpu = torch.mm(A_gpu, B_gpu)
end_time_gpu = time.time()
gpu_time = end_time_gpu - start_time_gpu
print(f"GPU 计算时间: {gpu_time:.6f} 秒")

# 验证结果
C_cpu_np = C_cpu.cpu().numpy()
C_gpu_np = C_gpu.cpu().numpy()