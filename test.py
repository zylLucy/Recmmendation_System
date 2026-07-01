# 测试 MPS 设备
import torch

device = torch.device("mps")

x = torch.randn(10000, 10000).to(device)

print(x.device)