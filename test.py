import torch
import sys
print(sys.path)
# 檢查 torch 是否成功導入
print(torch.__file__)  # 顯示 torch 的安裝位置
print(torch.__version__)  # 顯示版本號

# 檢查 CUDA 是否可用（GPU 支持）
print(torch.cuda.is_available())  # 返回 True 表示 GPU 可用

# 檢查設備資訊（CPU/GPU）
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")