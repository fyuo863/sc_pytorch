# 并行计算
import torch

# 示例函数
def my_function(x):
    return x * x  # 示例操作

# 创建张量
arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# 使用向量化操作
result = my_function(arr)

print(result)
