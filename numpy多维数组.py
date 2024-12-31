import numpy as np
import re

# 定义一个示例函数
def convert_to_matrix_index_string(a):
    # 将输入数组转换为索引字符串
    nested_string = f"[{']['.join(map(str, a))}]"
    
    # 使用正则表达式提取所有数字
    indices = tuple(map(int, re.findall(r'\d+', nested_string)))
    
    return indices

# 创建一个 5x5x5 的三维 NumPy 数组
matrix = np.zeros([5, 5, 5])

# 示例数组
a = [2, 3, 4]

# 获取索引元组
index_tuple = convert_to_matrix_index_string(a)

# 使用元组作为索引访问矩阵
matrix[index_tuple] = 1

print(matrix)