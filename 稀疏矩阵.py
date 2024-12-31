import numpy as np
from scipy.sparse import coo_matrix

# 假设我们创建一个4维稀疏矩阵，形状为 (3, 3, 3, 3)，但只有少数非零元素
shape = (3, 3, 3, 3)

# 定义非零元素的值
data = np.array([1, 2, 3, 4, 5])

# 定义非零元素的索引
row_indices = np.array([0, 1, 2, 0, 1])  # 第1维的索引
col_indices = np.array([0, 1, 2, 1, 0])  # 第2维的索引
depth_indices = np.array([0, 0, 1, 1, 2])  # 第3维的索引
channel_indices = np.array([0, 1, 0, 2, 1])  # 第4维的索引

# 将索引和数据传递给 coo_matrix 来创建稀疏矩阵
sparse_matrix = coo_matrix((data, (row_indices, col_indices, depth_indices, channel_indices)), shape=shape)

print("高维稀疏矩阵（四维）：")
print(sparse_matrix)
