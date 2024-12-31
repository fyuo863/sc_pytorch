import numpy as np

def generate_high_dim_matrices(n):
    matrices = []
    
    for dim in range(2, n + 1):
        # 生成一个dim维的矩阵，这里使用的是一个固定大小，比如3
        shape = (n,) * dim  # 假设矩阵的每一维都是3，形状可以根据需求调整
        matrix = np.zeros(shape)  # 生成随机数填充矩阵
        matrices.append(matrix)
    
    return matrices

# 示例：生成从2维到3维的矩阵
n = 2
matrices = generate_high_dim_matrices(n)

# 打印结果
for i, matrix in enumerate(matrices, 2):
    print(f"{i}维列表:")
    print(matrix)
    print("=" * 20)
