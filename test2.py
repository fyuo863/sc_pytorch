import torch
import time
import matplotlib.pyplot as plt
import numpy as np

def rk4(f, y0, t0, t_end, h):
    """
    使用向量化实现 RK4 数值解法
    :param f: 微分方程 dy/dt = f(t, y)
    :param y0: 初始状态 (torch.Tensor)
    :param t0: 起始时间 (float)
    :param t_end: 结束时间 (float)
    :param h: 时间步长 (float)
    :return: 时间序列和解的序列
    """
    # 创建时间序列
    t_values = torch.arange(t0, t_end + h, h)
    n_steps = len(t_values)
    
    # 初始化结果存储
    y_values = torch.zeros((n_steps, *y0.shape))
    y_values[0] = y0
    
    # 批量计算时间步对应的状态
    for i in range(1, n_steps):
        t = t_values[i - 1]
        y = y_values[i - 1]
        
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        
        y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    return t_values, y_values


# 这里涉及到邻接矩阵 A 和张量索引 temp
def f(t, y):
    # 假设邻接矩阵 A 和一个常数 K
    global A, K, alpha

    # 计算与邻接矩阵相关的项
    temp = torch.arange(y.shape[0], dtype=torch.long)  # 使用索引
    interaction = K * torch.sum(A[:, temp] * torch.tanh(alpha * t), dim=1)  # 批量处理
    #print(interaction)
    # 微分方程
    return -y + interaction


# 初始化参数
num_nodes = 10
K = 2
alpha = 2

# 随机生成一个邻接矩阵
# 用随机数生成0或1，0表示没有连接，1表示有连接
A = np.random.randint(0, 2, size=(num_nodes, num_nodes))

# 为避免自环，将对角线上的元素置为0
np.fill_diagonal(A, 0)
# 将 A 转换为 PyTorch Tensor
A = torch.tensor(A, dtype=torch.float32)

oponions = np.random.uniform(-1, 1, num_nodes)

# 初始条件
y0 = torch.tensor(oponions)  # 初始值 y(0) = 随机值
t0 = 0.0  # 起始时间
t_end = 1.0  # 结束时间
h = 0.1  # 时间步长

# 测量运行时间
start_time = time.time()

# 使用向量化 RK4 方法
t_values, y_values = rk4(f, y0, t0, t_end, h)

end_time = time.time()
print(f"运行时间: {end_time - start_time:.6f} 秒")

# 可视化结果
plt.plot(t_values.numpy(), y_values.numpy())
plt.xlabel("Time t")
plt.ylabel("y")
plt.title("Runge-Kutta 4th Order (Optimized)")
plt.grid()
plt.show()

