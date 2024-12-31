import torch
import time

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

# 示例微分方程：dy/dt = -y + sin(t)
def example_func(t, y):
    return -y + torch.sin(t)

# 初始条件
y0 = torch.tensor(torch.rand(10))  # 初始值 y(0) = 1
t0 = 0.0                 # 起始时间
t_end = 10.0             # 结束时间
h = 0.01                  # 时间步长

# 测量运行时间
start_time = time.time()

# 使用向量化 RK4 方法
t_values, y_values = rk4(example_func, y0, t0, t_end, h)

end_time = time.time()
print(f"运行时间: {end_time - start_time:.6f} 秒")

# 可视化结果（需要 matplotlib）
import matplotlib.pyplot as plt
plt.plot(t_values.numpy(), y_values.numpy(), label="RK4 Solution")
plt.xlabel("Time t")
plt.ylabel("y")
plt.title("Runge-Kutta 4th Order (Optimized)")
plt.legend()
plt.grid()
plt.show()