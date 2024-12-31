import torch
import time

def rk4_step(f, y, t, h):
    """
    执行一次 RK4 步骤
    :param f: 微分方程 dy/dt = f(t, y)
    :param y: 当前状态 (torch.Tensor)
    :param t: 当前时间 (float)
    :param h: 时间步长 (float)
    :return: 更新后的状态
    """
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# 示例微分方程：dy/dt = -y + sin(t)
def example_func(t, y):
    return -y + torch.sin(torch.tensor(t))  # 修正此处

# 初始条件
y0 = torch.tensor([1.0])  # 初始值 y(0) = 1
t0 = 0.0                 # 起始时间
h = 0.1                  # 时间步长
steps = 100              # 总步数

# 测量运行时间
start_time = time.time()

# 数值解
t_values = [t0]
y_values = [y0]
y = y0
t = t0

for _ in range(steps):
    y = rk4_step(example_func, y, t, h)
    t += h
    t_values.append(t)
    y_values.append(y)

# 将结果转换为张量
t_values = torch.tensor(t_values)
y_values = torch.stack(y_values)

end_time = time.time()
print(f"运行时间: {end_time - start_time:.6f} 秒")

# 可视化结果（需要 matplotlib）
import matplotlib.pyplot as plt
plt.plot(t_values.numpy(), y_values.numpy(), label="RK4 Solution")
plt.xlabel("Time t")
plt.ylabel("y")
plt.title("Runge-Kutta 4th Order")
plt.legend()
plt.grid()
plt.show()
