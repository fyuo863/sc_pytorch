import numpy as np
import matplotlib.pyplot as plt

# 假设 opinions 是一个二维数组，每行是不同折线的 y 坐标
opinions = np.array([
    [1, 2, 3, 4, 5],  # 第一条折线的 y 坐标
    [2, 3, 4, 5, 6],  # 第二条折线的 y 坐标
    [3, 4, 5, 6, 7]   # 第三条折线的 y 坐标
])

# 假设每列的 x 坐标
#x = np.array([0, 1, 2, 3, 4])

# 使用 matplotlib 绘制所有折线
plt.figure(figsize=(8, 6))

# 使用 .T 转置将每行转为每列，然后一次性传递给 plt.plot()
plt.plot(range(len(opinions[0, :])), opinions.T)  # opinions.T 会将每列作为一条折线

# 添加标签、标题和图例
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Multiple Lines Plot')
plt.legend([f'Line {i+1}' for i in range(opinions.shape[0])])  # 自动生成图例

# 显示图像
plt.show()