import matplotlib.pyplot as plt
import math

def create_subplots(num_subplots):
    # 根据子图数量计算行列数
    ncols = math.ceil(math.sqrt(num_subplots))  # 列数为子图数量的平方根
    nrows = math.ceil(num_subplots / ncols)  # 行数为子图数量除以列数的向上取整
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
    print(axes)
    # 如果只有一个子图，axes 会是单一的 Axes 对象，需转为列表处理
    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # 将二维数组展平为一维
    print(axes)
    # 绘制每个子图
    for i in range(num_subplots):
        ax = axes[i]
        ax.plot([1, 2, 3], [1, 2, 3])  # 示例绘图
        ax.set_title(f"Subplot {i+1}")
    
    # 隐藏多余的子图
    for i in range(num_subplots, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# 示例：创建 1 个子图
create_subplots(5)
