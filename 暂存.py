import torch
import numpy as np
import math
import random

num_individuals = 10  # 个体数
time_step = 10
beta = 2.0
gamma = 2.1

def homophily_get(opinions, node_index):
    """
    计算给定节点的同质性。

    参数：
    opinions (list): 节点的意见列表。
    node_index (int): 目标节点的索引。

    返回：
    float: 目标节点的同质性。
    """
    # 将列表转换为 NumPy 数组
    opinions = np.array(opinions)
    
    # 计算所有节点与目标节点之间的差异(|xi-xj|)，并取倒数
    opinion_diff = np.abs(opinions - opinions[node_index] + 1e-10) ** -beta
    
    # 避免对自己计算同质性，设置自己的同质性为0
    opinion_diff[node_index] = 0
    
    # 计算同质性
    probabilities = opinion_diff / (np.sum(opinion_diff) + 1e-10)
    print(probabilities)
    # 返回目标节点的同质性
    return probabilities


def activity_get(size):
    """
    生成服从幂律分布的活动概率。

    返回：
    list: 生成的活动概率列表。
    """
    alpha = 2.5  # 指数参数
    x0 = 0.01     # 下界
    x1 = 1.0    # 上界
    y = np.random.uniform(0, 1, size)  # 生成 0 到 1 之间的随机数

    return np.power((math.pow(x1, alpha + 1) - math.pow(x0, alpha + 1)) * y + math.pow(x0, alpha + 1), 1 / (alpha + 1))



if __name__ == '__main__':
    # 初始化意见
    opinions = [0.1, 0.3, 0.5, 0.7, 0.9, 0, 1, 0.15, 0.05]
    # 计算同质性
    homogeneity = homophily_get(opinions, 0)
    print(f"同质性: {homogeneity}")
    print(activity_get(10))
