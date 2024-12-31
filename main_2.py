import torch
import numpy as np
import math
import random
import xgi
import matplotlib.pyplot as plt

# 创建一个空超图
H = xgi.Hypergraph()

num_individuals = 1000  # 个体数
time_duration = 10
time_step = 0.01
beta = 2.0
gamma = 2.1
m = 10# 每个代理最多影响m个其他代理

class Simplex_computer:
    def __init__(self):
        self.simplexs = []  # 用于存储单纯形，每个单纯形是一个集合

    def add_simplexs(self, agent, nodes):
        """
        添加一个超边，将节点添加到对应的单纯形中
        agent: 当前节点
        nodes: 被连节点集合
        """
        self.simplexs.append(set(np.append(nodes, agent)))

    def del_some_simplexs(self, index):
        """
        删除指定单纯形
        示例[1,2]
        """
        self.simplexs = [s for i, s in enumerate(self.simplexs) if i not in index]

    
    def del_all_simplexs(self):
        """
        删除所有单纯形
        """
        self.simplexs = []

    def display_simplexs(self):
        """
        打印所有单纯形
        """
        for i, edge in enumerate(self.simplexs):
            print(f"Simplex {i + 1}: {edge}")
            
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
    # print(probabilities)
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
    simplex = Simplex_computer()# 实例化
    # 初始化意见数组与0时刻意见
    opinions = np.zeros((num_individuals, int(time_duration / time_step)))#num_individuals行，10000列
    opinions[:, 0] = np.random.uniform(-1, 1, num_individuals)
    #主循环
    for tick in range(1, int(time_duration / time_step)):
        if tick > 1:# 测试
            break
        print(f"当前tick{tick}")
        #1. 清空所有单纯形
        simplex.del_all_simplexs()
        #2.激活节点
        for item in range(num_individuals):
            print(f"当前节点{item}")
            a_list = activity_get(num_individuals)#计算活动性
            if random.uniform(0, 1) <= a_list[item]:
                #激活当前节点，当前节点选择节点进行连接(根据同质性)
                print(f"当前节点{item}活跃")
                #获取同质性
                homophily = homophily_get(opinions[:, tick - 1], item)
                #根据同质性选择m个节点进行连接
                selected_indices = np.random.choice(num_individuals, size=m, replace=False, p=homophily)
                print("m个节点",selected_indices)
                simplex.add_simplexs(item, selected_indices)
            #3.意见交换(龙格库塔四阶)
        print("所有单纯形",simplex.simplexs)

            
    simplex.display_simplexs()
    H.add_edges_from(simplex.simplexs)

    # 绘制超图
    xgi.draw(H)

    # 显示图形
    plt.show()


