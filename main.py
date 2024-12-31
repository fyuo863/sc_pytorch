import torch
import numpy as np
import math
import random
import xgi
import matplotlib.pyplot as plt
import inspect
from tqdm import tqdm
import itertools
from scipy.sparse import csr_matrix
import time

#活动性高的节点更有可能连接其他节点，度分布满足幂律分布





# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为黑体字体，SimHei 是常见的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建一个空超图
H = xgi.Hypergraph()

num_individuals = 1000  # 个体数
time_duration = 10
time_step = 0.01
alpha = 2.0
beta = 2.0
gamma = 2.1
m = int(10)# 每个1000代理最多影响10个其他代理
K = 2

class Simplex_computer:# 单纯形相关
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
            
class Draw_picture:# 绘图相关
    def __init__(self):
        # 获取类中所有的函数
        self.methods = [func for func, obj in inspect.getmembers(Draw_picture, predicate=inspect.isfunction)]
        print(f"类中有 {len(self.methods)} 个函数：", self.methods)

        self.simplexs = []  # 用于存储单纯形，每个单纯形是一个集合
    
    def Draw_Simplex(self, H, ax = False):
        """
        绘制单纯形
        """
        if ax == True:
            #print("🐮")
            return H
        else:
            # 使用 barycenter_spring_layout 布局算法计算节点位置，布局基于春力模型，seed用于固定布局结果
            pos = xgi.barycenter_spring_layout(H, seed=1)

            # 创建一个6x2.5英寸的图形和坐标轴
            fig, ax = plt.subplots(figsize=(6, 2.5))

            # 绘制超图H
            ax, collections = xgi.draw(
                H,  # 超图H
                pos=pos,  # 节点的布局位置
                node_fc=H.nodes.degree,  # 节点的颜色映射：节点度数（连接的超边数）
                edge_fc=H.edges.size,  # 边的颜色映射：超边大小（连接的节点数）
                edge_fc_cmap="viridis",  # 边的颜色映射使用viridis配色方案
                node_fc_cmap="mako_r",  # 节点的颜色映射使用反转的Mako配色方案
            )

            # 从collections中提取节点颜色集合、边颜色集合（中间部分忽略）
            node_col, _, edge_col = collections

            # 为节点度数的颜色映射添加颜色条，并标注为"Node degree"
            plt.colorbar(node_col, label="Node degree")

            # 为超边大小的颜色映射添加颜色条，并标注为"Edge size"
            plt.colorbar(edge_col, label="Edge size")

            # 显示绘制的图形
            plt.show()
            return H
    
    def Draw_colorbar(self):
        print()
    
    def Draw_degree(self, data, ax = False):
        """
        绘制度分布
        """
        # 数据集合
        self.data = data

        # 创建一个空字典来存储每个节点的度数
        self.degree = {}

        # 遍历数据集合，计算每个节点的度数
        for group in data:
            # 计算当前集合内节点的度数
            for node in group:
                if node not in self.degree:
                    self.degree[node] = 0
                self.degree[node] += len(group) - 1  # 节点的度数是它所在集合中其他节点的数量

        # # 输出每个节点的度数
        # for node, deg in self.degree.items():
        #     print(f"节点 {node} 的度数为 {deg}")

        # 将度数按从小到大排序
        self.sorted_degrees = sorted(self.degree.values())

        # 统计每个度数的出现次数
        self.degree_counts = {}
        for deg in self.sorted_degrees:
            if deg not in self.degree_counts:
                self.degree_counts[deg] = 0
            self.degree_counts[deg] += 1

        # 将度数和其出现次数分别作为x和y坐标
        self.x = list(self.degree_counts.keys())
        self.y = list(self.degree_counts.values())

        if ax == True:
            #print("🐮")
            return self.x, self.y
        else:
            # 绘制散点图
            plt.figure(figsize=(8, 6))
            #plt.scatter(x, y, color='red')
            # 创建折线图
            plt.plot(self.x, self.y, marker='o', linestyle='-', color='r', label='折线')
            plt.title(f'度分布(平均度数:{sum(draw.degree_counts) / len(draw.degree_counts)})')
            plt.xlabel('度')
            plt.ylabel('出现次数')
            plt.grid(True)
            plt.show()
            return self.x, self.y
    
    def Draw_log10(self, ax = False):
        """
        绘制对数图
        """
        if ax == True:
            #print("🐮")
            return self.x, self.y
        else:
            self.Draw_degree(simplex.simplexs, True)
            # 对数变换后的图
            plt.figure(figsize=(8, 6))
            #plt.scatter(x, y, color='red')
            # 创建折线图
            plt.plot(self.x, self.y, marker='o', linestyle='-', color='r', label='折线')
            plt.title(f'度分布log10(平均度数:{sum(draw.degree_counts) / len(draw.degree_counts)})')
            plt.xlabel('度log(x)')
            plt.ylabel('出现次数log(y)')
            # 设置x轴和y轴为对数刻度
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            plt.show()

    def Draw_subplots(self):
        """
        绘制子图
        """
        print(len(self.methods) - 2)

        # 根据子图数量计算行列数
        ncols = math.ceil(math.sqrt(len(self.methods) - 2))  # 列数为子图数量的平方根
        nrows = math.ceil((len(self.methods) - 2) / ncols)  # 行数为子图数量除以列数的向上取整
        
        fig, self.axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
        print(self.axes)
        # 如果只有一个子图，axes 会是单一的 Axes 对象，需转为列表处理
        if len(self.methods) - 2 == 1:
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flatten()  # 将二维数组展平为一维
        print(self.axes)
        # 处理子图的排列
        self.axes = self.axes.flatten()  # 将二维数组展平为一维
        
        # 1
        self.temp = draw.Draw_Simplex(H, True)
        #pos = xgi.barycenter_spring_layout(self.temp, seed=1)
        ax, collections = xgi.draw(
        self.temp,
        node_fc=self.temp.nodes.degree,
        edge_fc=self.temp.edges.size,
        edge_fc_cmap="viridis",
        node_fc_cmap="mako_r",
        ax = self.axes[0])

        # 2
        node_col, _, edge_col = collections

        plt.colorbar(node_col, label="Node degree", ax=self.axes[1])
        plt.colorbar(edge_col, label="Edge size", ax=self.axes[1])
        self.axes[1].axis('off')  # 关闭所有轴元素

        
        # 3 
        x, y = draw.Draw_degree(simplex.simplexs, True)
        # 在传入的子图ax上绘制折线图
        self.axes[2].plot(x, y, marker='o', linestyle='-', color='r', label='折线')
        self.axes[2].set_title(f'度分布(平均度数:{sum(self.degree_counts.values()) / len(self.degree_counts)})')
        self.axes[2].set_xlabel('度')
        self.axes[2].set_ylabel('出现次数')
        self.axes[2].grid(True)
        
        # 4
        self.axes[3].plot(self.x, self.y, marker='o', linestyle='-', color='r', label='折线')
        self.axes[3].set_title(f'度分布log10(平均度数:{sum(self.degree_counts.values()) / len(self.degree_counts)})')
        self.axes[3].set_xlabel('度log(x)')
        self.axes[3].set_ylabel('出现次数log(xy)')
        # 设置x轴和y轴为对数刻度
        self.axes[3].set_xscale('log')
        self.axes[3].set_yscale('log')
        self.axes[3].grid(True)
        
        
        # 隐藏多余的子图
        for i in range(len(self.methods) - 2, len(self.axes)):
            self.axes[i].axis('off')

        plt.tight_layout()
        plt.show()
    

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

def generate_high_dim_matrices(n):
    """
    生成高维矩阵的列表。
    高维矩阵列表的行列为其本身
    参数：
    n (int): 矩阵的维度。

    返回：
    list: 生成的矩阵列表。
    """
    matrices = []
    
    for dim in range(2, n + 1):
        # if dim == 2:
        #     # 生成一个dim维的矩阵，这里使用的是一个固定大小，比如3
        #     shape = (num_individuals,) * dim  # 假设矩阵的每一维都是3，形状可以根据需求调整
        #     matrix = np.zeros(shape)  # 生成随机数填充矩阵
        #     matrices.append(matrix)
        # else:
        # 生成一个dim维的矩阵，这里使用的是一个固定大小，比如3
        shape = (n,) * dim  # 假设矩阵的每一维都是3，形状可以根据需求调整
        matrix = np.zeros(shape)  # 生成随机数填充矩阵
        matrices.append(matrix)
    
    return matrices

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
    y_values[0] = torch.tensor(y0, dtype=torch.float32)
    
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
    global matrices, K, alpha
    matrices_tensor = [torch.tensor(matrix, dtype=torch.float32) for matrix in matrices]
    # 计算与邻接矩阵相关的项
    temp = torch.arange(y.shape[0], dtype=torch.long)  # 使用索引
    print(temp)
    interaction = K * torch.sum(matrices_tensor[0][:, temp] * torch.tanh(alpha * t), dim=1)  # 批量处理
    #print(interaction)
    # 微分方程
    return -y + interaction

if __name__ == '__main__':
    # 实例化
    simplex = Simplex_computer()
    draw = Draw_picture()
    # 初始化意见数组与0时刻意见
    opinions = np.zeros((num_individuals, int(time_duration / time_step)))#num_individuals行，10000列
    opinions[:, 0] = np.random.uniform(-1, 1, num_individuals)
    #主循环
    for tick in tqdm(range(1, int(time_duration / time_step))):
        if tick > 1:# 测试
            break
        print(f"当前tick{tick}")
        #1. 清空所有单纯形
        simplex.del_all_simplexs()
        #2.激活节点
        for item in tqdm(range(num_individuals)):
            #print(f"当前节点{item}")
            a_list = activity_get(num_individuals)#计算活动性
            if random.uniform(0, 1) <= a_list[item]:
                #激活当前节点，当前节点选择节点进行连接(根据同质性)
                #print(f"当前节点{item}活跃")
                #获取同质性
                homophily = homophily_get(opinions[:, tick - 1], item)
                #根据同质性选择m个节点进行连接
                selected_indices = np.random.choice(num_individuals, size=m, replace=False, p=homophily)
                #print("m个节点",selected_indices)
                #尝试连接这m个节点
                homogeneity_values = np.array([homophily[value] for value in selected_indices])
                selected_agents = selected_indices[np.random.uniform(0, 1, size=len(selected_indices)) <= homogeneity_values]
                #print("连接的节点",selected_agents)
                if list(selected_agents) != []:
                    simplex.add_simplexs(item, selected_agents)
        #3.给矩阵赋值
        # 定义2至最高维度的数组
        print(f"包含节点最多的单纯形：{max(simplex.simplexs, key=len)}，长度为{len(max(simplex.simplexs, key=len))}")
        #matrices = generate_high_dim_matrices(len(max(simplex.simplexs, key=len)))

        # 数组赋值
        for item in simplex.simplexs:# 遍历每个单纯形
            print("----------")
            print(item)
            print(len(item))
            # 计算所有组合
            combinations = []
            for r in range(2, len(item) + 1):
                combinations.extend(itertools.permutations(item, r))
            
            matrices = generate_high_dim_matrices(len(max(combinations, key=len)))
            for combo in combinations:
                print(combo)
                # 找到对应的矩阵
                #print(matrices[len(combo) - 2])
                # if len(combo) == 2:# 二维数组填进A，以后统一计算
                #     index = [range(num_individuals).index(x) for x in combo]
                # else:# 高维数组直接计算
                index = [list(item).index(x) for x in combo]
                matrices[len(set(combo)) - 2][tuple(index)] = 1
                print(matrices[len(combo) - 2])
                print(combinations)
                print(len(max(combinations, key=len)))
                print(item)
                print(index)
                #time.sleep(3)
            #4.意见交换(龙格库塔四阶)
            # 初始条件
            new_arr = np.take(opinions[:, tick - 1], list(item))
            y0 = torch.tensor(new_arr)  # 初始值 y(0) = 随机值
            t0 = 0.0  # 起始时间
            t_end = 1.0  # 结束时间
            h = 0.1  # 时间步长
            # 使用向量化 RK4 方法
            t_values, y_values = rk4(f, y0, t0, t_end, h)
            print(y_values[-1])
            opinions[(list(item)), tick] = y_values[-1]



        

        #print("所有单纯形",simplex.simplexs)

            
    #simplex.display_simplexs()
    H.add_edges_from(simplex.simplexs)

    
    draw.Draw_Simplex(H)# 绘制单纯形

    # draw.Draw_simplitical_complex(H)# 绘制单纯形+节点

    # draw.Draw_degree(simplex.simplexs)# 绘制度分布

    draw.Draw_log10()# 绘制度分布log10

    # draw.Draw_subplots()# 全绘制

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
