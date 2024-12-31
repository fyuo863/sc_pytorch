import matplotlib.pyplot as plt  # 导入matplotlib库中的pyplot模块，用于绘制图形
import xgi  # 导入xgi库，用于处理超图

# 定义超边列表，每个超边包含多个节点
hyperedges = [[1, 2, 3], [3, 4, 5], [3, 6], [6, 7, 8, 9], [1, 4, 10, 11, 12], [1, 4]]
# 创建超图H，其中包含多个超边
H = xgi.Hypergraph(hyperedges)
#H = xgi.load_xgi_data("coauth-dblp")

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
