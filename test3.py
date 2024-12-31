import numpy as np

# 给定的数组a
a = [2, 4, 8]

# 要替换的元素数组
b = [2, 4, 8]

# 使用列表推导式来找到每个元素在a中的索引
indices = [a.index(x) for x in b]

print(indices)  # 输出: [1, 3, 7]