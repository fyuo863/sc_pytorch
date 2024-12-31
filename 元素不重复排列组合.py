import itertools

# 定义集合
elements = [30, 120, 250]

# 生成所有组合
combinations = []
for r in range(1, len(elements) + 1):
    combinations.extend(itertools.combinations(elements, r))

# 打印所有组合
for combo in combinations:
    print(set(combo))