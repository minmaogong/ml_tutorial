"""
    感知机
"""

# 简单逻辑电路 - 与门
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    res = w1 * x1 + w2 * x2
    # 判断是否超过阈值
    if res <= theta:
        return 0
    else:
        return 1

# 与非门
def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    res = w1 * x1 + w2 * x2
    # 判断是否超过阈值
    if res <= theta:
        return 0
    else:
        return 1

# 或门
def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0
    res = w1 * x1 + w2 * x2
    if res <= theta:
        return 0
    else:
        return 1

if __name__ == '__main__':
    print("与门输出结果：")
    print(AND(0, 0))  # 0
    print(AND(0, 1))
    print(AND(1, 0))
    print(AND(1, 1))
    print("与非门输出结果：")
    print(NAND(0, 0))
    print(NAND(0, 1))
    print(NAND(1, 0))
    print(NAND(1, 1))
    print("或门输出结果：")
    print(OR(0, 0))
    print(OR(0, 1))
    print(OR(1, 0))
    print(OR(1, 1))


