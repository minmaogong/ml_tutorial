"""
    实现通用的感知机类
"""
import numpy as np


class Perceptron:
    # 初始化, 传入权重w（向量）和偏置b（标量）
    def __init__(self, w, b):
        self.w = w
        self.b = b

    # 预测方法，传入x（向量）
    def predict(self, x):
        # 两个向量作点积运算，就是将两个向量对应位置的元素相乘后再求和 （w1*x1 + w2*x2 + w3*x3 + w4*x4 ...）
        res = self.w @ x + self.b # 线性加权求和
        # 判断是否超过阈值
        if res <= 0:
            return 0
        else:
            return 1


if __name__ == '__main__':
    # 定义输入向量x
    x1 = np.array([0, 0])
    x2 = np.array([0, 1])
    x3 = np.array([1, 0])
    x4 = np.array([1, 1])

    # 1. 与门
    w = np.array([0.5, 0.5]) # 权重向量
    b = -0.7 # 偏置
    and_gate = Perceptron(w=w, b=b)
    print("与门输出结果：")
    print(and_gate.predict(x=x1))  # 0
    print(and_gate.predict(x=x2))  # 0
    print(and_gate.predict(x=x3)) # 0
    print(and_gate.predict(x=x4))  # 1

    # 2. 与非门
    w = np.array([-0.5, -0.5])
    b = 0.7
    nand_gate = Perceptron(w=w, b=b)
    print("与非门输出结果：")
    print(nand_gate.predict(x=x1))
    print(nand_gate.predict(x=x2))
    print(nand_gate.predict(x=x3))
    print(nand_gate.predict(x=x4))

    # 3. 或门
    w = np.array([0.5, 0.5])
    b = 0
    or_gate = Perceptron(w=w, b=b)
    print("或门输出结果：")
    print(or_gate.predict(x=x1))
    print(or_gate.predict(x=x2))
    print(or_gate.predict(x=x3))
    print(or_gate.predict(x=x4))
