"""
阶段一 · 练习 1：NumPy 矩阵乘法 + 结合律验证

任务：
    用 numpy 完成矩阵乘法，并验证结合律： (A @ B) @ C == A @ (B @ C)

配套知识点：
    ndarray 的创建、shape、随机数、矩阵乘法 @ 与 np.dot

运行： python matmul.py
"""

import numpy as np


def main():
    # ------------------------------------------------------------------
    # 1. 创建三个随机矩阵：A(2×3)、B(3×4)、C(4×5)
    #    矩阵相乘的条件：左边矩阵的列数 == 右边矩阵的行数
    #    所以 A@B 合法（3==3）、B@C 合法（4==4），再套一层也合法
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)        # 固定随机种子 -> 每次运行结果一致，便于对照
    A = rng.random((2, 3))
    B = rng.random((3, 4))
    C = rng.random((4, 5))

    print("A 的 shape:", A.shape, " B 的 shape:", B.shape, " C 的 shape:", C.shape)

    # ------------------------------------------------------------------
    # 2. 验证结合律：(A@B)@C  ==  A@(B@C)
    #    两种加括号方式结果应该完全相同（矩阵乘法满足结合律）
    # ------------------------------------------------------------------
    left = (A @ B) @ C        # 先算 A@B，再和 C 乘
    right = A @ (B @ C)       # 先算 B@C，再和 A 乘

    print("(A@B)@C 的 shape:", left.shape)
    print("A@(B@C) 的 shape:", right.shape)

    # 浮点数有舍入误差，不能用 == 直接比较，用 np.allclose 判断"几乎相等"
    equal = np.allclose(left, right)
    print("结合律成立 (A@B)@C == A@(B@C) ?", equal)

    # ------------------------------------------------------------------
    # 3. 几个新手容易混的点
    # ------------------------------------------------------------------
    # (@) 和 np.dot 是等价的两种写法
    dot_result = np.dot(A, B)
    at_result = A @ B
    print("np.dot(A,B) 与 A@B 等价 ?", np.allclose(dot_result, at_result))

    # (*) 是逐元素乘法：对应位置相乘，两个矩阵 shape 必须相同
    # 它和"矩阵乘法"完全不同，深度学习里到处用到，别搞混
    elementwise = A * A
    print("A*A（逐元素乘法）的 shape:", elementwise.shape, " <- 结果不是 A@A，注意区分")


if __name__ == "__main__":
    main()
