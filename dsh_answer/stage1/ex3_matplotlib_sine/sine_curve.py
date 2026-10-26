"""
阶段一 · 练习 3：Matplotlib 画正弦曲线

任务：
    画出 y = sin(x) 在 [0, 2π] 上的曲线，
    加上标题、坐标轴标签、网格、图例，并保存为 PNG 图片。

配套知识点：
    numpy.linspace（生成均匀间隔的点）、plot / title / xlabel / ylabel / grid / legend / savefig

运行： python sine_curve.py
产出： 同一目录下生成 sine_curve.png
"""

from pathlib import Path

import numpy as np
import matplotlib

# Agg 是不需要图形窗口的后端，保证在无界面环境也能保存图片。
# 在本地有屏幕的电脑上想直接弹窗查看：注释掉这一行，并取消文件末尾 plt.show() 的注释。
# matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  必须放在 matplotlib.use() 之后导入


def main():
    # --------------------------------------------------------------
    # 1. 生成数据：0 到 2π 均匀取 200 个点
    # --------------------------------------------------------------
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)

    # --------------------------------------------------------------
    # 2. 绘图
    # --------------------------------------------------------------
    plt.figure(figsize=(8, 5))                     # 画布大小（英寸）
    plt.plot(x, y, label="y = sin(x)", color="C0", linewidth=2)
    plt.title("Sine Curve")
    plt.xlabel("x (radian)")
    plt.ylabel("sin(x)")
    plt.grid(True, linestyle="--", alpha=0.6)      # 虚线网格
    plt.legend()                                    # 显示图例（需要 plot 里有 label）

    # --------------------------------------------------------------
    # 3. 保存图片（dpi 越大越清晰）
    # --------------------------------------------------------------
    out_file = Path(__file__).parent / "sine_curve.png"
    plt.savefig(out_file, dpi=150)
    print("图片已保存到:", out_file)

    # 本地有窗口环境想直接看：取消下面这行的注释
    plt.show()


if __name__ == "__main__":
    main()
