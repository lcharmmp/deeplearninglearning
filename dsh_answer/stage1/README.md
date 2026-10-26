# 阶段一：Python / NumPy / Pandas 速成

对应 `学习路线.md` 的「阶段 1」。三个独立小练习，把深度学习最常用的三个库先混个脸熟：

- **numpy** —— 数值计算，深度学习的地基（张量、矩阵运算）
- **pandas** —— 表格数据处理，后面 Kaggle 阶段全靠它
- **matplotlib** —— 画图，后面画 loss 曲线、看数据分布都靠它

## 目录结构

```
stage1/
├── README.md
├── ex1_numpy_matmul/       练习1：NumPy 矩阵乘法与结合律验证
│   └── matmul.py
├── ex2_pandas_csv_stats/   练习2：Pandas 读取 CSV 并统计缺失值/均值
│   ├── house_prices.csv     样例房价数据（含缺失值）
│   └── csv_stats.py
└── ex3_matplotlib_sine/    练习3：Matplotlib 画正弦曲线
    └── sine_curve.py
```

## 怎么运行

在 PyCharm 里打开对应文件直接运行，或在终端执行：

```bash
conda activate deplrng
cd dsh_answer/stage1
python ex1_numpy_matmul/matmul.py
python ex2_pandas_csv_stats/csv_stats.py
python ex3_matplotlib_sine/sine_curve.py
```

## 每个练习做什么

1. **ex1（matmul.py）**：创建随机矩阵 A/B/C，用 `@` 做矩阵乘法，验证结合律 `(A@B)@C == A@(B@C)`；顺带区分 `@`（矩阵乘法）和 `*`（逐元素乘法）。
2. **ex2（csv_stats.py）**：读取样例房价 CSV，统计每列缺失值数量和均值；再用列均值填充缺失值，观察前后变化。数据是样例数据，学完后可以换成真实数据集（如 Kaggle 的 house prices）再跑一遍。
3. **ex3（sine_curve.py）**：用 numpy 生成 sin 曲线数据，matplotlib 画图并保存为 PNG。想本地弹窗看图的话，把 `matplotlib.use("Agg")` 注释掉、取消 `plt.show()` 的注释。

## 完成后自检（来自学习路线）

- [ ] 能不看文档写出：「读 CSV → 按条件筛选 → 转 NumPy → 矩阵运算」的 20 行脚本
- [ ] 能说出：`@` 和 `*` 的区别、`isna()` 是做什么的、`linspace` 是做什么的
- [ ] 三个脚本都能一次跑通，没有报错
