"""
阶段一 · 练习 2：Pandas 读取 CSV，统计缺失值与均值

任务：
    1. 用 pandas 读取 house_prices.csv（同一目录下的样例房价数据）
    2. 统计每一列的缺失值数量（和缺失率）
    3. 计算每一列数值列的均值（mean 会自动跳过缺失值）

配套知识点：
    read_csv、shape、head()、isna()、mean()、fillna()、describe()

运行： python csv_stats.py    （脚本会自动定位同目录下的 CSV，在哪个目录运行都行）
"""

from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).parent / "house_prices.csv"


def main():
    # --------------------------------------------------------------
    # 1. 读取 CSV
    # --------------------------------------------------------------
    df = pd.read_csv(DATA_FILE)
    print("数据形状（行, 列）:", df.shape)
    print("\n前 5 行：")
    print(df.head())

    # --------------------------------------------------------------
    # 2. 缺失值统计：isna() 逐格判断是否是 NaN，sum() 按列求和
    # --------------------------------------------------------------
    missing_counts = df.isna().sum()           # 每列缺失的个数
    missing_rates = df.isna().mean() * 100     # 每列缺失率（百分比）
    print("\n每列缺失值数量：")
    print(missing_counts)
    print("\n每列缺失率（%）:")
    print(missing_rates.round(1))

    # --------------------------------------------------------------
    # 3. 均值：mean() 在计算时自动跳过 NaN
    #    numeric_only=True 表示只对数值列求均值（防止对字符串列报错）
    # --------------------------------------------------------------
    means = df.mean(numeric_only=True)
    print("\n每列均值（自动跳过缺失值）：")
    print(means)

    # --------------------------------------------------------------
    # 4. 附加练习：用"列均值"填充缺失值，再看统计结果，对比前后
    # --------------------------------------------------------------
    df_filled = df.fillna(df.mean(numeric_only=True))
    print("\n填充后是否还有缺失值：", df_filled.isna().sum().sum(), "（0 表示没有）")
    print("\n填充前后均值对比：")
    comparison = pd.DataFrame({
        "填充前均值": df.mean(numeric_only=True),
        "填充后均值": df_filled.mean(numeric_only=True),
    })
    print(comparison)

    # --------------------------------------------------------------
    # 5. 附加：describe() 一次给出常用统计量（计数、均值、标准差、分位数）
    # --------------------------------------------------------------
    print("\ndescribe() 汇总：")
    print(df.describe().round(2))


if __name__ == "__main__":
    main()
