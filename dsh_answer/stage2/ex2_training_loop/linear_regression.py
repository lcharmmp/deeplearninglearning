"""
阶段二 · 练习 2：完整训练套路（线性回归，合成数据，不需要联网下载）

任务：
    用 PyTorch 从头写一遍"训练的标准套路"——深度学习万能模板，背下来：
        定义模型(nn.Module) -> 定义 loss -> 定义优化器 ->
        循环每个 epoch：前向 forward -> 算 loss -> 梯度清零 -> 反向 backward -> 更新 step
    数据用合成的 y = 2x + 1 + 噪声，训练完看看学出来的 w、b 是否接近 2 和 1。

运行： python linear_regression.py
产出： 同一目录下生成 loss_curve.png（loss 随 epoch 下降的曲线）
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")   # 本地有窗口想弹图：注释这行，并取消文件末尾 plt.show() 的注释
import matplotlib.pyplot as plt


def main():
    torch.manual_seed(0)              # 固定随机种子，结果可复现

    # --------------------------------------------------------------
    # 1. 造数据：y = 2x + 1 + 噪声（真实世界的 w=2, b=1）
    # --------------------------------------------------------------
    true_w, true_b = 2.0, 1.0
    x = torch.linspace(-3, 3, 300).unsqueeze(1)           # (300, 1)
    y = true_w * x + true_b + torch.randn_like(x) * 0.3   # 加一点噪声，更像真实数据

    # 打包成数据集，用 DataLoader 分小批（batch）喂给模型
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"数据集 {len(dataset)} 个样本，每批 {loader.batch_size} 个，共 {len(loader)} 批")

    # --------------------------------------------------------------
    # 2. 定义模型：y = w * x + b，参数 w、b 就是要学的东西
    # --------------------------------------------------------------
    class LinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(1))   # nn.Parameter 自带 requires_grad=True
            self.b = nn.Parameter(torch.zeros(1))

        def forward(self, x):                       # 前向传播：输入 -> 输出
            return self.w * x + self.b

    model = LinearRegression()

    # --------------------------------------------------------------
    # 3. 定义 loss（均方误差）和优化器（随机梯度下降 SGD）
    # --------------------------------------------------------------
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # --------------------------------------------------------------
    # 4. 训练循环（核心套路，5 行，背下来）
    # --------------------------------------------------------------
    epochs = 60
    epoch_losses = []
    for epoch in range(epochs):
        model.train()                               # 训练模式（对 dropout/BN 有用，这里无影响）
        total_loss = 0.0
        for xb, yb in loader:                       # 取一批数据
            pred = model(xb)                        # ① 前向：预测
            loss = loss_fn(pred, yb)                # ② 算 loss
            optimizer.zero_grad()                   # ③ 梯度清零（否则会累加）
            loss.backward()                         # ④ 反向传播：算梯度
            optimizer.step()                         # ⑤ 更新参数
            total_loss += loss.item() * len(xb)
        epoch_losses.append(total_loss / len(dataset))

        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:3d}/{epochs} | loss = {epoch_losses[-1]:.5f}")

    # --------------------------------------------------------------
    # 5. 看学到的参数，和真实的 w=2, b=1 对比
    # --------------------------------------------------------------
    print(f"\n学到的 w = {model.w.item():.4f}   (真实值 {true_w})")
    print(f"学到的 b = {model.b.item():.4f}   (真实值 {true_b})")

    # --------------------------------------------------------------
    # 6. 画 loss 曲线（阶段一 matplotlib 的实战运用）
    # --------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses, label="loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Training Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    out_file = Path(__file__).parent / "loss_curve.png"
    plt.savefig(out_file, dpi=150)
    print("loss 曲线已保存到:", out_file)

    # 本地有窗口想直接看：取消下面这行的注释
    # plt.show()


if __name__ == "__main__":
    main()
