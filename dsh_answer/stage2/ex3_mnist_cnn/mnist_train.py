"""
阶段二 · 练习 3b：MNIST 手写数字识别 —— 完整训练（本阶段主项目）

流程（整个深度学习最标准的套路）：
    读数据(datasets + DataLoader) -> 定义模型 -> 定义 loss 和优化器 ->
    训练循环（每个 epoch：训练 -> 验证 -> 保存最优模型）

运行： python mnist_train.py
说明：
    - 首次运行会自动下载 MNIST 数据集（约 11MB）到 本目录/data，需要联网；
      如果下载卡住/失败，把报错发给我，我教你用国内镜像。
    - CPU 上 5 个 epoch 几分钟跑完；有 N 卡会自动用 GPU。
产出： 验证集准确率最高的模型保存为 mnist_cnn.pt
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN

DATA_ROOT = Path(__file__).parent / "data"


def main():
    # 超参数（调参实验就改这里，README 里有对照表）
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)

    # --------------------------------------------------------------
    # 1. 数据
    #    ToTensor：把 0-255 的像素变成 0-1 的浮点张量，并自动加上通道维 (1,28,28)
    #    Normalize：用数据集的全局均值/标准差标准化。
    #        —— "为什么要归一化"的答案：输入数值范围差异过大时，梯度更新不稳定、
    #           收敛慢；标准化后数据居中且量级一致，训练又快又稳。
    # --------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST 的全局均值 / 标准差
    ])
    train_data = datasets.MNIST(root=str(DATA_ROOT), train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=transform)

    # batch 的作用：一次喂一批，用批内平均梯度更新参数——
    #   比逐个样本稳（梯度不抖）、比整个数据集省内存。batch 越大越接近真梯度，但内存越贵。
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

    # --------------------------------------------------------------
    # 2. 模型 / loss / 优化器
    # --------------------------------------------------------------
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()      # 多分类标配：softmax + 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --------------------------------------------------------------
    # 3. 训练 + 验证
    # --------------------------------------------------------------
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # ---------- 训练 ----------
        model.train()          # 训练模式（这里没有 dropout 等，主要是养成习惯）
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)                        # ① 前向
            loss = loss_fn(pred, labels)                # ② loss
            optimizer.zero_grad()                       # ③ 清零梯度
            loss.backward()                             # ④ 反向传播
            optimizer.step()                            # ⑤ 更新参数
            total_loss += loss.item() * images.size(0)
            correct += (pred.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)
        train_acc = correct / seen * 100

        # ---------- 验证 ----------
        # model.eval() + torch.no_grad()：不更新参数、不存计算图（省内存、提速）
        model.eval()
        correct, seen = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images)
                correct += (pred.argmax(dim=1) == labels).sum().item()
                seen += images.size(0)
        val_acc = correct / seen * 100

        print(f"epoch {epoch}/{EPOCHS} | loss {total_loss / seen:.4f} | "
              f"train_acc {train_acc:.2f}% | val_acc {val_acc:.2f}%")

        # ---------- 保存验证集上最优的模型 ----------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Path(__file__).parent / "mnist_cnn.pt")
            print(f"  -> 新最优，已保存 mnist_cnn.pt（val_acc {val_acc:.2f}%）")

    print(f"\n训练完成，最优 val_acc = {best_acc:.2f}%   （路线里程碑：>= 97%）")


if __name__ == "__main__":
    main()
