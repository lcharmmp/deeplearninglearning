"""
阶段二 · 练习 3c：加载训练好的模型，评估 + 可视化预测

任务：
    加载 mnist_train.py 训练出的 mnist_cnn.pt，
    在测试集上计算准确率，并画出 6 张图的预测结果。

运行： python mnist_eval.py    （需先运行过 mnist_train.py 生成 mnist_cnn.pt）
产出： prediction.png（6 张小图，绿色=预测对，红色=预测错）
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # 本地有窗口想弹图：注释这行，并取消末尾 plt.show() 的注释
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN

DATA_ROOT = Path(__file__).parent / "data"
MODEL_FILE = Path(__file__).parent / "mnist_cnn.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # 和训练时保持一致，必须一样！
    ])
    test_data = datasets.MNIST(root=str(DATA_ROOT), train=False, download=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

    # 加载模型参数
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    # 1. 测试集准确率
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = model(images.to(device)).argmax(dim=1)
            correct += (pred == labels.to(device)).sum().item()
            total += labels.size(0)
    print(f"测试集准确率: {correct / total * 100:.2f}%")

    # 2. 可视化：随机取 6 张图展示预测结果
    images, labels = next(iter(DataLoader(test_data, batch_size=10, shuffle=True)))
    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1).cpu()

    fig, axes = plt.subplots(5, 10,figsize=(12, 2.5))
    for i, ax in enumerate(axes):
        img = images[i].squeeze(0)                    # 去掉 1 通道维 -> (28, 28) 灰度图
        ax.imshow(img, cmap="gray")
        ok = preds[i].item() == labels[i].item()
        ax.set_title(f"pred {preds[i].item()} / true {labels[i].item()}",
                     color="green" if ok else "red")
        ax.axis("off")
    plt.tight_layout()
    out_file = Path(__file__).parent / "prediction.png"
    plt.savefig(out_file, dpi=120)
    print("预测结果图已保存到:", out_file)


if __name__ == "__main__":
    main()
