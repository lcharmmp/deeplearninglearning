"""
CIFAR-10 独立测试脚本。

复用 imageclassifierCIFAR10.py 中的 Net、CLASSES、配置常量，
加载已保存的 state_dict，在测试集上评估准确率，并给出单张预测示例。

用法:
    python imageclassifierCIFAR10TEST.py
    python imageclassifierCIFAR10TEST.py --model ./cifar_net.pt --data ./data
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import imageclassifierCIFAR10 as ic


def build_test_transform():
    """与训练集一致：只做 ToImage / ToDtype / Normalize，不加随机增强。"""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def build_testloader(data_root, batch_size, num_workers, download=False):
    transform = build_test_transform()
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=download, transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    return testset, testloader


def load_model(model_path, device):
    """实例化训练脚本里的 Net，并加载 state_dict。"""
    net = ic.Net().to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


@torch.no_grad()
def evaluate(net, testloader, device):
    """整体准确率 + 每类准确率。"""
    correct = 0
    total = 0
    class_correct = torch.zeros(10)
    class_total = torch.zeros(10)

    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for label, pred in zip(labels, predicted):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

    acc = 100.0 * correct / total if total else 0.0
    print(f"\nOverall accuracy: {acc:.2f}%  ({correct}/{total})")

    print("\nPer-class accuracy:")
    for i, name in enumerate(ic.CLASSES):
        n = int(class_total[i].item())
        c = int(class_correct[i].item())
        class_acc = 100.0 * c / n if n else 0.0
        print(f"  {name:>5s}: {class_acc:6.2f}%  ({c}/{n})")

    return acc


@torch.no_grad()
def predict_single(net, image_tensor, device):
    """
    单张预测。
    image_tensor: 形状 (1, 3, 32, 32)，已做与测试集一致的 transform。
    返回 (预测类别索引, 类别名, 各类概率)
    """
    net.eval()
    image_tensor = image_tensor.to(device)
    logits = net(image_tensor)                    # (1, 10)
    probs = F.softmax(logits, dim=1)[0]           # (10,)
    pred_idx = int(probs.argmax().item())
    return pred_idx, ic.CLASSES[pred_idx], probs.cpu()


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 test script")
    parser.add_argument("--model", type=str, default=ic.MODEL_PATH,
                        help="path to saved model state_dict")
    parser.add_argument("--data", type=str, default=ic.DATA_ROOT,
                        help="CIFAR-10 data root")
    parser.add_argument("--batch-size", type=int, default=ic.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=ic.NUM_WORKERS,
                        help="0 is safest on Windows/macOS/notebook")
    parser.add_argument("--download", action="store_true",
                        help="allow downloading if data missing")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path.resolve()}\n"
            f"请先运行 imageclassifierCIFAR10.py 生成 {model_path}。"
        )

    net = load_model(str(model_path), device)

    testset, testloader = build_testloader(
        data_root=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    print(f"Test samples: {len(testset)}")

    evaluate(net, testloader, device)

    # 单张预测示例
    print("\nSingle-sample demo:")
    img, label = testset[0]                       # img: (3,32,32)
    pred_idx, pred_name, probs = predict_single(
        net, img.unsqueeze(0), device
    )
    print(f"  true label : {ic.CLASSES[label]}")
    print(f"  prediction : {pred_name} (idx={pred_idx})")
    print(f"  confidence : {probs[pred_idx].item():.4f}")


if __name__ == "__main__":
    main()