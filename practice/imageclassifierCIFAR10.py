from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader


# ---------- 配置 ----------
DATA_ROOT = "./data"
MODEL_PATH = "./cifar_net.pt"
BATCH_SIZE = 4
EPOCHS = 3
LR = 0.001
MOMENTUM = 0.9
NUM_WORKERS = 2          # Windows/macOS 或 notebook 先设 0
DOWNLOAD = True          # 数据已就位后可改 False

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ---------- 模型 ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)      # (N,6,28,28)
        self.pool = nn.MaxPool2d(2, 2)       # 尺寸减半
        self.conv2 = nn.Conv2d(6, 16, 5)     # (N,16,10,10)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (N,6,14,14)
        x = self.pool(F.relu(self.conv2(x)))   # (N,16,5,5)
        x = torch.flatten(x, 1)                # (N,400)
        x = F.relu(self.fc1(x))                # (N,120)
        x = F.relu(self.fc2(x))                # (N,84)
        x = self.fc3(x)                        # (N,10) logits
        return x


# ---------- 数据 ----------
def build_loaders():
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=DOWNLOAD, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=DOWNLOAD, transform=transform
    )

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS
    )
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False,   # 测试集不打乱
        num_workers=NUM_WORKERS
    )
    return trainloader, testloader


# ---------- 训练 ----------
def train(net, trainloader, criterion, optimizer, device):
    net.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print(f"[epoch {epoch+1}, batch {i+1}] "
                      f"loss: {running_loss/1000:.3f}")
                running_loss = 0.0

        avg_loss = running_loss / max(1, len(trainloader) % 1000)
        print(f"[epoch {epoch+1}] done, tail avg loss: {avg_loss:.3f}")


# ---------- 评估 ----------
def evaluate(net, testloader, device):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test accuracy: {acc:.2f}%")
    return acc


# ---------- 主流程 ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = build_loaders()
    print(f'trainloaderlen: {len(trainloader)}')

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)

    train(net, trainloader, criterion, optimizer, device)
    print("Finished Training")

    torch.save(net.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    evaluate(net, testloader, device)


if __name__ == "__main__":
    main()