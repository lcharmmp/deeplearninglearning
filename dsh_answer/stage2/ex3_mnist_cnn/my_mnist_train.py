import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

from model import SimpleCNN

DATA_ROOT = './data/'

def main():
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 0.005
    DOWNLOAD=False

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    train_data=datasets.MNIST(root=DATA_ROOT, train=True, transform=transform, download=DOWNLOAD)
    test_data=datasets.MNIST(root=DATA_ROOT, train=False, transform=transform, download=DOWNLOAD)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        seen=0
        for images, labels in train_loader:
            pred=model(images)
            loss = loss_fn(pred,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*images.size(0)
            correct += (pred.argmax(1) == labels).sum().item()
            seen += images.size(0)
        train_acc = correct / seen *100

        model.eval()
        correct = 0
        seen =0
        with torch.no_grad():
            for images, labels in test_loader:
                pred=model(images)
                correct += (pred.argmax(1) == labels).sum().item()
                seen += images.size(0)
        val_acc = correct / seen *100
        print(f"epoch {epoch}/{EPOCHS} | loss {total_loss / seen:.4f} | "
              f"train_acc {train_acc:.2f}% | val_acc {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("./model", exist_ok=True)
            torch.save(model.state_dict(),f'./model/model_LR{LR}.pth')
            print(f"best model saved with acc {best_acc:.2f}%")
if __name__ == '__main__':
    main()

