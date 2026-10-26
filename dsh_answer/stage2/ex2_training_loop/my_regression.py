import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

def main():
    true_a,true_b,true_c = 2,3,1
    x=torch.arange(-2,2,0.01)
    y=true_a*x**2+true_b*x+true_c+torch.randn(400)*0.2
    y_acc=true_a*x**2+true_b*x+true_c
    print(x[:10])
    print(y[:10])
    print(y_acc[:10])

    dataset = TensorDataset(x,y)
    loader = DataLoader(dataset,batch_size=100,shuffle=True)

    print(f'sample:{len(dataset)}batch_size:{loader.batch_size}batch:{len(loader)}')

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a=nn.Parameter(torch.randn(1))
            self.b=nn.Parameter(torch.randn(1))
            self.c=nn.Parameter(torch.randn(1))

        def forward(self,x):
            return (self.a*x**2)+(self.b*x)+(self.c)

    model=Net()
    print(f'model={model}')

    loss_fn = nn.MSELoss()
    print(f'loss_fn={loss_fn}')

    optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
    print(f'optimizer={optimizer}')

    epochs = 200
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred,yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        epoch_loss = train_loss / len(loader)

        if epoch % 10 == 9:
            print(f'epoch:{epoch+1},loss:{epoch_loss}')
            print(f'pred:{pred}')
    print(model.a.item(),model.b.item(),model.c.item())

if __name__ == "__main__":
        main()