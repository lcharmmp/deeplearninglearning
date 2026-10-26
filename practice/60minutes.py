#
import numpy as np
import torch
# from torchvision.models import resnet18, ResNet18_Weights
# model = resnet18(weights=ResNet18_Weights.DEFAULT)
# data = torch.rand(1, 3, 64, 64)
# labels = torch.rand(1, 1000)
# prediction = model(data) # forward pass
# loss = (prediction - labels).sum()
# loss.backward() # backward pass
# print(loss.item())
# optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# optim.step() #gradient descent
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Fully connected layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


net = Net()
#print(net)
# params = list(net.parameters())
# for param in params:
#     print(param.size())
input = torch.randn(1, 1, 32, 32)#NCHW
out = net(input)
print(out)
net.zero_grad()
target=torch.tensor([0.5]*10)
target=target.view(1,-1)#equal to just (1,10)
criterion = nn.MSELoss()
loss = criterion(out, target)

learning_rate = 0.005
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
optimizer.zero_grad()
for t in range(500):
    optimizer.zero_grad()
    out = net(input)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
out = net(input)
print(out)

