import torch
from torchvision import models
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet50()
        self.model.fc = nn.Linear(2048, 50)
        # print(self.model)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    net = Net()
    x = torch.randn(1, 3, 500, 500)
    print(net(x).shape)
