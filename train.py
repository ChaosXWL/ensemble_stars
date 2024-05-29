import os

from torch import nn, optim
import torch
from es_dataset import *
from es_net import *
from torch.utils.data import DataLoader
from config import *


if __name__ == '__main__':
    result_dir = "./output"
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    weights = os.path.join(result_dir, "epoch_20.pth")
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('loading successfully')
    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()
    dataset = MyDataset('./es_data', es_dict)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    max_epoch = 300
    for epoch in range(1, max_epoch + 1):
        for i, (image, label, _) in enumerate(data_loader):
            image, label = image.to(device), label.to(device)
            out = net(image)
            train_loss = loss_fun(out, label)
            print(f'{epoch}-{i}-train_loss:{train_loss.item()}')
            opt.zero_grad()
            train_loss.backward()
            opt.step()
        if epoch % 1 == 0:
            torch.save(net.state_dict(), os.path.join(result_dir, f'epoch_{epoch}.pth'))
            print(f'save {epoch} epoch successfully')
