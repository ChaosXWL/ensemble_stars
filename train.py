import os

from torch import nn, optim
import torch
from es_dataset import *
from es_net import *
from torch.utils.data import DataLoader


if __name__ == '__main__':
    result_dir = "./output"
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    weights=os.path.join(result_dir, "epoch_59.pth")
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('loading successfully')
    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()
    label_to_name = {
        "0": "神崎飒马",
        "1": "守泽千秋",
        "2": "朔间凛月",
        "3": "朔间零",
        "4": "天城燐音",
        "5": "天城一彩",
        "6": "天满光",
        "7": "天祥院英智",
        "8": "仙石忍",
        "9": "衣更真绪",
        "10": "乙狩阿多尼斯",
        "11": "樱河琥珀",
        "12": "影片美伽",
        "13": "游木真",
        "14": "羽风薰",
        "15": "斋宫宗",
        "16": "真白友也",
        "17": "朱樱司",
        "18": "椎名丹希",
        "19": "紫之创",
        "20": "HiMERU",
        "21": "巴日和",
        "22": "白鸟蓝良",
        "23": "冰鹰北斗",
        "24": "春川宙",
        "25": "大神晃牙",
        "26": "风早巽",
        "27": "伏见弓弦",
        "28": "高峯翠",
        "29": "鬼龙红郎",
        "30": "姬宫桃李",
        "31": "葵日向",
        "32": "葵裕太",
        "33": "濑名泉",
        "34": "月永雷欧",
        "35": "礼濑真宵",
        "36": "莲巳敬人",
        "37": "涟纯",
        "38": "乱凪砂",
        "39": "明星昴流",
        "40": "鸣上岚",
        "41": "南云铁虎",
        "42": "逆先夏目",
        "43": "七种茨",
        "44": "青叶纺",
        "45": "仁兔成鸣",
        "46": "日日树涉",
        "47": "三毛缟斑",
        "48": "深海奏汰"
    }
    es_dict = {v: k for k, v in label_to_name.items()}
    dataset = MyDataset('./es_data', es_dict)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    max_epoch = 300
    for epoch in range(60, max_epoch + 1):
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
