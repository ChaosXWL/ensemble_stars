import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import cv2
from torchvision import transforms


tf = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, img_path, es_dict):
        img_files = []
        for root, dirs, files in os.walk(img_path):
            load_num = 0
            for file in files:
                if ".jpg" not in file:
                    continue
                img_files.append(osp.join(root, file))
                load_num += 1
                if load_num > 5:
                    break
        self.dataset = img_files
        self.es_dict = es_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        # print(img_path)
        label = osp.basename(osp.dirname(img_path))
        img = cv2.imread(img_path)
        img = cv2.resize(img, (500, 500))

        return tf(img), torch.tensor(int(self.es_dict[label])), osp.splitext(osp.basename(img_path))[0]


if __name__ == '__main__':
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
    # print(es_dict)
    data = MyDataset('es_data', es_dict)
    for i in data:
        print(i[0].shape)
        print(i[1])
        break
