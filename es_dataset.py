import torch
from torch.utils.data import Dataset
import os
import os.path as osp
import cv2
from torchvision import transforms
from config import *


tf = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, img_path, es_dict):
        img_files = []
        for root, dirs, files in os.walk(img_path):
            # load_num = 0
            for file in files:
                if ".jpg" not in file:
                    continue
                img_files.append(osp.join(root, file))
                # load_num += 1
                # if load_num > 5:
                #     break
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
    # print(es_dict)
    data = MyDataset('es_data', es_dict)
    for i in data:
        print(i[0].shape)
        print(i[1])
        break
