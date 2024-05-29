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


class TestDataset(Dataset):
    def __init__(self, img_path, es_dict):
        img_files = []
        for root, dirs, files in os.walk(img_path):
            for file in files:
                if ".jpg" not in file:
                    continue
                img_files.append(osp.join(root, file))
        self.dataset = img_files
        self.es_dict = es_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path = self.dataset[index]
        # print(img_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (500, 500))

        return tf(img), osp.splitext(osp.basename(img_path))[0]


if __name__ == '__main__':
    # print(es_dict)
    data = TestDataset('es_data', es_dict)
    for i in data:
        print(i[0].shape)
        print(i[1])
        break
