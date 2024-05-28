import cv2
import os.path as osp
from es_dataset import *
from es_net import *
from torch.utils.data import DataLoader
import shutil
from config import *

if __name__ == '__main__':
    result_dir = "./output"
    predict_dir = "./predict_res"
    if osp.exists(predict_dir):
        shutil.rmtree(predict_dir)
    os.makedirs(predict_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    weights = os.path.join(result_dir, "epoch_210.pth")
    net.load_state_dict(torch.load(weights))
    net.eval()
    dataset = MyDataset('./es_data', es_dict)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i, (images, labels, image_names) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.tolist()
        batch_num = images.shape[0]
        out = net(images)
        cls_res = torch.softmax(out, 1)
        cls_scores, cls_indices = torch.max(cls_res, 1)
        cls_scores = cls_scores.tolist()
        cls_indices = cls_indices.tolist()
        print(batch_num, image_names, labels, cls_indices, cls_scores)
        for j in range(batch_num):
            img = images[j, :]
            img = img.cpu().permute(1, 2, 0).numpy()
            img = img * 255
            predict_name = label_to_name[str(cls_indices[j])]
            label_name = label_to_name[str(labels[j])]
            img_name = image_names[j]
            if predict_name != label_name:
                cv2.imwrite(osp.join(predict_dir, f"{i}_{j}_{img_name}_{predict_name}_{label_name}.jpg"), img)
