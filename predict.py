import cv2
import os.path as osp
from es_dataset import *
from es_net import *
from torch.utils.data import DataLoader
import shutil


if __name__ == '__main__':
    result_dir = "./output"
    predict_dir = "./predict_res"
    if osp.exists(predict_dir):
        shutil.rmtree(predict_dir)
    os.makedirs(predict_dir, exist_ok=True)
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
        "48": "深海奏汰",
        "49": "其他"
    }
    es_dict = {v: k for k, v in label_to_name.items()}
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
