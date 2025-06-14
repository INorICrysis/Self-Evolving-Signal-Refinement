# 使用：from Utils.data_setting import TestDataset, TrainDataset, img_transform, mask_transform
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from Utils.plot_setting import label_order, label_name_to_value, colormap, weights, create_label_map, apply_colormap
import numpy as np
from torch.utils.data import Sampler
import random
from torch.utils.data import Sampler

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据集类
class TestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # 将掩码图像转换为标签图像
        mask = create_label_map(mask, colormap)
        mask = torch.from_numpy(mask).long()

        return image, mask

    def get_filename(self, idx):
        return self.imgs[idx]

class TrainDataset(Dataset):
    def __init__(self, img_dir, mask_dir, unlabel_dir, img_transform=None, mask_transform=None, unlabel_ratio=2):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.unlabel_dir = unlabel_dir
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.unlabels = sorted(os.listdir(unlabel_dir))  # Keep filenames sorted for consistency

        self.unlabel_ratio = unlabel_ratio  # 保存 unlabel_ratio 为实例变量

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # 随机选择未标注样本
        unlabel_paths = random.sample(self.unlabels, min(len(self.unlabels), self.unlabel_ratio))  # 使用 self.unlabel_ratio
        unlabel = []
        for unlabel_path in unlabel_paths:
            full_path = os.path.join(self.unlabel_dir, unlabel_path)
            temp = Image.open(full_path).convert("RGB")
            if self.img_transform:
                temp = self.img_transform(temp)
            unlabel.append(temp)

        # 将掩码图像转换为标签图像
        mask = create_label_map(mask, colormap)
        mask = torch.from_numpy(mask).long()

        return image, mask, unlabel

    def get_filename(self, idx):
        return self.imgs[idx]

class DualDataset(Dataset):
    def __init__(self, img_dir, mask_dir, unlabel_dir1, unlabel_dir2, img_transform=None, mask_transform=None, unlabel_ratio=1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.unlabel_dir1 = unlabel_dir1
        self.unlabel_dir2 = unlabel_dir2
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.unlabels1 = sorted(os.listdir(unlabel_dir1)) 
        self.unlabels2 = sorted(os.listdir(unlabel_dir2))  

        self.unlabel_ratio = unlabel_ratio  # 保存 unlabel_ratio 为实例变量

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        # 随机选择未标注样本
        unlabel_paths1 = random.sample(self.unlabels1, min(len(self.unlabels1), self.unlabel_ratio))  # 使用 self.unlabel_ratio
        unlabel_paths2 = random.sample(self.unlabels2, min(len(self.unlabels2), self.unlabel_ratio))  # 使用 self.unlabel_ratio
        unlabel1 = []
        unlabel2 = []
        for unlabel_path1 in unlabel_paths1:
            full_path = os.path.join(self.unlabel_dir1, unlabel_path1)
            temp1 = Image.open(full_path).convert("RGB")
            if self.img_transform:
                temp1 = self.img_transform(temp1)
            unlabel1.append(temp1)
        for unlabel_path2 in unlabel_paths2:
            full_path = os.path.join(self.unlabel_dir2, unlabel_path2)
            temp2 = Image.open(full_path).convert("RGB")
            if self.img_transform:
                temp2 = self.img_transform(temp2)
            unlabel2.append(temp2)

        # 将掩码图像转换为标签图像
        mask = create_label_map(mask, colormap)
        mask = torch.from_numpy(mask).long()

        return image, mask, unlabel1, unlabel2

    def get_filename(self, idx):
        return self.imgs[idx]
# 数据转换
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def mask_transform(mask):
    mask = mask.resize((256, 256), Image.NEAREST)
    return mask