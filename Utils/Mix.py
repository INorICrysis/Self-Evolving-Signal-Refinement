import os
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_GaussBoundary_mask(patch_size, Boundary_width=4, noise_weight=0.5, mean=1, std=1, device="cpu"):
    patch_x, patch_y = patch_size
    
    # 创建基础CutMix mask
    cutmix_mask = np.ones((patch_x, patch_y))
    cutmix_mask[Boundary_width:-Boundary_width, Boundary_width:-Boundary_width] = 0
    
    # 创建高斯噪声 mask
    random_gaussian_noise = np.random.normal(loc=0.5, scale=0.5, size=(patch_x, patch_y))  # 随机生成高斯噪声
    random_gaussian_noise = np.clip(random_gaussian_noise, 0, 1)  # 限制在0到1之间
    
    # 在边界混合
    mask = (1 - noise_weight) * cutmix_mask + noise_weight * random_gaussian_noise
    mask[Boundary_width:-Boundary_width, Boundary_width:-Boundary_width] = 1
    return torch.tensor(mask, dtype=torch.float32, device=device)

def Mix(images, masks=None, mask_ratio=0.2, mixType="GaussBoundaryMix", sigma=100, Boundary_width=4, noise_weight=0.5, mean=1, std=1):
    batch_size, channels, img_x, img_y = images.size()  # 获取batch_size, img的C, H, W
    patch_x, patch_y = int(img_x * mask_ratio), int(img_y * mask_ratio)  # 初始化patch的长宽

    # 生成掩码
    if mixType == "SmoothMix":
        gaussian_mask = cv2.getGaussianKernel(patch_x, sigma) @ cv2.getGaussianKernel(patch_y, sigma).T
        inverse_gaussian_mask = 1 - gaussian_mask / np.max(gaussian_mask)
        patch_mask = torch.tensor(inverse_gaussian_mask, dtype=torch.float32, device=images.device)  # 直接创建在 GPU 上

    elif mixType == "CutMix":
        patch_mask = torch.ones((patch_x, patch_y), dtype=torch.float32, device=images.device)

    elif mixType == "GaussBoundaryMix":
        patch_mask = generate_GaussBoundary_mask(patch_size=(patch_x, patch_y), Boundary_width=Boundary_width, noise_weight=noise_weight, mean=mean, std=std, device=images.device)

    # 随机分组
    indices = list(range(batch_size))
    random.shuffle(indices)  # 随机打乱索引
    groups = [(indices[i], indices[i + 1]) for i in range(0, batch_size, 2)]
    if len(indices) % 2 != 0:  # batch_size 为奇数时
        groups.append((indices[-1], indices[-1]))

    # Mix
    for idx1, idx2 in groups:
        x, y = random.randint(0, img_x - patch_x), random.randint(0, img_y - patch_y)   # 确定patch的左上角起始位置，交换位置相同，防止损失信息

        # img混合
        patch_img1 = images[idx2, :, x:x + patch_x, y:y + patch_y] * patch_mask
        patch_img2 = images[idx1, :, x:x + patch_x, y:y + patch_y] * patch_mask

        images[idx1, :, x:x + patch_x, y:y + patch_y] = (
            images[idx1, :, x:x + patch_x, y:y + patch_y] * (1 - patch_mask) + patch_img1
        )
        images[idx2, :, x:x + patch_x, y:y + patch_y] = (
            images[idx2, :, x:x + patch_x, y:y + patch_y] * (1 - patch_mask) + patch_img2
        )
        
        # mask混合
        if masks is not None:
            patch_mask_label1 = masks[idx2, x:x + patch_x, y:y + patch_y] * patch_mask
            patch_mask_label2 = masks[idx1, x:x + patch_x, y:y + patch_y] * patch_mask

            masks[idx1, x:x + patch_x, y:y + patch_y] = (
                masks[idx1, x:x + patch_x, y:y + patch_y] * (1 - patch_mask) + patch_mask_label1
            )
            masks[idx2, x:x + patch_x, y:y + patch_y] = (
                masks[idx2, x:x + patch_x, y:y + patch_y] * (1 - patch_mask) + patch_mask_label2
            )

    return (images, masks) if masks is not None else images