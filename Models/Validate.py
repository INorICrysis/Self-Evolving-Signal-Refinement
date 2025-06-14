# 使用：from Models.Validate import validate_model

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from Utils.plot_setting import label_order, label_name_to_value, colormap, weights, create_label_map, apply_colormap

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型验证
def validate_model(model, data_loader, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    predicted_masks = []
    true_masks = []

    for i, (inputs, masks) in enumerate(data_loader):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()

        for j, (pred, true_mask) in enumerate(zip(preds, masks.numpy())):
            predicted_masks.append(pred)
            true_masks.append(true_mask)

            # 获取图像文件名
            img_idx = i * data_loader.batch_size + j
            filename = data_loader.dataset.get_filename(img_idx)

            # 创建彩色调色板图像
            pred_colored = apply_colormap(pred, colormap)
            pred_img = Image.fromarray(pred_colored)
            pred_img.save(os.path.join(output_dir, filename))

    return np.array(predicted_masks), np.array(true_masks)

