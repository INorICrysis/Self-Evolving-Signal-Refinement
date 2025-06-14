# 使用：from Utils.plot_setting import label_order, label_name_to_value, colormap, weights, create_label_map, apply_colormap, unlabel_predict

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义标签和颜色映射
label_order = [
    '_background_', 
    'tooth'
]
label_name_to_value = {
    '_background_': 0, 
    'tooth': 1
}
colormap = [
    [0, 0, 0],          # Background：黑的
    [255, 255, 255]     # Tooth：白的          
]

# 权重比例设置
weights = torch.tensor([1, 1], dtype=torch.float32).to(device)

# 创建标签映射
def create_label_map(mask, colormap):
    mask = np.array(mask)
    label_map = np.zeros(mask.shape[:2], dtype=np.uint8)
    for idx, color in enumerate(colormap):
        mask_ = np.all(mask == color, axis=-1)
        label_map[mask_] = idx
    return label_map

# 将预测掩码转换为彩色图像
def apply_colormap(mask, colormap):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in enumerate(colormap):
        color_mask[mask == label] = color
    return color_mask

def unlabel_predict(model, data_loader, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, (inputs, img_paths) in enumerate(data_loader):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()

        for j, (pred, img_path) in enumerate(zip(preds, img_paths)):
            # 创建彩色调色板图像
            pred_colored = apply_colormap(pred, colormap)
            pred_img = Image.fromarray(pred_colored)
            img_name = os.path.basename(img_path)
            pred_img.save(os.path.join(output_dir, img_name))