# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_iou(pred_masks, true_masks, per_image=False):
    iou_list = []
    for pred, true in zip(pred_masks, true_masks):
        intersection = np.logical_and(pred, true)
        union = np.logical_or(pred, true)
        iou = np.sum(intersection) / np.sum(union)
        iou_list.append(iou)
    if per_image:
        return iou_list
    return np.mean(iou_list)

# Dice系数
def calculate_dice(pred_masks, true_masks, per_image=False):
    dice_list = []
    for pred, true in zip(pred_masks, true_masks):
        intersection = np.logical_and(pred, true)
        dice = 2 * np.sum(intersection) / (np.sum(pred) + np.sum(true))
        dice_list.append(dice)
    if per_image:
        return dice_list  
    return np.mean(dice_list) 

# 2D Hausdorff 距离
def calculate_hausdorff(pred_masks, true_masks, per_image=False):
    hausdorff_list = []
    
    # 获取图像尺寸和对角线长度（假设所有图像尺寸相同）
    image_height, image_width = pred_masks[0].shape[0], pred_masks[0].shape[1]
    diagonal_length = np.sqrt(image_height**2 + image_width**2)
    
    for pred, true in zip(pred_masks, true_masks):
        # 提取坐标点
        pred_points = np.column_stack(np.where(pred == 1))
        true_points = np.column_stack(np.where(true == 1))
        
        # 处理空点集情况
        if len(pred_points) == 0 and len(true_points) == 0:
            normalized_hausdorff = 0.0
        elif len(pred_points) == 0 or len(true_points) == 0:
            normalized_hausdorff = 1.0
        else:
            try:
                # 计算双向Hausdorff距离
                hd1 = directed_hausdorff(pred_points, true_points)[0]
                hd2 = directed_hausdorff(true_points, pred_points)[0]
                hausdorff_distance = max(hd1, hd2)
                normalized_hausdorff = hausdorff_distance / diagonal_length
            except:
                normalized_hausdorff = 1.0  # 异常时设为最大值
            
            # 强制确保结果在[0,1]范围内
            normalized_hausdorff = np.clip(normalized_hausdorff, 0.0, 1.0)
        
        hausdorff_list.append(normalized_hausdorff)
    
    return hausdorff_list if per_image else np.nanmean(hausdorff_list)


# def calculate_hausdorff(pred_masks, true_masks, per_image=False):
#     hausdorff_list = []
    
#     # 调试信息1：检查输入掩码的尺寸一致性
#     for i in range(1, len(pred_masks)):
#         if pred_masks[i].shape != pred_masks[0].shape:
#             raise ValueError(f"预测掩码尺寸不一致！第0张尺寸为{pred_masks[0].shape}，第{i}张为{pred_masks[i].shape}")
    
#     # 计算对角线长度（假设所有图像尺寸相同）
#     image_height, image_width = pred_masks[0].shape[0], pred_masks[0].shape[1]
#     diagonal_length = np.sqrt(image_height**2 + image_width**2)
#     print(f"[调试] 图像尺寸：{image_height}x{image_width}，对角线长度：{diagonal_length:.2f}")  # 调试信息2
    
#     for idx, (pred, true) in enumerate(zip(pred_masks, true_masks)):
#         # 调试信息3：检查掩像素值合法性
#         unique_pred = np.unique(pred)
#         unique_true = np.unique(true)
#         if not np.array_equal(unique_pred, [0, 1]) and not np.array_equal(unique_pred, [0]):
#             print(f"[警告] 第{idx}张预测掩码包含非法值：{unique_pred}")
#         if not np.array_equal(unique_true, [0, 1]) and not np.array_equal(unique_true, [0]):
#             print(f"[警告] 第{idx}张真实掩码包含非法值：{unique_true}")
        
#         # 提取坐标点
#         pred_points = np.column_stack(np.where(pred == 1))
#         true_points = np.column_stack(np.where(true == 1))
        
#         # 调试信息4：打印坐标点统计信息
#         print(f"\n[样本 {idx}] 预测前景点数：{len(pred_points)}，真实前景点数：{len(true_points)}")
        
#         # 处理空点集情况
#         if len(pred_points) == 0 and len(true_points) == 0:
#             normalized_hausdorff = 0.0
#             print(f"  [处理] 两者均为空，距离设为0")
#         elif len(pred_points) == 0 or len(true_points) == 0:
#             normalized_hausdorff = 1.0
#             print(f"  [处理] 单边为空，距离设为1.0")
#         else:
#             # 调试信息5：检查坐标范围
#             if (np.any(pred_points[:, 0] < 0) or 
#                 np.any(pred_points[:, 0] >= image_height) or
#                 np.any(pred_points[:, 1] < 0) or 
#                 np.any(pred_points[:, 1] >= image_width)):
#                 print(f"  [错误] 预测坐标越界！最小坐标：{pred_points.min(axis=0)}，最大坐标：{pred_points.max(axis=0)}")
            
#             if (np.any(true_points[:, 0] < 0) or 
#                 np.any(true_points[:, 0] >= image_height) or
#                 np.any(true_points[:, 1] < 0) or 
#                 np.any(true_points[:, 1] >= image_width)):
#                 print(f"  [错误] 真实坐标越界！最小坐标：{true_points.min(axis=0)}，最大坐标：{true_points.max(axis=0)}")
            
#             try:
#                 # 调试信息6：捕获Scipy计算异常
#                 hd1 = directed_hausdorff(pred_points, true_points)[0]
#                 hd2 = directed_hausdorff(true_points, pred_points)[0]
#                 hausdorff_distance = max(hd1, hd2)
#                 normalized_hausdorff = hausdorff_distance / diagonal_length
#                 print(f"  [计算] 原始距离：hd1={hd1:.2f}, hd2={hd2:.2f}, 最终距离={hausdorff_distance:.2f}, 归一化后={normalized_hausdorff:.4f}")
#             except Exception as e:
#                 print(f"  [异常] Hausdorff计算失败：{str(e)}")
#                 normalized_hausdorff = 1.0  # 异常时设为最大值
        
#         # 调试信息7：检查归一化结果合法性
#         if not (0 <= normalized_hausdorff <= 1.0):
#             print(f"  [错误] 非法归一化距离：{normalized_hausdorff}，已修正为1.0")
#             normalized_hausdorff = 1.0
        
#         hausdorff_list.append(normalized_hausdorff)
    
    # 调试信息8：统计最终结果
    print("\n[汇总] 所有样本Hausdorff距离统计：")
    print(f"  最小值：{np.min(hausdorff_list):.4f}")
    print(f"  最大值：{np.max(hausdorff_list):.4f}")
    print(f"  平均值：{np.mean(hausdorff_list):.4f}")
    print(f"  Inf/NaN数量：{sum(np.isinf(hausdorff_list) | np.isnan(hausdorff_list))}")
    
    if per_image:
        return hausdorff_list
    return np.nanmean(hausdorff_list)  # 忽略NaN求平均

def calculate_score(dice, IoU, haus):
    return 0.4 * dice + 0.3 * IoU + 0.3 * (1 - haus)

def SegModel_eval(pred_masks, true_masks, isDice=True, isIoU=True, isHaus=True, isScore=True, num_classes=2, per_image=False):
    if num_classes < 2: 
        print("Class_num Error!")
    SegEval = {}
  
    if isDice:
        dice = calculate_dice(pred_masks, true_masks, per_image=per_image)
        SegEval["dice"] = dice

    if isIoU:
        IoU = calculate_iou(pred_masks, true_masks, per_image=per_image)
        SegEval["iou"] = IoU

    if isHaus:
        haus = calculate_hausdorff(pred_masks, true_masks, per_image=per_image)
        SegEval["haus"] = haus

    if isScore:
        score = calculate_score(dice, IoU, haus)
        SegEval["score"] = score

    return SegEval
        