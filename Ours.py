import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import random
import math

# 工具包
from Utils.data_setting import TestDataset, TrainDataset, img_transform, mask_transform
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from Models.Validate import validate_model
from Utils.evaluate_setting import SegModel_eval
from Utils.plot_setting import label_name_to_value, colormap, weights
from Models.Unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定种子
def set_deterministic(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class Args:
    deterministic = True
    seed = 42    
args = Args()
if args.deterministic:
    set_deterministic(args.seed)

def save_metrics(evaluate, perEval, epoch, running_loss, train_loader):
    evaluate["Loss"].append(running_loss / len(train_loader))
    evaluate["iou"].append(perEval["iou"])
    evaluate["dice"].append(perEval["dice"])
    evaluate["haus"].append(perEval["haus"])
    evaluate["score"].append(perEval["score"])

    return evaluate

def create_model(in_channels=3, num_classes=2, dropout_rate=0.0):
    # 创建模型
    model = UNet(in_channels, num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    model = nn.DataParallel(model)
    model.to(device)
    return model

### -------------------------------------------------------------全局区--------------------------------------------------------------- ###

# 一般训练参数设置
num_epochs = 120  # 训练的总轮数
patience = 120  # 早停的耐心次数
batch_size = 12  # 批次大小

# 超参数设置
lr_setting = 0.003
CosineAnnealingWarmRestarts_T_0 = 8
CosineAnnealingWarmRestarts_T_mult = 2

# 创建模型
sup_model = create_model(in_channels=3, num_classes=2, dropout_rate=0.05)    # 纯监督模型
ema_model = create_model(in_channels=3, num_classes=2, dropout_rate=0.0)    # EMA模型
con_model = create_model(in_channels=3, num_classes=2, dropout_rate=0.05)    # 一致性模型
ema_model.load_state_dict(con_model.state_dict())  # 初始化EMA模型参数

# 监督损失计算器、优化器、学习率调度器
sup_criterion = nn.CrossEntropyLoss(weight=weights)
sup_optimizer = optim.Adam(sup_model.parameters(), lr=lr_setting)
sup_scheduler = CosineAnnealingWarmRestarts(sup_optimizer, T_0=CosineAnnealingWarmRestarts_T_0, T_mult=CosineAnnealingWarmRestarts_T_mult)

con_optimizer = optim.Adam(con_model.parameters(), lr=lr_setting)
con_scheduler = CosineAnnealingWarmRestarts(con_optimizer, T_0=CosineAnnealingWarmRestarts_T_0, T_mult=CosineAnnealingWarmRestarts_T_mult)

# 输出路径
ExpName = "Ours" 
sup_path = rf'Sup'
ema_path = rf'Ema'
con_path = rf'Con'

sup_pred_path = rf"./predicted_masks/{ExpName}/{sup_path}"            
ema_pred_path = rf"./predicted_masks/{ExpName}/{ema_path}" 
con_pred_path = rf"./predicted_masks/{ExpName}/{con_path}"              

sup_model_path = rf'./predicted_masks/{ExpName}/{sup_path}/best_model'  
ema_model_path = rf'./predicted_masks/{ExpName}/{ema_path}/best_model'    
con_model_path = rf'./predicted_masks/{ExpName}/{con_path}/best_model'    

os.makedirs(sup_pred_path, exist_ok=True)
os.makedirs(ema_pred_path, exist_ok=True)
os.makedirs(con_pred_path, exist_ok=True)
os.makedirs(sup_model_path, exist_ok=True)
os.makedirs(ema_model_path, exist_ok=True)
os.makedirs(con_model_path, exist_ok=True)

# 数据集加载
train_img_dir = "./dataset/train/label"
train_mask_dir = "./dataset/train/gt"
pe_img_dir = "./dataset/train/unl"
test_img_dir = "./dataset/test/label"
test_mask_dir = "./dataset/test/gt"

unlabel_ratio = 2

train_dataset = TrainDataset(train_img_dir, train_mask_dir, pe_img_dir, img_transform, mask_transform, unlabel_ratio=unlabel_ratio)
test_dataset = TestDataset(test_img_dir, test_mask_dir, img_transform, mask_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def softmax_mse_loss(input_logits, target_logits):
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.mse_loss(input_softmax, target_softmax, reduction='mean')

def softmax_KL_loss(input_logits, target_logits, threshold=0.5):
    # 损失计算的Softmax，target用log‐softmax
    input_logp = F.log_softmax(input_logits, dim=1)    # [B, C, H, W]
    target_p   = F.softmax(target_logits, dim=1)       # [B, C, H, W]

    # 计算置信度和掩码，置信度取最大值，掩码大于阈值的像素点
    conf, _ = target_p.max(dim=1, keepdim=True)        # [B, 1, H, W]
    mask = (conf > threshold).float()                  # [B, 1, H, W]

    # 算KL散度: KL(target || input)
    kl_map = F.kl_div(input_logp, target_p, reduction='none')  # [B, C, H, W]
    kl_per_pixel = kl_map.sum(dim=1, keepdim=True)             # [B, 1, H, W]

    # 置信度掩码，只保留有效像素点的KL散度
    masked_kl = kl_per_pixel * mask
    valid_count = mask.sum()

    if valid_count.item() > 0:
        return masked_kl.sum() / valid_count
    else:
        # 如果没有有效像素点，返回均值
        return kl_per_pixel.mean()

def mutual_hard_label_consistency(logits1, logits2, threshold= 0.5):
    # 解耦合
    logits1 = logits1.detach()
    logits2 = logits2.detach()

    # 硬标签 from logits2
    with torch.no_grad(): 
        prob2 = F.softmax(logits2, dim=1)
        conf2, pseudo2 = torch.max(prob2, dim=1)
        mask2 = (conf2 > threshold).float()
        
    # 模型1损失
    loss1 = (F.cross_entropy(logits1, pseudo2, reduction='none') * mask2).mean()
    
    # 硬标签 from logits1
    with torch.no_grad():
        # 计算模型1的伪标签和置信度
        prob1 = F.softmax(logits1, dim=1)
        conf1, pseudo1 = torch.max(prob1, dim=1)
        mask1 = (conf1 > threshold).float()
        
    # 模型2损失
    loss2 = (F.cross_entropy(logits2, pseudo1, reduction='none') * mask1).mean()
    
    return loss1, loss2

def get_current_lamba(iter, max_iter, max_lamda):
    if iter < max_iter:
        sigma = max_iter / 3
        exponent = -((max_iter - iter)**2) / (2 * sigma**2)
        return max_lamda * (1 - math.exp(exponent)), iter + 1
    return max_lamda, iter + 1

def merge_logits_by_confidence(con_logits, ema_logits):

    con_conf = F.softmax(con_logits, dim=1).max(dim=1, keepdim=True)[0]  # shape: [B, 1, H, W]
    ema_conf = F.softmax(ema_logits, dim=1).max(dim=1, keepdim=True)[0]  # shape: [B, 1, H, W]
    
    con_higher = con_conf >= ema_conf  # shape: [B, 1, H, W]
    con_higher = con_higher.expand_as(con_logits)  # shape: [B, C, H, W]

    merged_logits = torch.where(con_higher, con_logits, ema_logits)

    return merged_logits

def DTK(
    sup_model, 
    ema_model,
    con_model, 
    train_loader, 
    test_loader, 
    sup_criterion, 
    sup_optimizer,
    con_optimizer, 
    sup_scheduler,
    con_scheduler, 
    num_epochs, 
    patience,
    max_lamda=0.5,
    warmup_ratio=0.1,
    threshold = 0.5,
    hard_threshold = 0.5,  
    ema = 0.99,    
    ):
    # 初始化训练过程中的相关变量
    sup_eval = {"Loss": [], "iou": [], "dice": [], "haus": [], "score": []}
    ema_eval = {"Loss": [], "iou": [], "dice": [], "haus": [], "score": []}
    con_eval = {"Loss": [], "iou": [], "dice": [], "haus": [], "score": []}

    best_score = 0
    patience_counter = 0

    total_iter = 0
    total_train_iters = num_epochs * len(train_loader)
    warmup_iters = int(total_train_iters * warmup_ratio)

    # 训练循环
    for epoch in range(num_epochs):
        # 早停判断
        if patience_counter >= patience:
            break
        
        running_loss = 0.0
        sup_total_loss_items = 0.0
        con_total_loss_items = 0.0

        # 训练模式
        sup_model.train()
        ema_model.train()
        con_model.train()

        for labeled_images, labeled_masks, unlabeled_images in train_loader:
            # 加载数据
            images, masks = labeled_images, labeled_masks
            images, masks = images.to(device), masks.to(device)

            unlabeled_images = torch.cat(unlabeled_images, dim=0).to(device)
            unlabeled_images = unlabeled_images.to(device)

            # sup传递监督信号
            for sup_p, con_p in zip(sup_model.parameters(), con_model.parameters()):
                con_p.data = ema * con_p.data + (1 - ema) * sup_p.data

            # 前传
            sup_l_logist = sup_model(images)

            with torch.no_grad():  # ema模型前向时关闭梯度
                ema_u_logist = ema_model(unlabeled_images)
                
            sup_u_logist = sup_model(unlabeled_images)
            con_u_logist = con_model(unlabeled_images)
            
            # 计算监督损失
            sup_l_loss = sup_criterion(sup_l_logist, masks)

            # 软标签“知识迁移”
            con_u_loss_soft_KL = softmax_KL_loss(con_u_logist, sup_u_logist.detach(), threshold=threshold)
            # sup_u_loss_soft_KL = softmax_KL_loss(sup_u_logist, con_u_logist.detach(), threshold=threshold)
            # sup_u_loss_soft_KL = softmax_KL_loss(
            #     sup_u_logist, 
            #     merge_logits_by_confidence(
            #         con_u_logist.detach(), 
            #         ema_u_logist.detach()
            #     ), 
            #     threshold=threshold
            # )
            sup_u_loss_soft_KL = softmax_KL_loss(
                sup_u_logist, 
                merge_logits_by_confidence(
                    con_u_logist.detach(), 
                    con_u_logist.detach()
                ), 
                threshold=threshold
            )
            

            # 软标签一致性
            con_u_loss_hard_CE, _ = mutual_hard_label_consistency(con_u_logist, ema_u_logist.detach(), threshold=hard_threshold)

            current_lamda, total_iter= get_current_lamba(
                iter=total_iter,
                max_iter=warmup_iters,
                max_lamda=max_lamda
            )

            sup_total_loss = sup_l_loss + current_lamda * sup_u_loss_soft_KL
            con_total_loss = current_lamda * con_u_loss_hard_CE

            total_loss = sup_total_loss + con_total_loss

            # 优化器梯度清零，反传、优化器更新
            sup_optimizer.zero_grad()
            con_optimizer.zero_grad()

            total_loss.backward()

            sup_optimizer.step()  
            con_optimizer.step()

            # 监督损失记录
            running_loss += total_loss.item()
            sup_total_loss_items += sup_l_loss.item()

            for con_p, ema_p in zip(con_model.parameters(), ema_model.parameters()):
                ema_p.data = ema * ema_p.data + (1 - ema) * con_p.data

        # 每个epoch更新一次学习率优化器
        sup_scheduler.step()
        con_scheduler.step()

        # 启动模型验证
        sup_model.eval()
        ema_model.eval()
        con_model.eval()
        
        # 获取模型预测和真实标签
        sup_pred, gts = validate_model(sup_model, test_loader, sup_pred_path, device)
        ema_pred, _ = validate_model(ema_model, test_loader, ema_pred_path, device)
        con_pred, _ = validate_model(con_model, test_loader, con_pred_path, device)

        # 计算评价指标
        sup_matrix = SegModel_eval(sup_pred, gts)
        ema_matrix = SegModel_eval(ema_pred, gts)
        con_matrix = SegModel_eval(con_pred, gts)

        # 打印测试集结果
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {sup_total_loss_items / len(train_loader):.4f}, IoU: {sup_matrix["iou"]:.4f}, Dice: {sup_matrix["dice"]:.4f}, Hausdorff: {sup_matrix["haus"]:.4f}, Score: {sup_matrix["score"]:.4f}')

        # 保存指标
        sup_eval = save_metrics(sup_eval, sup_matrix, epoch, sup_total_loss_items, train_loader)
        ema_eval = save_metrics(ema_eval, ema_matrix, epoch, 0.0, train_loader)
        con_eval = save_metrics(con_eval, con_matrix, epoch, 0.0, train_loader)

        sup_socre = sup_matrix["score"]
        ema_socre = ema_matrix["score"]
        con_socre = con_matrix["score"]

        if max(sup_socre, ema_socre, con_socre) == sup_socre:
             better_matrix = sup_matrix
             model_name = '1'
        elif max(sup_socre, ema_socre, con_socre) == ema_socre:
             better_matrix = ema_matrix
             model_name = '2'
        else:
             better_matrix = con_matrix
             model_name = '3'

        # 保存最佳模型
        if better_matrix["score"] > best_score:
            patience_counter = 0
            best_score = better_matrix["score"]
            if model_name == '1':
                torch.save(sup_model.state_dict(), os.path.join(sup_model_path, 'sup_model_best.pth'))
                torch.save(ema_model.state_dict(), os.path.join(ema_model_path, 'ema_model.pth'))
                torch.save(con_model.state_dict(), os.path.join(con_model_path, 'con_model.pth'))
            elif model_name == '2':
                torch.save(sup_model.state_dict(), os.path.join(sup_model_path, 'sup_model.pth'))
                torch.save(ema_model.state_dict(), os.path.join(ema_model_path, 'ema_model_best.pth'))
                torch.save(con_model.state_dict(), os.path.join(con_model_path, 'con_model.pth'))
            else:
                torch.save(sup_model.state_dict(), os.path.join(sup_model_path, 'sup_model.pth'))
                torch.save(ema_model.state_dict(), os.path.join(ema_model_path, 'ema_model.pth'))
                torch.save(con_model.state_dict(), os.path.join(con_model_path, 'con_model_best.pth'))
            print(f"best_model is {model_name}, update at {epoch + 1}epoch: best_score={best_score}")
        else:
            patience_counter += 1

    # 返回所有过程的指标
    print(f"Train Have Been done! The Best Score: {best_score}")
    return sup_eval, ema_eval, con_eval

if __name__ == "__main__":
    # 训练模型
    sup_eval, ema_eval, con_eval = DTK(
        sup_model, 
        ema_model,
        con_model, 
        train_loader, 
        test_loader, 
        sup_criterion, 
        sup_optimizer,
        con_optimizer, 
        sup_scheduler,
        con_scheduler, 
        num_epochs, 
        patience,
        max_lamda=0.5,
        warmup_ratio=0.1,
        threshold = 0.2,
        hard_threshold = 0.2,
        ema = 0.99,         
    )