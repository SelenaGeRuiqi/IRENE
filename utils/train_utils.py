"""
训练核心模块
负责训练循环、验证、保存checkpoint等核心训练功能
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with tqdm(dataloader, desc="Training", leave=False) as pbar:
        for batch_idx, (imgs, labels, ccs, demos, labs) in enumerate(pbar):
            labels = labels.to(device, non_blocking=True)
            
            if config.modality.mode == 'image':
                # 纯图像模式
                imgs = imgs.to(device, non_blocking=True)
                loss = model(x=imgs, cc=None, lab=None, sex=None, age=None, labels=labels)
                
            elif config.modality.mode == 'text':
                # 纯文本模式
                ccs = ccs.view(-1, config.cc_len, ccs.shape[3]).to(device, non_blocking=True).float()
                demos = demos.view(-1, 1, demos.shape[1]).to(device, non_blocking=True).float()
                labs = labs.view(-1, labs.shape[1], 1).to(device, non_blocking=True).float()
                sex = demos[:, :, 1].view(-1, 1, 1).to(device, non_blocking=True).float()
                age = demos[:, :, 0].view(-1, 1, 1).to(device, non_blocking=True).float()
                loss = model(x=None, cc=ccs, lab=labs, sex=sex, age=age, labels=labels)
                
            elif config.modality.mode == 'multimodal':
                # 多模态模式
                imgs = imgs.to(device, non_blocking=True)
                ccs = ccs.view(-1, config.cc_len, ccs.shape[3]).to(device, non_blocking=True).float()
                demos = demos.view(-1, 1, demos.shape[1]).to(device, non_blocking=True).float()
                labs = labs.view(-1, labs.shape[1], 1).to(device, non_blocking=True).float()
                sex = demos[:, :, 1].view(-1, 1, 1).to(device, non_blocking=True).float()
                age = demos[:, :, 0].view(-1, 1, 1).to(device, non_blocking=True).float()
                loss = model(x=imgs, cc=ccs, lab=labs, sex=sex, age=age, labels=labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_epoch(model, dataloader, criterion, device, config, disease_list):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        with tqdm(dataloader, desc="Evaluating", leave=False) as pbar:
            for batch_idx, (imgs, labels, ccs, demos, labs) in enumerate(pbar):
                labels = labels.to(device, non_blocking=True)
                
                if config.modality.mode == 'image':
                    # 纯图像模式
                    imgs = imgs.to(device, non_blocking=True)
                    logits, _, _ = model(x=imgs, cc=None, lab=None, sex=None, age=None)
                    
                elif config.modality.mode == 'text':
                    # 纯文本模式
                    ccs = ccs.view(-1, config.cc_len, ccs.shape[3]).to(device, non_blocking=True).float()
                    demos = demos.view(-1, 1, demos.shape[1]).to(device, non_blocking=True).float()
                    labs = labs.view(-1, labs.shape[1], 1).to(device, non_blocking=True).float()
                    sex = demos[:, :, 1].view(-1, 1, 1).to(device, non_blocking=True).float()
                    age = demos[:, :, 0].view(-1, 1, 1).to(device, non_blocking=True).float()
                    logits, _, _ = model(x=None, cc=ccs, lab=labs, sex=sex, age=age)
                    
                elif config.modality.mode == 'multimodal':
                    # 多模态模式
                    imgs = imgs.to(device, non_blocking=True)
                    ccs = ccs.view(-1, config.cc_len, ccs.shape[3]).to(device, non_blocking=True).float()
                    demos = demos.view(-1, 1, demos.shape[1]).to(device, non_blocking=True).float()
                    labs = labs.view(-1, labs.shape[1], 1).to(device, non_blocking=True).float()
                    sex = demos[:, :, 1].view(-1, 1, 1).to(device, non_blocking=True).float()
                    age = demos[:, :, 0].view(-1, 1, 1).to(device, non_blocking=True).float()
                    logits, _, _ = model(x=imgs, cc=ccs, lab=labs, sex=sex, age=age)
                
                loss = criterion(logits, labels.float())
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    individual_aurocs = []
    num_classes = all_labels.shape[1]
    
    for i in range(num_classes):
        try:
            auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            individual_aurocs.append(auroc)
        except ValueError:
            # 如果某个类别的标签全为0或全为1，跳过该类别
            individual_aurocs.append(0.5)
    
    avg_auroc = np.mean(individual_aurocs)
    
    return avg_loss, avg_auroc, individual_aurocs


def save_checkpoint(model, optimizer, epoch, metrics, config, save_path):
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_metrics = checkpoint.get('metrics', {})
    
    print(f"Resumed from epoch {start_epoch}")
    print(f"Best metrics: {best_metrics}")
    
    return start_epoch, best_metrics


def save_training_log(log_data, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def print_epoch_results(epoch, epochs, train_loss, val_loss, val_auroc, individual_aurocs, disease_list):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val AUROC: {val_auroc:.4f}")
    
    print("Individual AUROCs:")
    for i, (disease, auroc) in enumerate(zip(disease_list, individual_aurocs)):
        print(f"  {disease}: {auroc:.4f}")


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")