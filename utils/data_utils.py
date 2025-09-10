"""
数据管理模块
负责加载疾病信息、创建数据加载器、设置数据变换等功能
"""

import os
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def load_disease_info(data_dir):
    disease_info_path = os.path.join(data_dir, 'disease_info.pkl')
    
    if not os.path.exists(disease_info_path):
        raise FileNotFoundError(f"Disease info file not found: {disease_info_path}")
    
    with open(disease_info_path, 'rb') as f:
        disease_info = pickle.load(f)
    
    print(f"Loaded disease info: {disease_info['num_classes']} classes")
    for i, disease in enumerate(disease_info['disease_list']):
        print(f"  {i}: {disease}")
    
    return disease_info


def setup_data_transforms(mode):
    if mode in ['image', 'multimodal']:
        # 图像模态需要数据变换【TBD: 根据需要调整】
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        }
    else:
        # 纯文本模态不需要图像变换
        data_transforms = {
            'train': None,
            'test': None
        }
    
    return data_transforms


def create_dataloaders(config):
    from irene import FlexibleData
    
    data_transforms = setup_data_transforms(config.modality.mode)
    
    img_dir = os.path.join(config.paths.data_dir, 'images')
    
    train_dataset = FlexibleData(
        set_type='train',
        img_dir=img_dir,
        mode=config.modality.mode,
        transform=data_transforms['train']
    )
    
    test_dataset = FlexibleData(
        set_type='test',
        img_dir=img_dir,
        mode=config.modality.mode,
        transform=data_transforms['test']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Mode: {config.modality.mode}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Image dir: {img_dir}")
    
    return train_loader, test_loader