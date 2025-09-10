"""
模型管理模块
负责创建模型、加载权重、应用冻结策略等功能
"""

import torch
import torch.nn as nn
import os
from models.modeling_irene import IRENE, CONFIGS
from models.configs import FREEZE_STRATEGIES


def create_model(mode, num_classes):
    config = CONFIGS["IRENE"]
    
    if mode == 'image':
        config.modality.use_image = True
        config.modality.use_text = False
    elif mode == 'text':
        config.modality.use_image = False
        config.modality.use_text = True
    elif mode == 'multimodal':
        config.modality.use_image = True
        config.modality.use_text = True
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    config.modality.mode = mode

    model = IRENE(config, 224, zero_head=True, num_classes=num_classes)
    
    print(f"Created IRENE model:")
    print(f"  Mode: {mode}")
    print(f"  Use image: {config.modality.use_image}")
    print(f"  Use text: {config.modality.use_text}")
    print(f"  Number of classes: {num_classes}")
    
    return model


def load_pretrained_weights(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        print(f"Warning: Pretrained weights not found at {pretrained_path}")
        return
    
    print(f"Loading pretrained weights from {pretrained_path}")
    
    try:
        pretrained_weights = torch.load(pretrained_path, map_location='cpu')
        model_weights = model.state_dict()
        
        loaded_keys = []
        skipped_keys = []
        
        for key, value in pretrained_weights.items():
            if key in model_weights:
                if model_weights[key].shape == value.shape:
                    model_weights[key] = value
                    loaded_keys.append(key)
                else:
                    print(f"Shape mismatch for {key}: "
                          f"model {model_weights[key].shape} vs pretrained {value.shape}")
                    skipped_keys.append(key)
            else:
                skipped_keys.append(key)
        
        model.load_state_dict(model_weights)
        
        print(f"Successfully loaded {len(loaded_keys)} layers")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} layers due to mismatch or absence")
            
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        print("Continuing with random initialization")


def apply_freeze_strategy(model, strategy):
    if strategy not in FREEZE_STRATEGIES:
        raise ValueError(f"Unknown freeze strategy: {strategy}. "
                        f"Available strategies: {list(FREEZE_STRATEGIES.keys())}")
    
    for param in model.parameters():
        param.requires_grad = True
    
    freeze_info = FREEZE_STRATEGIES[strategy]
    print(f"Applying freeze strategy: {strategy}")
    print(f"Description: {freeze_info['description']}")
    
    if strategy == 'freeze_backbone':
        for param in model.transformer.parameters():
            param.requires_grad = False
        print("Froze transformer parameters")
            
    elif strategy == 'freeze_head':
        for param in model.head.parameters():
            param.requires_grad = False
        print("Froze head parameters")
    
    # 'none'策略不需要额外操作


def setup_model_for_training(config, num_classes):
    model = create_model(config.modality.mode, num_classes)
    if config.paths.pretrained_path:
        load_pretrained_weights(model, config.paths.pretrained_path)
    apply_freeze_strategy(model, config.freeze.strategy)
    return model


def setup_device_and_model(model, config):
    if config.device.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device(s): {torch.cuda.device_count()} GPU(s)")
        
        model = model.to(device)
        
        if config.device.use_multi_gpu and torch.cuda.device_count() > 1:
            if config.device.gpu_ids:
                model = nn.DataParallel(model, device_ids=config.device.gpu_ids)
                print(f"Using specified GPUs: {config.device.gpu_ids}")
            else:
                model = nn.DataParallel(model)
                print(f"Using all available GPUs: {list(range(torch.cuda.device_count()))}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return model, device


def create_optimizer(model, config):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    print(f"Created AdamW optimizer:")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Weight decay: {config.training.weight_decay}")
    
    return optimizer


def create_criterion():
    # 使用BCEWithLogitsLoss用于多标签分类
    criterion = nn.BCEWithLogitsLoss()
    print("Created BCEWithLogitsLoss for multi-label classification")
    return criterion