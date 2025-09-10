import os
import sys
import argparse
import json
import time
from datetime import datetime

import torch
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.configs import get_IRENE_config, update_config_from_args
from utils.data_utils import load_disease_info, create_dataloaders
from utils.model_utils import setup_model_for_training, setup_device_and_model, create_optimizer, create_criterion
from utils.train_utils import (
    train_epoch, evaluate_epoch, save_checkpoint, load_checkpoint, 
    save_training_log, print_epoch_results, set_random_seed
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="IRENE Training")
    
    parser.add_argument('--mode', choices=['image', 'text', 'multimodal'], 
                        default='image', help='Training mode')
    
    parser.add_argument('--freeze', choices=['none', 'freeze_backbone', 'freeze_head'],
                        default='none', help='Freeze strategy')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument('--data_dir', default='./processed_data', help='Data directory')
    parser.add_argument('--output_dir', default='./runs', help='Output directory')
    parser.add_argument('--pretrained_path', default=None, help='Pretrained weights path')
    
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    parser.add_argument('--save_frequency', type=int, default=10, help='Save checkpoint every N epochs')
    
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--gpu_ids', nargs='+', type=int, help='Specific GPU IDs to use')
    
    parser.add_argument('--exp_name', default=None, help='Experiment name')
    
    return parser.parse_args()


def create_experiment_dir(config, args):
    if args.exp_name:
        exp_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{config.modality.mode}_{config.freeze.strategy}_{timestamp}"
    
    exp_dir = os.path.join(config.paths.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    
    return exp_dir, exp_name


def save_config(config, exp_dir):
    config_path = os.path.join(exp_dir, 'config.json')
    
    config_dict = {}
    for key, value in config.items():
        if hasattr(value, 'to_dict'):
            config_dict[key] = value.to_dict()
        else:
            config_dict[key] = value
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Config saved to: {config_path}")


def main():
    args = parse_arguments()
    
    config = get_IRENE_config()
    update_config_from_args(config, args)
    
    set_random_seed(config.training.seed)
    
    exp_dir, exp_name = create_experiment_dir(config, args)
    
    print("="*60)
    print(f"IRENE Training Started")
    print(f"Experiment: {exp_name}")
    print(f"Mode: {config.modality.mode}")
    print(f"Freeze Strategy: {config.freeze.strategy}")
    print(f"Output Directory: {exp_dir}")
    print("="*60)
    
    save_config(config, exp_dir)

    print("\nLoading disease information...")
    disease_info = load_disease_info(config.paths.data_dir)
    num_classes = disease_info['num_classes']
    disease_list = disease_info['disease_list']
    
    print("\nCreating data loaders...")
    train_loader, test_loader = create_dataloaders(config)
    
    print("\nSetting up model...")
    model = setup_model_for_training(config, num_classes)
    model, device = setup_device_and_model(model, config)
    
    print("\nSetting up optimizer and criterion...")
    optimizer = create_optimizer(model, config)
    criterion = create_criterion()
    
    start_epoch = 0
    best_metrics = {'auroc': 0.0, 'loss': float('inf')}
    training_log = []
    
    if args.resume:
        print(f"\nResuming training from checkpoint...")
        start_epoch, best_metrics = load_checkpoint(args.resume, model, optimizer)
        
        log_path = os.path.join(os.path.dirname(args.resume), 'training_log.json')
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                training_log = json.load(f)
    
    print(f"\nStarting training from epoch {start_epoch+1}")
    print(f"Best AUROC so far: {best_metrics.get('auroc', 0.0):.4f}")
    
    for epoch in range(start_epoch, config.training.epochs):
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
        
        val_loss, val_auroc, individual_aurocs = evaluate_epoch(
            model, test_loader, criterion, device, config, disease_list
        )
        
        epoch_time = time.time() - epoch_start_time
        
        print_epoch_results(epoch, config.training.epochs, train_loss, val_loss, 
                          val_auroc, individual_aurocs, disease_list)
        print(f"Epoch time: {epoch_time:.2f}s")
        
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auroc': val_auroc,
            'individual_aurocs': individual_aurocs,
            'epoch_time': epoch_time
        }
        training_log.append(epoch_log)
        
        if val_auroc > best_metrics.get('auroc', 0.0):
            best_metrics = {'auroc': val_auroc, 'loss': val_loss, 'epoch': epoch + 1}
            best_model_path = os.path.join(exp_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_metrics, config, best_model_path)
            print(f"New best model saved! AUROC: {val_auroc:.4f}")
        
        if (epoch + 1) % config.training.save_frequency == 0:
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, 
                          {'auroc': val_auroc, 'loss': val_loss}, config, checkpoint_path)

        log_path = os.path.join(exp_dir, 'logs', 'training_log.json')
        save_training_log(training_log, log_path)
        
        print("-" * 60)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best AUROC: {best_metrics.get('auroc', 0.0):.4f} (Epoch {best_metrics.get('epoch', 0)})")
    print(f"Best model saved at: {os.path.join(exp_dir, 'best_model.pth')}")
    print(f"Training log saved at: {log_path}")
    print(f"Experiment directory: {exp_dir}")
    
    print("\n" + "="*60)
    print("Final evaluation with best model...")
    
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model)
        
        final_loss, final_auroc, final_individual_aurocs = evaluate_epoch(
            model, test_loader, criterion, device, config, disease_list
        )
        
        print(f"Final Test Results:")
        print(f"Test Loss: {final_loss:.4f}")
        print(f"Test AUROC: {final_auroc:.4f}")
        print("Individual AUROCs:")
        for disease, auroc in zip(disease_list, final_individual_aurocs):
            print(f"  {disease}: {auroc:.4f}")
        
        final_results = {
            'final_test_loss': final_loss,
            'final_test_auroc': final_auroc,
            'final_individual_aurocs': dict(zip(disease_list, final_individual_aurocs)),
            'best_epoch': best_metrics.get('epoch', 0),
            'total_epochs': config.training.epochs
        }
        
        results_path = os.path.join(exp_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"Final results saved to: {results_path}")
    
    print("="*60)


if __name__ == "__main__":
    main()