# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections

def get_IRENE_config():
    """Returns the IRENE configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.2
    config.transformer.dropout_rate = 0.3
    config.classifier = 'token'
    config.representation_size = None
    config.cc_len = 40
    config.lab_len = 92

    config.modality = ml_collections.ConfigDict()
    config.modality.use_image = True
    config.modality.use_text = False
    config.modality.mode = 'image'  # 'image', 'text', 'multimodal'
    
    # ========== 训练配置 ==========
    config.training = ml_collections.ConfigDict()
    config.training.epochs = 50
    config.training.batch_size = 16
    config.training.learning_rate = 3e-5
    config.training.weight_decay = 0.01
    config.training.save_frequency = 10  # 每10个epoch保存一次checkpoint
    config.training.seed = 42
    config.training.num_workers = 8  # DataLoader的worker数量
    
    # ========== 冻结策略配置 ==========
    config.freeze = ml_collections.ConfigDict()
    config.freeze.strategy = 'none'  # 'none', 'freeze_backbone', 'freeze_head'
    
    # ========== 路径配置 ==========
    config.paths = ml_collections.ConfigDict()
    config.paths.data_dir = './processed_data'
    config.paths.output_dir = './runs'
    config.paths.pretrained_dir = './runs/pretrained'
    config.paths.pretrained_path = None  # 具体的预训练权重文件路径
    
    # ========== 设备配置 ==========
    config.device = ml_collections.ConfigDict()
    config.device.use_cuda = True
    config.device.use_multi_gpu = True  # 是否使用多GPU
    config.device.gpu_ids = None
    
    return config

FREEZE_STRATEGIES = {
    'none': {
        'description': '全部参数可训练',
        'freeze_modules': []
    },
    'freeze_backbone': {
        'description': '冻结backbone，只训练分类头',
        'freeze_modules': ['transformer']
    },
    'freeze_head': {
        'description': '冻结分类头，只训练backbone', 
        'freeze_modules': ['head']
    }
}


def update_config_from_args(config, args):
    if hasattr(args, 'mode') and args.mode:
        config.modality.mode = args.mode
        if args.mode == 'image':
            config.modality.use_image = True
            config.modality.use_text = False
        elif args.mode == 'text':
            config.modality.use_image = False
            config.modality.use_text = True
        elif args.mode == 'multimodal':
            config.modality.use_image = True
            config.modality.use_text = True
    
    if hasattr(args, 'freeze') and args.freeze:
        config.freeze.strategy = args.freeze
    
    if hasattr(args, 'epochs') and args.epochs:
        config.training.epochs = args.epochs
        
    if hasattr(args, 'batch_size') and args.batch_size:
        config.training.batch_size = args.batch_size
        
    if hasattr(args, 'lr') and args.lr:
        config.training.learning_rate = args.lr
        
    if hasattr(args, 'data_dir') and args.data_dir:
        config.paths.data_dir = args.data_dir
        
    if hasattr(args, 'output_dir') and args.output_dir:
        config.paths.output_dir = args.output_dir
        
    if hasattr(args, 'pretrained_path') and args.pretrained_path:
        config.paths.pretrained_path = args.pretrained_path
        
    if hasattr(args, 'seed') and args.seed:
        config.training.seed = args.seed