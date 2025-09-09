# coding=utf-8
"""
IRENE模型的Transformer编码器实现

这个文件实现了IRENE模型的核心编码器组件，负责处理多模态数据的深层特征融合。
编码器采用分层融合策略，在不同层级逐步整合图像和临床文本信息。

关键特性：
1. 分层多模态融合：前两层分别处理图像和文本，第三层开始融合
2. 渐进式特征整合：从独立处理到完全融合的渐进过程
3. 注意力权重可视化：支持返回注意力权重用于模型解释

设计理念：
- 早期层：保持模态独立性，避免过早融合导致信息丢失
- 中期层：开始跨模态交互，学习模态间的关联
- 后期层：完全融合，学习统一的多模态表示
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs
from models.attention import Attention
from models.embed import Embeddings 
from models.mlp import Mlp
from models.block import Block

class Encoder(nn.Module):
    """
    IRENE的多模态Transformer编码器， 分层多模态融合策略。
    
    1. 前2层：多模态分离处理（mm=True的Block）
       - 图像和文本在各自的子空间中进行自注意力计算
       - 保持模态特异性，避免过早融合造成的信息损失
    
    2. 第3层：模态融合点
       - 将图像特征和文本特征在序列维度上拼接
       - 从此层开始，所有特征统一处理
    
    3. 后续层：统一处理
       - 对融合后的多模态序列进行标准的Transformer处理
       - 学习跨模态的高级语义表示
    """
    
    def __init__(self, config, vis):
        # super()用于调用父类(nn.Module)的初始化方法
        # 确保nn.Module中的参数和缓冲区被正确初始化
        super(Encoder, self).__init__()
        self.vis = vis
        
        # 存储所有Transformer层的列表
        self.layer = nn.ModuleList()
        
        # 最终的层归一化，在输出前对特征进行标准化
        # eps=1e-6 是为了数值稳定性，避免除零错误
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        
        # 构建分层的Transformer架构
        for i in range(config.transformer["num_layers"]):
            if i < 2:
                # 前两层：多模态分离处理
                # mm=True 表示这是多模态Block，会分别处理图像和文本
                layer = Block(config, vis, mm=True)
            else:
                # 后续层：标准Transformer处理
                # 处理已融合的多模态序列
                layer = Block(config, vis)
            
            # 使用copy.deepcopy确保每层都是独立的实例
            # 避免参数共享导致的训练问题
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, text=None):
        """
        编码器前向传播
        
        Args:
            hidden_states: 图像的嵌入表示 [batch_size, num_patches+1, hidden_dim]
                          包含CLS token和图像patch embeddings
            text: 文本的嵌入表示 [batch_size, text_seq_len, hidden_dim]
                  包含临床描述、实验室结果、性别、年龄等信息
        
        Returns:
            encoded: 编码后的多模态特征 [batch_size, total_seq_len, hidden_dim]
            attn_weights: 注意力权重列表（如果启用可视化）
        
        处理流程详解：
        1. 层0-1：分离处理阶段
           - 图像和文本各自进行自注意力计算
           - 保持模态边界，学习模态内的依赖关系
        
        2. 层2：融合阶段
           - 将图像和文本特征在序列维度拼接
           - 形成统一的多模态序列表示
        
        3. 层3+：统一处理阶段
           - 对融合序列进行标准Transformer处理
           - 学习跨模态的高级语义关联
        """
        attn_weights = []  # 存储每层的注意力权重
        
        # 逐层处理
        for (i, layer_block) in enumerate(self.layer):
            if i == 2:
                # 第3层（索引2）：执行模态融合
                # 将图像特征和文本特征在序列维度上拼接
                # hidden_states: [batch, img_seq_len, hidden_dim] 
                # text: [batch, text_seq_len, hidden_dim]
                # 拼接后: [batch, img_seq_len + text_seq_len, hidden_dim]
                hidden_states = torch.cat((hidden_states, text), 1)  
                
                # 从此层开始，只有一个输出（融合后的特征）
                hidden_states, weights = layer_block(hidden_states)
                
            elif i < 2:
                # 前两层：多模态分离处理
                # 返回三个值：处理后的图像特征、处理后的文本特征、注意力权重
                hidden_states, text, weights = layer_block(hidden_states, text)
                
            else:
                # 第3层之后：标准Transformer处理
                # 输入和输出都是融合后的特征序列
                hidden_states, weights = layer_block(hidden_states)

            # 如果启用可视化，收集注意力权重
            if self.vis:
                attn_weights.append(weights)
        
        # 最终层归一化
        # 对所有特征进行标准化，确保输出的数值稳定性
        encoded = self.encoder_norm(hidden_states)
        
        return encoded, attn_weights


