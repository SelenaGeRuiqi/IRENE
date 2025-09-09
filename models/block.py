# coding=utf-8
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

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

class Block(nn.Module):
    """
    IRENE的Transformer Block（变换器块）
    
    这是IRENE模型的基础构建单元，每个Block包含：
    1. 多头自注意力机制
    2. 前馈神经网络（MLP）
    3. 层归一化
    4. 残差连接
    
    支持两种工作模式：
    
    模式1 - 标准模式（mm=False）：
    - 处理单一的融合后序列
    - 类似标准Transformer Block的行为
    - 用于模态融合后的层
    
    模式2 - 多模态模式（mm=True）：
    - 同时处理图像和文本两个独立序列
    - 每个模态有独立的归一化层和MLP
    - 保持模态特异性，用于早期层
    """
    
    def __init__(self, config, vis, mm=False):
        """
        初始化Transformer Block
        
        Args:
            config: 模型配置，包含hidden_size等参数
            vis: 是否启用可视化，返回注意力权重
            mm: 是否为多模态模式
                - True: 创建双路径结构，分别处理图像和文本
                - False: 创建单路径结构，处理融合序列
        """
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        
        # 图像模态的归一化层和前馈网络（所有Block都有）
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)  # 注意力前的归一化
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)        # MLP前的归一化
        
        # 如果是多模态模式，为文本模态创建独立的组件
        if mm:
            # 文本模态专用的归一化层
            self.att_norm_text = LayerNorm(config.hidden_size, eps=1e-6)   # 文本注意力前归一化
            self.ffn_norm_text = LayerNorm(config.hidden_size, eps=1e-6)   # 文本MLP前归一化
            # 文本模态专用的前馈网络
            self.ffn_text = Mlp(config)

        # 图像模态的前馈网络（所有Block都有）
        self.ffn = Mlp(config)
        
        # 注意力机制（根据mm参数决定是单模态还是多模态注意力）
        self.attn = Attention(config, vis, mm)

    def forward(self, x, text=None):
        """
        前向传播函数
        
        Args:
            x: 图像特征 [batch_size, img_seq_len, hidden_dim]
            text: 文本特征 [batch_size, text_seq_len, hidden_dim]（可选）
        
        Returns:
            标准模式: (x, weights) - 处理后的特征和注意力权重
            多模态模式: (x, text, weights) - 处理后的图像特征、文本特征和注意力权重
        """
        if text is None:
            # 标准模式：处理单一序列（通常是融合后的多模态序列）
            # 这种模式用于模态融合后的层（第3层及以后）
            
            # 第一个子层：多头自注意力 + 残差连接
            h = x  # 保存残差连接的输入
            x = self.attention_norm(x)      # Pre-LN：先归一化
            x, weights = self.attn(x)       # 多头自注意力
            x = x + h                       # 残差连接
            
            # 第二个子层：前馈网络 + 残差连接
            h = x  # 保存残差连接的输入
            x = self.ffn_norm(x)           # Pre-LN：先归一化
            x = self.ffn(x)                # 前馈网络（MLP）
            x = x + h                      # 残差连接
            
            return x, weights
            
        else:
            # 多模态模式：分别处理图像和文本序列
            # 这种模式用于早期层（第0-1层），保持模态独立性
            
            # === 第一个子层：多头自注意力 + 残差连接 ===
            # 保存残差连接的输入
            h = x           # 图像特征的残差输入
            h_text = text   # 文本特征的残差输入
            
            # 分别对两个模态进行归一化
            x = self.attention_norm(x)        # 图像特征归一化
            text = self.att_norm_text(text)   # 文本特征归一化
            
            # 多模态注意力：可能包含跨模态交互
            x, text, weights = self.attn(x, text)
            
            # 残差连接
            x = x + h           # 图像特征的残差连接
            text = text + h_text # 文本特征的残差连接

            # === 第二个子层：前馈网络 + 残差连接 ===
            # 保存残差连接的输入
            h = x           # 图像特征的残差输入
            h_text = text   # 文本特征的残差输入
            
            # 分别对两个模态进行归一化
            x = self.ffn_norm(x)            # 图像特征归一化
            text = self.ffn_norm_text(text) # 文本特征归一化
            
            # 分别通过各自的前馈网络
            x = self.ffn(x)           # 图像前馈网络
            text = self.ffn_text(text) # 文本前馈网络
            
            # 残差连接
            x = x + h           # 图像特征的残差连接
            text = text + h_text # 文本特征的残差连接
            
            return x, text, weights

    def load_from(self, weights, n_block):
        """
        从预训练权重加载Block参数
        
        Args:
            weights: 预训练权重字典
            n_block: 当前Block的编号（用于构建权重键名）
        
        这个函数处理复杂的权重加载逻辑，包括：
        1. 注意力机制的Q、K、V、O矩阵权重
        2. MLP的两层全连接权重
        3. 各层归一化的权重和偏置
        
        注意：权重需要从numpy格式转换为PyTorch格式，并进行适当的转置
        """
        # 构建当前Block的权重路径前缀
        ROOT = f"Transformer/encoderblock_{n_block}"
        
        with torch.no_grad():
            # === 加载注意力机制权重 ===
            # 加载Q、K、V、O矩阵的权重和偏置
            # 注意：需要进行view和转置操作以匹配PyTorch的权重格式
            
            # pjoin: 跨平台的路径拼接函数. 示例：pjoin("Transformer", "encoderblock_0", "query", "kernel") → "Transformer/encoderblock_0/query/kernel"
            # view: pytorch tensor的类似numpy reshape
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            # 复制权重到模型参数
            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            # === 加载MLP权重 ===
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            # 复制MLP权重到模型参数
            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            # === 加载层归一化权重 ===
            # 注意力层归一化参数
            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            # MLP层归一化参数
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


