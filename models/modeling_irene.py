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
from torch.nn.modules.utils import _pair  # 用于处理卷积核大小参数的工具函数
from scipy import ndimage  # 用于位置编码的插值操作

import models.configs as configs
from models.attention import Attention
from models.embed import Embeddings 
from models.mlp import Mlp
from models.block import Block
from models.encoder import Encoder
import pdb

logger = logging.getLogger(__name__)

def np2th(weights, conv=False):
    """
    将numpy数组转换为PyTorch tensor
    
    Args:
        weights: numpy权重数组
        conv: 是否为卷积层权重，如果是则需要转置维度
        
    Returns:
        转换后的PyTorch tensor
        
    说明：
    - 预训练权重通常以numpy格式存储
    - 卷积层权重需要从HWIO格式转换为PyTorch的OIHW格式
    """    
    if conv:
        # 将权重从HWIO (Height, Width, Input, Output) 转换为 OIHW (Output, Input, Height, Width)
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        """
        vis: 是否启用可视化（返回注意力权重）
        """
        super(Transformer, self).__init__()
        self.config = config
        self.use_image = config.modality.use_image
        self.use_text = config.modality.use_text

        # 多模态嵌入层：将图像patches和临床数据转换为统一的嵌入表示
        self.embeddings = Embeddings(config, img_size=img_size)
        # Transformer编码器：处理序列化的多模态数据
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, cc=None, lab=None, sex=None, age=None):
        """
        前向传播函数
        
        Args:
            input_ids: 输入图像tensor
            cc: 临床主诉描述 (chief complaint)
            lab: 实验室检查结果 (laboratory tests)
            sex: 性别信息
            age: 年龄信息
            
        Returns:
            encoded: 编码后的多模态特征表示
            attn_weights: 注意力权重（如果启用可视化）
            
        处理流程：
        1. 通过嵌入层将各模态数据转换为统一格式
        2. 将临床数据（cc, lab, sex, age）拼接为文本序列
        3. 通过编码器进行多模态特征融合
        """
        # Step 1: 多模态嵌入
        # embedding_output, cc, lab, sex, age = self.embeddings(input_ids, cc, lab, sex, age)
        image_embeddings, text_embeddings = self.embeddings(input_ids, cc, lab, sex, age)

        if self.use_image and self.use_text:
            # 多模态
            primary_embeddings = image_embeddings
            auxiliary_embeddings = text_embeddings
        elif self.use_image:
            # 纯图像
            primary_embeddings = image_embeddings
            auxiliary_embeddings = None
        elif self.use_text:
            # 纯文本
            primary_embeddings = text_embeddings
            auxiliary_embeddings = None
        else:
            raise ValueError("At least one modality (image or text) must be enabled")
        
        # Step 3: Transformer编码
        encoded, attn_weights = self.encoder(primary_embeddings, auxiliary_embeddings)
        return encoded, attn_weights


class IRENE(nn.Module):
    """
    IRENE完整模型类
    
    这是IRENE模型的顶层接口，整合了：
    1. Transformer多模态编码器
    2. 分类头用于疾病预测
    3. 损失函数计算
    4. 预训练权重加载
    
    IRENE采用多标签分类方式，可以同时预测多种疾病的存在概率。
    """
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(IRENE, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier  # 分类器类型（通常为'token'）

        self.config = config
        self.use_image = config.modality.use_image
        self.use_text = config.modality.use_text

        # 核心Transformer模块
        self.transformer = Transformer(config, img_size, vis)
        
        # 分类头：将Transformer输出映射到疾病类别概率
        # config.hidden_size通常为768，对应Transformer的隐藏维度
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, cc=None, lab=None, sex=None, age=None, labels=None):
        """
        IRENE模型前向传播
        
        Args:
            x: 输入医学影像 [batch_size, 3, 224, 224]
            cc: 临床主诉描述 [batch_size, seq_len, hidden_dim]
            lab: 实验室检查结果 [batch_size, num_labs, 1]
            sex: 性别信息 [batch_size, 1, 1]
            age: 年龄信息 [batch_size, 1, 1]
            labels: 真实疾病标签（训练时提供）
            
        Returns:
            训练时: 返回损失值
            推理时: 返回(logits, 注意力权重, 平均特征)
            
        模型流程：
        1. 通过Transformer编码多模态输入
        2. 全局平均池化得到序列级表示
        3. 通过分类头预测疾病概率
        4. 计算损失（训练时）或返回预测结果（推理时）
        """
        if self.use_image and x is None:
            raise ValueError("Image input required when use_image=True")
        if self.use_text and (cc is None or lab is None or sex is None or age is None):
            raise ValueError("Text inputs (cc, lab, sex, age) required when use_text=True")

        # Step 1: Transformer多模态编码
        features, attn_weights = self.transformer(input_ids=x, cc=cc, lab=lab, sex=sex, age=age)
        
        # Step 2: 全局平均池化
        # 将序列表示 [batch, seq_len, hidden_dim] 池化为 [batch, hidden_dim]
        # 这里使用平均池化来聚合所有token的信息
        pooled_features = torch.mean(features, dim=1)
        
        # Step 3: 分类预测
        logits = self.head(pooled_features)

        # Step 4: 损失计算或结果返回
        if labels is not None:
            # 训练模式：计算多标签分类损失
            # BCEWithLogitsLoss结合了sigmoid激活和二元交叉熵损失
            # 适用于多标签分类任务（一个样本可能有多个正标签）
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.float())
            return loss
        else:
            # 推理模式：返回预测logits、注意力权重和特征表示
            return logits, attn_weights, pooled_features

    def load_from(self, weights):
        """
        从预训练权重加载模型参数
        
        Args:
            weights: 预训练权重字典
            
        这个函数处理复杂的权重加载逻辑，包括：
        1. 分类头权重的处理（零初始化或预训练权重）
        2. 位置编码的尺寸适配
        3. 各层参数的逐一加载
        
        注意：这通常在模型微调时使用，允许在预训练模型基础上适配新任务
        """
        with torch.no_grad():
            if self.zero_head:
                # 零初始化分类头（用于新任务微调）
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                # 加载预训练分类头权重
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            if self.use_image:
                self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
                self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
                self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))

                posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
                posemb_new = self.transformer.embeddings.position_embeddings
                if posemb.size() == posemb_new.size():
                    self.transformer.embeddings.position_embeddings.copy_(posemb)
                else:
                    logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                    ntok_new = posemb_new.size(1)

                    if self.classifier == "token":
                        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                        ntok_new -= 1
                    else:
                        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                    gs_old = int(np.sqrt(len(posemb_grid)))
                    gs_new = int(np.sqrt(ntok_new))
                    print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                    posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                    posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                    self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if hasattr(self.transformer.embeddings, 'hybrid') and self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

CONFIGS = {
    'IRENE': configs.get_IRENE_config(),
}
