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
import pdb

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        tk_lim = config.cc_len
        num_lab = config.lab_len

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.use_image:
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=patch_size)
            self.position_embeddings = nn.Parameter(torch.zeros(1, 1 + n_patches, config.hidden_size))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.dropout = Dropout(config.transformer["dropout_rate"])

        if self.use_text:
            self.cc_embeddings = Linear(768, config.hidden_size)
            self.lab_embeddings = Linear(1, config.hidden_size)
            self.sex_embeddings = Linear(1, config.hidden_size)
            self.age_embeddings = Linear(1, config.hidden_size)

            self.pe_cc = nn.Parameter(torch.zeros(1, tk_lim, config.hidden_size))
            self.pe_lab = nn.Parameter(torch.zeros(1, num_lab, config.hidden_size))
            self.pe_sex = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.pe_age = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

            self.dropout_cc = Dropout(config.transformer["dropout_rate"])
            self.dropout_lab = Dropout(config.transformer["dropout_rate"])
            self.dropout_sex = Dropout(config.transformer["dropout_rate"])
            self.dropout_age = Dropout(config.transformer["dropout_rate"])

            if not self.use_image:
                self.text_cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, x, cc, lab, sex, age):
        image_embeddings = None
        text_embeddings = None

        if self.use_image and x is not None:
            B = x.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)

            if self.hybrid:
                x = self.hybrid_model(x)
            x = self.patch_embeddings(x)
            x = x.flatten(2)
            x = x.transpose(-1, -2)
            x = torch.cat((cls_tokens, x), dim=1)

            image_embeddings = x + self.position_embeddings
            image_embeddings = self.dropout(image_embeddings)

        if self.use_text and cc is not None and lab is not None and sex is not None and age is not None:
            cc = self.cc_embeddings(cc)
            lab = self.lab_embeddings(lab)
            sex = self.sex_embeddings(sex)
            age = self.age_embeddings(age)

            cc_embeddings = cc + self.pe_cc
            lab_embeddings = lab + self.pe_lab
            sex_embeddings = sex + self.pe_sex
            age_embeddings = age + self.pe_age

            cc_embeddings = self.dropout_cc(cc_embeddings)
            lab_embeddings = self.dropout_lab(lab_embeddings)
            sex_embeddings = self.dropout_sex(sex_embeddings)
            age_embeddings = self.dropout_age(age_embeddings)

            text_embeddings = torch.cat((cc_embeddings, lab_embeddings, sex_embeddings, age_embeddings), 1)

            if not self.use_image:
                B = text_embeddings.shape[0]
                text_cls_tokens = self.text_cls_token.expand(B, -1, -1)
                text_embeddings = torch.cat((text_cls_tokens, text_embeddings), dim=1)

        return image_embeddings, text_embeddings