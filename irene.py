from __future__ import print_function, division 
import os
import sys
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn
import pickle
import pandas as pd
from PIL import Image
import argparse
# from apex import amp  # NVIDIA的混合精度训练库，可以加速训练并减少显存使用
# from sklearn.metrics.ranking import roc_auc_score  # ROC-AUC评估指标，医疗诊断中的重要性能指标
from sklearn.metrics import roc_auc_score
from models.modeling_irene import IRENE, CONFIGS  # 导入IRENE模型架构和配置
from tqdm import tqdm  # 进度条显示库
import argparse
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# 设置临床描述文本的最大token长度限制
# 这是为了保证所有样本的文本输入长度一致，便于批处理
tk_lim = 40

# 定义8种肺部疾病类别
# IRENE模型可以同时预测多种疾病的存在概率（多标签分类）
disease_list = ['COPD', 'Bronchiectasis', 'Pneumothorax', 'Pneumonia', 'ILD', 'Tuberculosis', 'Lung cancer', 'Pleural effusion']

def load_weights(model, weight_path):
    pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
    model_weights = model.state_dict()

    load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

    model_weights.update(load_weights)
    model.load_state_dict(model_weights)
    print("Loading IRENE...")
    return model

def computeAUROC (dataGT, dataPRED, classCount=8):
    """
    计算每个疾病类别的AUROC（Area Under the ROC Curve）值
    
    Args:
        dataGT: 真实标签（Ground Truth），形状为 [样本数, 疾病数]
        dataPRED: 模型预测概率，形状为 [样本数, 疾病数]  
        classCount: 疾病类别数量，默认为8
    
    Returns:
        每个疾病类别的AUROC值列表
    
    AUROC是医疗诊断中的重要评估指标：
    - 值越接近1.0表示模型性能越好
    - 0.5表示随机猜测的水平
    - 在多标签分类中，每个疾病都会有独立的AUROC值
    """
    outAUROC = []
        
    # 将GPU上的tensor转换为CPU上的numpy数组，便于sklearn处理
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
        
    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC

class FlexibleData(Dataset):
    """
    自定义数据集类，用于加载多模态医疗数据
    
    IRENE模型需要处理4种不同模态的数据：
    1. 医学影像（胸片）
    2. 临床描述文本（主诉症状）
    3. 人口统计学信息（年龄、性别）
    4. 实验室检查结果
    
    这个类继承自PyTorch的Dataset类，实现了数据加载的标准接口
    """
    def __init__(self, set_type, img_dir, mode = 'image', transform=None, target_transform=None):
        """
        初始化数据集
        
        Args:
            set_type: 数据集类型标识符（如'train'、'test'等）
            img_dir: 医学影像文件所在目录
            transform: 图像预处理变换
            target_transform: 标签预处理变换
        """
        dict_path = set_type + '.pkl'
        f = open(dict_path, 'rb')
        self.mm_data = pickle.load(f)
        f.close()
        self.idx_list = list(self.mm_data.keys())
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.mode = mode
        self.use_image = mode in ['image', 'multimodal']
        self.use_text = mode in ['text', 'multimodal']

        print(f"data loading mode: {mode}, use image: {self.use_image}, use text: {self.use_text}")

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        """
        根据索引获取一个样本的所有模态数据
        
        Returns:
            img: 预处理后的医学影像
            label: 疾病标签（8种疾病的二进制标签）
            cc: 临床主诉描述的特征向量
            demo: 人口统计学信息（年龄、性别）
            lab: 实验室检查结果
        """
        k = self.idx_list[idx]
        label = self.mm_data[k]['label'].astype('float32')

        img = None
        cc = None
        demo = None
        lab = None
        
        img_path = os.path.join(self.img_dir, k) + '.png'
        img = Image.open(img_path).convert('RGB')  # 转换为RGB格式
        
        # image
        if self.use_image:
            img_path = os.path.join(self.img_dir, k) + '.png'
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        # text
        if self.use_text:
            cc = torch.from_numpy(self.mm_data[k]['pdesc']).float()
            demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()
            lab = torch.from_numpy(self.mm_data[k]['bts']).float()

        if self.target_transform:
            label = self.target_transform(label)
        
        return img, label, cc, demo, lab

# create IRENE based on mode
def create_model_with_mode(mode, num_classes):
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
    print(f"mode: {mode}, image: {config.modality.use_image}, text: {config.modality.use_text}")

    return model, config

def test(args):
    """
    模型测试函数
    1. 初始化IRENE模型
    2. 加载预训练权重
    3. 准备测试数据
    4. 执行前向推理
    5. 计算性能指标
    """
    torch.manual_seed(0)
    num_classes = args.CLS  # 疾病类别数量
    mode = args.MODE

    irene, config = create_model_with_mode(mode, num_classes)
    irene = load_weights(irene, 'model.pth')
    img_dir = args.DATA_DIR

    if config.modality.use_image:
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]),
        }
        transform = data_transforms['test']
    else:
        transform = None

    test_data = FlexibleData(args.SET_TYPE, img_dir, mode=mode, transform=transform)
    testloader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=16, pin_memory=True)

    # optimizer_irene = torch.optim.AdamW(irene.parameters(), lr=3e-5, weight_decay=0.01)
    # irene, optimizer_irene = amp.initialize(irene.cuda(), optimizer_irene, opt_level="O1")
    # 移除apex 相关代码，使用标准 PyTorch
    irene = irene.cuda ()
    optimizer_ irene = torch.optim.AdamW(irene.parameters(), 1r=3e-5, weight_decay=0.01)

    irene = torch.nn.DataParallel(irene)

    #----- 开始测试 ------
    print('--------Start testing-------')
    print(f'test mode: {mode}, batch size: {args.BSZ}, num of classes: {num_classes}')
    irene.eval()

    with torch.no_grad():
        outGT = torch.FloatTensor().cuda(non_blocking=True)
        outPRED = torch.FloatTensor().cuda(non_blocking=True)

        for data in tqdm(testloader):
            # get the inputs; data is a list of [inputs, labels]
            imgs, labels, cc, demo, lab = data
            labels = labels.cuda(non_blocking=True)

            # 根据模态模式处理输入
            if mode == 'image':
                # 纯图像模式
                imgs = imgs.cuda(non_blocking=True)
                preds = irene(x=imgs, cc=None, lab=None, sex=None, age=None)[0]

            elif mode == 'text':
                # 纯文本模式
                cc = cc.view(-1, tk_lim, cc.shape[3]).cuda(non_blocking=True).float()
                demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
                lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
                sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
                age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
                preds = irene(x=None, cc=cc, lab=lab, sex=sex, age=age)[0]

            elif mode == 'multimodal':
                # 多模态模式
                imgs = imgs.cuda(non_blocking=True)
                cc = cc.view(-1, tk_lim, cc.shape[3]).cuda(non_blocking=True).float()
                demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
                lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
                sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()
                age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()
                preds = irene(x=imgs, cc=cc, lab=lab, sex=sex, age=age)[0]

            probs = torch.sigmoid(preds)
            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)

        aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
        aurocMean = np.array(aurocIndividual).mean()

        print(f'[result] {mode}  average AUROC: {str(aurocMean)}')

        for i in range(0, len(aurocIndividual)):
            print(f'[result] {disease_list[i]}: {str(aurocIndividual[i])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRENE flexible training")
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int,
                        help='分类数量')
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int,
                        help='批次大小')
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str,
                        help='数据目录')
    parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', required=True, type=str,
                        help='数据集类型')
    # 模态模式参数
    parser.add_argument('--MODE', action='store', dest='MODE', required=True, type=str,
                        choices=['image', 'text', 'multimodal'],
                        help='模态模式: image(纯图像), text(纯文本), multimodal(多模态)')

    args = parser.parse_args()
    print(f"[start] IRENE {args.MODE} mode test")
    test(args)