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
from apex import amp  # NVIDIA的混合精度训练库，可以加速训练并减少显存使用
from sklearn.metrics.ranking import roc_auc_score  # ROC-AUC评估指标，医疗诊断中的重要性能指标
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

class Data(Dataset):
    """
    自定义数据集类，用于加载多模态医疗数据
    
    IRENE模型需要处理4种不同模态的数据：
    1. 医学影像（胸片）
    2. 临床描述文本（主诉症状）
    3. 人口统计学信息（年龄、性别）
    4. 实验室检查结果
    
    这个类继承自PyTorch的Dataset类，实现了数据加载的标准接口
    """
    def __init__(self, set_type, img_dir, transform=None, target_transform=None):
        """
        初始化数据集
        
        Args:
            set_type: 数据集类型标识符（如'train'、'test'等）
            img_dir: 医学影像文件所在目录
            transform: 图像预处理变换
            target_transform: 标签预处理变换
        """
        # 加载临床数据字典文件（.pkl格式）
        # 这个文件包含了每个患者的临床文本、人口统计学信息、实验室结果和疾病标签
        dict_path = set_type+'.pkl'
        f = open(dict_path, 'rb') 
        self.mm_data = pickle.load(f)  # mm_data = multimodal_data
        f.close()
        self.idx_list = list(self.mm_data.keys())  # 获取所有患者ID列表
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

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
        
        img_path = os.path.join(self.img_dir, k) + '.png'
        img = Image.open(img_path).convert('RGB')  # 转换为RGB格式

        # 获取疾病标签并转换为float32类型
        label = self.mm_data[k]['label'].astype('float32')
        
        # 图像预处理变换
        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            label = self.target_transform(label)

        # 加载并转换临床文本数据
        # 'pdesc' = patient description，患者的临床主诉描述
        cc = torch.from_numpy(self.mm_data[k]['pdesc']).float()
        
        # 加载人口统计学信息
        # 'bics' = basic information clinical statistics，包含年龄和性别
        demo = torch.from_numpy(np.array(self.mm_data[k]['bics'])).float()
        
        # 加载实验室检查结果
        # 'bts' = blood test statistics，血液检查等实验室指标
        lab = torch.from_numpy(self.mm_data[k]['bts']).float()
        
        return img, label, cc, demo, lab

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
    
    config = CONFIGS["IRENE"]
    
    # 初始化IRENE模型
    # 224: 输入图像尺寸
    # zero_head=True: 使用零初始化的分类头
    model = IRENE(config, 224, zero_head=True, num_classes=num_classes)
    
    # 加载预训练权重
    irene = load_weights(model, 'model.pth')
    img_dir = args.DATA_DIR

    # 定义图像预处理流程
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),      # 缩放到256x256 ！！比较常规
            transforms.CenterCrop(224),  # 中心裁剪到224x224
            transforms.ToTensor(),       # 转换为tensor并归一化到[0,1]   ！！归一化可能会困难
        ]),
    }

    # 创建测试数据集
    test_data = Data(args.SET_TYPE, img_dir, transform=data_transforms['test'])

    # 创建数据加载器
    # shuffle=False: 测试时不需要打乱数据
    # num_workers: 多进程加载数据，加速IO
    # pin_memory=True: 将数据预加载到GPU内存，加速GPU传输
    testloader = DataLoader(test_data, batch_size=args.BSZ, shuffle=False, num_workers=16, pin_memory=True)

    # 设置优化器（虽然测试时不需要更新参数，但amp需要optimizer）
    optimizer_irene = torch.optim.AdamW(irene.parameters(), lr=3e-5, weight_decay=0.01)
    
    # 初始化混合精度训练
    # amp (Automatic Mixed Precision) 可以加速推理并减少显存使用
    # opt_level="O1": 保守的混合精度设置，保持数值稳定性
    irene, optimizer_irene = amp.initialize(irene.cuda(), optimizer_irene, opt_level="O1")

    # 使用DataParallel进行多GPU并行推理
    # 这允许模型同时在多个GPU上运行，加速推理过程
    irene = torch.nn.DataParallel(irene)

    #----- 开始测试 ------
    print('--------Start testing-------')
    irene.eval()  # 设置模型为评估模式，关闭dropout和batch normalization的训练行为
    
    with torch.no_grad():  # 关闭梯度计算，节省内存并加速推理
        # 初始化空的tensor来存储所有预测结果和真实标签
        outGT = torch.FloatTensor().cuda(non_blocking=True)      # Ground Truth
        outPRED = torch.FloatTensor().cuda(non_blocking=True)    # Predictions
        
        # 遍历所有测试批次
        for data in tqdm(testloader):
            # 解包数据
            imgs, labels, cc, demo, lab = data
            # 重塑多模态数据的维度以符合模型输入要求
            # cc: 临床描述 [batch_size, token_limit, feature_dim]
            cc = cc.view(-1, tk_lim, cc.shape[3]).cuda(non_blocking=True).float()
            
            # demo: 人口统计学信息 [batch_size, 1, feature_dim]  
            demo = demo.view(-1, 1, demo.shape[1]).cuda(non_blocking=True).float()
            
            # lab: 实验室结果 [batch_size, feature_num, 1]
            lab = lab.view(-1, lab.shape[1], 1).cuda(non_blocking=True).float()
            
            # 从人口统计学信息中分离出性别和年龄
            sex = demo[:, :, 1].view(-1, 1, 1).cuda(non_blocking=True).float()  # 性别信息
            age = demo[:, :, 0].view(-1, 1, 1).cuda(non_blocking=True).float()  # 年龄信息
            
            # 将图像和标签移到GPU
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            # IRENE模型前向推理
            # 输入包括：影像、临床描述、实验室结果、性别、年龄
            # 返回：预测logits, 注意力权重, 特征向量
            preds = irene(imgs, cc, lab, sex, age)[0]

            # 将logits转换为概率
            probs = torch.sigmoid(preds)
            
            outGT = torch.cat((outGT, labels), 0)
            outPRED = torch.cat((outPRED, probs.data), 0)

        # 计算每个疾病类别的AUROC值
        aurocIndividual = computeAUROC(outGT, outPRED, classCount=num_classes)
        
        # 计算平均AUROC值
        aurocMean = np.array(aurocIndividual).mean()
        
        # 输出结果
        print('mean AUROC:' + str(aurocMean))
         
        # 输出每个疾病的AUROC值
        for i in range (0, len(aurocIndividual)):
            print(disease_list[i] + ': '+str(aurocIndividual[i]))

if __name__ == '__main__':
    """
    使用argparse解析命令行参数，支持以下参数：
    --CLS: 疾病类别数量
    --BSZ: 批次大小  
    --DATA_DIR: 影像数据目录路径
    --SET_TYPE: 临床数据文件名（不含.pkl扩展名）
    
    示例运行命令：
    python irene.py --CLS 8 --BSZ 64 --DATA_DIR ./data --SET_TYPE test
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--CLS', action='store', dest='CLS', required=True, type=int)
    parser.add_argument('--BSZ', action='store', dest='BSZ', required=True, type=int)
    parser.add_argument('--DATA_DIR', action='store', dest='DATA_DIR', required=True, type=str)
    parser.add_argument('--SET_TYPE', action='store', dest='SET_TYPE', required=True, type=str)
    args = parser.parse_args()
    test(args)
