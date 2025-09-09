IRENE:
基于Transformer的、统一处理多模态医疗数据的诊断模型 。
将影像、主诉、化验单等多种信息统一输入到一个模型中，进行端到端的诊断，而不是‘先提取特征再融合(早/晚期融合)的传统方法 

1. embedding [→](embed.py)
image: 16*16 --> CNN --> 1*1 patch
text: cc, lab, sex, age --> linear

2. encoder [→](encoder.py) block [→](block.py)
Unified Transformer:
    模式1 - 标准模式（mm=False）：
    - 处理单一的融合后序列
    - 类似标准Transformer Block的行为
    - 用于模态融合后的层
    
    模式2 - 多模态模式（mm=True）：
    - 同时处理图像和文本两个独立序列
    - 每个模态有独立的归一化层和MLP
    - 保持模态特异性，用于早期层

    1. 层0-1：分离处理阶段，图像和文本各自进行自注意力计算
    2. 层2：融合阶段， 将图像和文本特征在序列维度拼接
    3. 层3-11：统一处理阶段，对融合序列进行标准Transformer处理

实现：attn [→](attention.py)
if mm: 多模态模式, 为文本模态创建独立的Q、K、V投影
    额外创建一套专门用于文本模态的query_text, key_text, value_text线性层。同时，还会创建额外的Dropout层，如attn_dropout_it (image-to-text) 和 attn_dropout_ti (text-to-image)，用于跨模态注意力的正则化。
else: 标准模式, 传统的自注意力计算 
    (1) text is not None: 在前两个block分别处理图像和文本
        分别生成图像和文本的KQV，计算文文，图图的self attn，和文图，图文的cross attn，最后文本输出文文和文图的平均，图片输出图图和图文的平均
    2） text is None: layer3以后，图像和文本token已被拼接成一个序列，只有一个hidden_states输入
        标准self attn



图像only，文本only，图像+文本

1. config.py
config.modality.use_image = True/False
config.modality.use_text = True/False
config.modality.mode = 'image'/'text'/'multimodal'

2. embedding [→](embed.py)
if self.use_image:
    self.patch_embeddings = Conv2d(...)  # 图像patch嵌入
    self.position_embeddings = nn.Parameter(...)  # 图像位置编码
    self.cls_token = nn.Parameter(...)  # 图像CLS token
 
if self.use_text:
    self.cc_embeddings = Linear(...)
    self.lab_embeddings = Linear(...)
    self.sex_embeddings = Linear(...)
    self.age_embeddings = Linear(...)

3. encoder [→](encoder.py)
在Encoder.__init__()中
for i in range(config.transformer["num_layers"]):
    只有多模态模式前2层才用多模态注意力
    if i < 2 and self.use_image and self.use_text:
        layer = Block(config, vis, mm=True)  # 多模态Block
    else:
        layer = Block(config, vis, mm=False)  # 标准Block

在Transformer.forward()中
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

### 这些参数在三种模态下都是相同的
self.transformer.encoder.encoder_norm  # ✅ 共享
self.head  # ✅ 共享
self.transformer.encoder.layer[3-11]  # ✅ 共享

### 这些参数是模态特定的
self.transformer.embeddings.patch_embeddings  # 🔸 仅图像模态
self.transformer.embeddings.cc_embeddings     # 🔸 仅文本模态
self.transformer.encoder.layer[0-1]  # 🔸 多模态注意力