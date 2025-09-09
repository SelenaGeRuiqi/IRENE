"""
IRENE模型的多模态注意力机制实现

1. 四种注意力模式：
   - 图像内部自注意力 (img→img)
   - 文本内部自注意力 (text→text)  
   - 图像到文本的跨模态注意力 (img→text)
   - 文本到图像的跨模态注意力 (text→img)

2. 模态感知的融合策略：
   - 每个模态接收来自自身和对方模态的信息
   - 通过平均融合实现跨模态信息整合

3. 双模式工作：
   - 标准模式：处理融合后的单一序列
   - 多模态模式：分别处理并交互两个模态
"""
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import models.configs as configs
import math

class Attention(nn.Module):
    
    def __init__(self, config, vis, mm=True):
        """
        初始化多模态注意力机制
        
        Args:
            config: 模型配置参数
            vis: 是否启用可视化，返回注意力权重
            mm: 是否为多模态模式
                - True: 创建跨模态注意力组件
                - False: 创建标准自注意力
        """
        super(Attention, self).__init__()
        self.vis = vis
        
        # 多头注意力的基本参数
        self.num_attention_heads = config.transformer["num_heads"]        # 注意力头数，通常为12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 每个头的维度，768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size       # 所有头的总维度，768

        # 图像模态的Q、K、V投影层（所有模式都有）
        self.query = Linear(config.hidden_size, self.all_head_size)   # 查询投影
        self.key = Linear(config.hidden_size, self.all_head_size)     # 键投影
        self.value = Linear(config.hidden_size, self.all_head_size)   # 值投影

        # 多模态模式：为文本模态创建独立的Q、K、V投影
        if mm:
            # 文本模态专用的投影层
            self.query_text = Linear(config.hidden_size, self.all_head_size)  # 文本查询投影
            self.key_text = Linear(config.hidden_size, self.all_head_size)    # 文本键投影
            self.value_text = Linear(config.hidden_size, self.all_head_size)  # 文本值投影
            self.out_text = Linear(config.hidden_size, config.hidden_size)    # 文本输出投影
            
            # 不同注意力模式的独立dropout层
            self.attn_dropout_text = Dropout(config.transformer["attention_dropout_rate"])  # 文本自注意力dropout
            self.attn_dropout_it = Dropout(config.transformer["attention_dropout_rate"])    # 图像→文本注意力dropout
            self.attn_dropout_ti = Dropout(config.transformer["attention_dropout_rate"])    # 文本→图像注意力dropout
            self.proj_dropout_text = Dropout(config.transformer["attention_dropout_rate"])  # 文本输出dropout

        # 图像模态的输出投影和dropout（所有模式都有）
        self.out = Linear(config.hidden_size, config.hidden_size)  # 图像输出投影
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])     # 图像注意力dropout
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])    # 图像输出dropout

        # Softmax激活函数，用于计算注意力权重
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        """
        重塑tensor以适应多头注意力计算
        
        将形状从 [batch_size, seq_len, hidden_size] 
        转换为 [batch_size, num_heads, seq_len, head_size]
        
        这种重塑允许每个注意力头并行计算，提高效率
        """
        # 计算新的形状：[batch_size, seq_len, num_heads, head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        
        # 交换维度：[batch_size, num_heads, seq_len, head_size]
        # 这样每个头可以独立处理
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, text=None):
        """
        多模态注意力前向传播
        
        Args:
            hidden_states: 图像特征 [batch_size, img_seq_len, hidden_dim]
            text: 文本特征 [batch_size, text_seq_len, hidden_dim]（可选）
        
        Returns:
            标准模式: (attention_output, weights)
            多模态模式: (img_output, text_output, weights)
        
        处理流程：
        1. 标准模式：执行传统的自注意力计算
        2. 多模态模式：执行四种注意力交互并融合结果
        """
        # === 第一步：计算图像模态的Q、K、V ===
        mixed_query_layer = self.query(hidden_states)   # 图像查询
        mixed_key_layer = self.key(hidden_states)       # 图像键
        mixed_value_layer = self.value(hidden_states)   # 图像值

        # 如果是多模态模式，计算文本模态的Q、K、V
        if text is not None:
            text_q = self.query_text(text)      # 文本查询
            text_k = self.key_text(text)        # 文本键
            text_v = self.value_text(text)      # 文本值

        # === 第二步：重塑为多头注意力格式 ===
        query_layer = self.transpose_for_scores(mixed_query_layer)   # [batch, heads, img_len, head_dim]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 多模态模式：重塑文本的Q、K、V
        if text is not None:
            # 保存图像的Q、K、V用于跨模态计算
            query_layer_img = query_layer
            key_layer_img = key_layer
            value_layer_img = value_layer
            
            # 重塑文本的Q、K、V
            query_layer_text = self.transpose_for_scores(text_q)     # [batch, heads, text_len, head_dim]
            key_layer_text = self.transpose_for_scores(text_k)
            value_layer_text = self.transpose_for_scores(text_v)

        # === 第三步：注意力计算 ===
        if text is None:
            # ========== 标准模式：单模态自注意力 ==========
            # 计算注意力分数：Q * K^T
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            
            # 缩放：除以sqrt(head_size)，防止softmax饱和
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            
            # 应用softmax得到注意力权重
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs if self.vis else None  # 可视化权重
            attention_probs = self.attn_dropout(attention_probs)

            # 应用注意力权重到值：Attention * V
            context_layer = torch.matmul(attention_probs, value_layer)
            
            # 重塑回原始格式：[batch, seq_len, hidden_size]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            
            # 输出投影和dropout
            attention_output = self.out(context_layer)
            attention_output = self.proj_dropout(attention_output)
            
            return attention_output, weights
            
        else:
            # ========== 多模态模式：四种注意力交互 ==========
            
            # 1. 图像内部自注意力 (img → img)
            attention_scores_img = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            
            # 2. 文本内部自注意力 (text → text)
            attention_scores_text = torch.matmul(query_layer_text, key_layer_text.transpose(-1, -2))
            
            # 3. 图像到文本的跨模态注意力 (img → text)
            # 图像的查询关注文本的键值
            attention_scores_it = torch.matmul(query_layer_img, key_layer_text.transpose(-1, -2))
            
            # 4. 文本到图像的跨模态注意力 (text → img)
            # 文本的查询关注图像的键值
            attention_scores_ti = torch.matmul(query_layer_text, key_layer_img.transpose(-1, -2))
            
            # === 第四步：分别计算四种注意力权重 ===
            
            # 图像自注意力
            attention_scores_img = attention_scores_img / math.sqrt(self.attention_head_size)
            attention_probs_img = self.softmax(attention_scores_img)
            weights = attention_probs_img if self.vis else None  # 返回图像注意力权重用于可视化
            attention_probs_img = self.attn_dropout(attention_probs_img)

            # 文本自注意力
            attention_scores_text = attention_scores_text / math.sqrt(self.attention_head_size)
            attention_probs_text = self.softmax(attention_scores_text)
            attention_probs_text = self.attn_dropout_text(attention_probs_text)

            # 图像→文本跨模态注意力
            attention_scores_it = attention_scores_it / math.sqrt(self.attention_head_size)
            attention_probs_it = self.softmax(attention_scores_it)
            attention_probs_it = self.attn_dropout_it(attention_probs_it)

            # 文本→图像跨模态注意力
            attention_scores_ti = attention_scores_ti / math.sqrt(self.attention_head_size)
            attention_probs_ti = self.softmax(attention_scores_ti)
            attention_probs_ti = self.attn_dropout_ti(attention_probs_ti)

            # === 第五步：应用注意力权重计算上下文向量 ===
            
            # 图像自注意力上下文
            context_layer_img = torch.matmul(attention_probs_img, value_layer_img)
            context_layer_img = context_layer_img.permute(0, 2, 1, 3).contiguous()
            
            # 文本自注意力上下文
            context_layer_text = torch.matmul(attention_probs_text, value_layer_text)
            context_layer_text = context_layer_text.permute(0, 2, 1, 3).contiguous()
            
            # 图像→文本跨模态上下文（图像查询关注文本值）
            context_layer_it = torch.matmul(attention_probs_it, value_layer_text)
            context_layer_it = context_layer_it.permute(0, 2, 1, 3).contiguous()
            
            # 文本→图像跨模态上下文（文本查询关注图像值）
            context_layer_ti = torch.matmul(attention_probs_ti, value_layer_img)
            context_layer_ti = context_layer_ti.permute(0, 2, 1, 3).contiguous()
            
            # === 第六步：重塑和融合上下文向量 ===
            
            # 重塑所有上下文向量为 [batch, seq_len, hidden_size]
            new_context_layer_shape = context_layer_img.size()[:-2] + (self.all_head_size,)
            context_layer_img = context_layer_img.view(*new_context_layer_shape)
            
            new_context_layer_shape = context_layer_text.size()[:-2] + (self.all_head_size,)
            context_layer_text = context_layer_text.view(*new_context_layer_shape)
            
            new_context_layer_shape = context_layer_it.size()[:-2] + (self.all_head_size,)
            context_layer_it = context_layer_it.view(*new_context_layer_shape)
            
            new_context_layer_shape = context_layer_ti.size()[:-2] + (self.all_head_size,)
            context_layer_ti = context_layer_ti.view(*new_context_layer_shape)
            
            # === 第七步：跨模态信息融合 ===
            # 核心创新：每个模态融合来自自身和对方模态的信息
            
            # 图像最终输出 = (图像自注意力 + 文本→图像跨模态) / 2
            # 这样图像特征既保持自身信息，又整合了文本提供的信息
            attention_output_img = self.out((context_layer_img + context_layer_ti) / 2)
            
            # 文本最终输出 = (文本自注意力 + 图像→文本跨模态) / 2  
            # 这样文本特征既保持自身信息，又整合了图像提供的信息
            attention_output_text = self.out_text((context_layer_text + context_layer_it) / 2)
            
            # 应用dropout
            attention_output_img = self.proj_dropout(attention_output_img)
            attention_output_text = self.proj_dropout_text(attention_output_text)

            return attention_output_img, attention_output_text, weights
