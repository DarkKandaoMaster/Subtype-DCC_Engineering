import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import itertools
import numpy as np
import os



# 定义一个基础的网络块（Block），包含一个线性层和一个激活层。
# in_c (int): 输入特征的维度（Input Channels/Dimensions）。
# out_c (int): 输出特征的维度（Output Channels/Dimensions）。
def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c), # 全连接层 (Fully Connected Layer)，执行线性变换 y = xA^T + b
        nn.ReLU(True) # ReLU 激活函数 (Rectified Linear Unit)，引入非线性，参数 True 表示原地操作 (in-place) 以节省内存
    ]
    return layers



# 编码器类 (Encoder)，用于将高维度的输入数据压缩为低维度的潜在表示 (Latent Representation)。
# 对应论文中提到的用于特征提取的深度神经网络。
class Encoder(nn.Module):
    def __init__(self,input_dim=9844,inter_dims=[5000,2000,1000,256]): #初始化编码器结构。
        # input_dim (int): 输入数据的维度。默认为 9844。
        #                  (结合论文 Page 2 "Benchmark datasets" 章节：
        #                   3105 Copy Number + 3217 mRNA + 383 miRNA + 3139 DNA methylation = 9844，
        #                   这对应四种组学数据的特征总和。)
        # inter_dims (list): 隐藏层和输出层的维度列表，依次递减 [5000, 2000, 1000, 256]。
        #                    (对应论文 Page 5 "Subtype-DCC uses MLPs with 5000, 2000, 1000 and 256 neurons")

        super(Encoder,self).__init__() # 调用父类 nn.Module 的初始化方法，这是 PyTorch 模型定义的标准写法

        self.encoder=nn.Sequential( #nn.Sequential 是一个容器，数据会按顺序通过容器中的每一层
            nn.Dropout(), #Dropout 层，随机丢弃一部分神经元，防止过拟合 (Overfitting)
            # *block(...) 使用了解包操作符 (*)，将 block 函数返回的列表元素（线性层和 ReLU）解压作为参数传入 Sequential
            *block(input_dim,inter_dims[0]), #第1层: 输入 9844 -> 输出 5000
            *block(inter_dims[0],inter_dims[1]), #第2层: 输入 5000 -> 输出 2000
            *block(inter_dims[1],inter_dims[2]), #第3层: 输入 2000 -> 输出 1000
            *block(inter_dims[2],inter_dims[3]) #第4层: 输入 1000 -> 输出 256 (这是最终的编码特征维度)
        )

    def forward(self, x): #前向传播函数 (Forward Propagation)，定义数据如何流过网络。
        # x (Tensor): 输入数据张量，形状通常为 (batch_size, input_dim)。
        z=self.encoder(x) #将输入 x 传入编码器网络，得到编码后的特征 z
        return z #返回潜在表示 z



# 解码器类 (Decoder)，通常用于将低维特征重构回高维空间。
# (注意：在 Subtype-DCC 方法中，主要利用的是 Encoder 提取特征进行对比学习，Decoder 在此处可能用于预训练或辅助任务)
class Decoder(nn.Module):
    def __init__(self,input_dim=9844,inter_dims=[5000,2000,1000,256]):
        super(Decoder,self).__init__() #初始化父类

        # 定义解码器的层级结构，顺序与编码器相反（从低维到高维）
        self.decoder=nn.Sequential(
            *block(inter_dims[-1],inter_dims[-2]), # 第1层: 输入 256 -> 输出 1000
            *block(inter_dims[-2],inter_dims[-3]), # 第2层: 输入 1000 -> 输出 2000
            *block(inter_dims[-3],inter_dims[-4]) # 第3层: 输入 2000 -> 输出 5000
            #没有第4层，此处代码并未重构回原始的 input_dim 9844，而是停在了 5000 维度
        )

    def forward(self, z): #前向传播函数。
        # z (Tensor): 编码器的输出（潜在表示）。
        x_out=self.decoder(z) #将潜在表示 z 传入解码器
        return x_out #返回解码器的输出


class AE(nn.Module):
    # 自动编码器类 (Autoencoder)，整合了 Encoder 和 Decoder。
    def __init__(self,hid_dim=256):
        # hid_dim (int): 潜在层的维度，即 Encoder 输出的特征维度。

        super(AE,self).__init__() # 初始化父类
        self.encoder=Encoder() # 实例化 Encoder 类
        self.decoder=Decoder() # 实例化 Decoder 类

        self.rep_dim = hid_dim # 记录表示层的维度 (Representation Dimension)，供外部调用（如在 network.py 中使用）

    def forward(self, x): #前向传播函数。
        z = self.encoder(x) # 数据通过编码器，得到特征 z
        x_out = self.decoder(z) # 特征 z 通过解码器（代码中执行了这一步，但在下面的 return 中没有返回它）
        
        return z # 仅返回编码后的特征 z，而没有返回x_out。(这表明在后续的 Subtype-DCC 训练流程中，主要使用的是 Encoder 提取的特征，而不是 AE 的重构输出)