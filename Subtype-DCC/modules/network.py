import torch.nn as nn
import torch
from torch.nn.functional import normalize

# 定义 Subtype-DCC 的主网络类。
# 该模型对应论文 Figure 1 中的整体架构，包含特征提取主干、实例级投影头和聚类级投影头。
class Network(nn.Module):
    def __init__(self, ae, feature_dim, class_num):
        #ae: 预定义好的自动编码器 (Autoencoder) 对象，用作特征提取的主干网络 (Backbone)。
        #    对应论文中的 "Pair Construction Backbone" 和 "Encoder" 部分。
        #feature_dim: 实例级对比学习的特征维度 (Feature Dimension)。
        #             在论文 Page 6 "Experimental settings" 中提到，该维度被设置为 128。
        #class_num: 聚类的簇数量 (Cluster Number)，即癌症亚型的数量 (M)。

        super(Network, self).__init__() #调用父类 nn.Module 的初始化方法
        self.ae = ae #保存传入的自动编码器实例，作为提取潜在表示 (Embedding) 的编码器
        self.feature_dim = feature_dim #保存实例特征维度参数
        self.cluster_num = class_num #保存聚类数量参数

        # 定义实例级投影头 (Instance-level Projector)，对应论文中的 "Instance-level Contrastive Head (ICH)"。
        # 论文提到：stack a two-layer nonlinear MLP to map the embedding matrix to a subspace。（堆叠双层非线性 MLP，将嵌入矩阵映射到子空间中。）
        self.instance_projector = nn.Sequential(
            nn.Linear(self.ae.rep_dim, self.ae.rep_dim), #第一层线性变换：输入维度为编码器的表示维度 (rep_dim，如 256)，输出维度保持不变
            nn.ReLU(), #ReLU 激活函数：引入非线性特征
            nn.Linear(self.ae.rep_dim, self.feature_dim), #第二层线性变换：将特征映射到用于对比学习的特定维度 (feature_dim，如 128)
        )
        # 定义聚类级投影头 (Cluster-level Projector)，对应论文中的 "Cluster-level Contrastive Head (CCH)"。
        # 论文提到：projecting the embedding matrix into an M-dimensional space using another two-layer MLP。（使用另一个双层 MLP 将嵌入矩阵投射到 M 维空间中。）
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.ae.rep_dim, self.ae.rep_dim), #第一层线性变换：和实例级投影头一样，输入维度为编码器的表示维度 (rep_dim，如 256)，输出维度保持不变
            nn.ReLU(), #ReLU 激活函数
            nn.Linear(self.ae.rep_dim, self.cluster_num), #第二层线性变换：将特征映射到聚类空间，输出维度为簇的数量 (cluster_num，即 M)。输出的向量的每个元素代表样本属于对应簇的概率分数
            nn.Softmax(dim=1) #Softmax 层：将输出转换为概率分布 (Soft Labels)。论文提到：the feature vector denotes its soft label accordingly。
        )

    # 定义训练时的前向传播逻辑 (Forward Propagation)。
    # 接收两个经过数据增强的视图 (Augmented Views)，对应论文中的 x_i^a 和 x_i^b。
    def forward(self, x_i, x_j):
        # x_i: 第一个视图的输入数据张量。
        # x_j: 第二个视图的输入数据张量（由同一样本经数据增强得到）。

        # 通过编码器 (Backbone) 提取两个视图的潜在特征表示 (Latent Embeddings)
        # h_i 和 h_j 对应论文中的 h_i^a 和 h_i^b
        h_i = self.ae(x_i)
        h_j = self.ae(x_j)

        # 1. 实例级路径 (Instance-level Path)
        # 将潜在特征通过实例投影头，并进行 L2 归一化 (Normalize)。
        # 归一化是为了计算余弦相似度 (Cosine Similarity)，这是对比损失函数 (Contrastive Loss) 的要求。
        # z_i, z_j 对应论文公式 (1) 中的 z_j^a
        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        # 2. 聚类级路径 (Cluster-level Path)
        # 将潜在特征通过聚类投影头，得到软聚类分配概率 (Soft Cluster Assignments)。
        # c_i, c_j 对应论文公式 (6) 附近描述的 y_i^a (即 cluster probability vector)
        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        # 返回四个输出：两个视图的实例特征 (z) 和聚类概率 (c)，用于计算 Instance Loss 和 Cluster Loss
        return z_i, z_j, c_i, c_j

    # 定义推理/测试阶段的前向传播逻辑。
    # 用于在模型训练完成后，对样本进行聚类预测。
    # 对应图 1 底部显示 "Predict subtype clustering"，即利用训练好的 CCH 分支输出最终的聚类结果。
    # 在推理阶段对模型进行测试的时候使用
    def forward_cluster(self, x):
        h = self.ae(x) #通过编码器提取潜在特征
        c = self.cluster_projector(h) #通过聚类投影头得到属于各个簇的概率
        c = torch.argmax(c, dim=1) #使用 argmax 获取概率最大的索引，作为最终的预测聚类标签 (Hard Cluster Label)
        return c,h #返回预测的聚类标签 c 和潜在特征 h (特征 h 可用于后续的可视化或分析，如 t-SNE)