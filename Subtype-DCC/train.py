import os
import numpy as np
import torch
import torchvision
import argparse
from modules import ae, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from dataloader import *
import copy
import pandas as pd
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

# 函数功能：在推理阶段对模型进行评估（测试），提取特征和聚类结果。
# loader: 数据加载器 (DataLoader)，用于批量提供测试数据。
# model: 训练好的神经网络模型。
# device: 计算设备（CPU 或 GPU）。
def inference(loader, model, device):
    model.eval() #将模型设置为评估模式 (Evaluation Mode)。这会禁用 Dropout 层和 Batch Normalization 层的统计更新，确保推理结果的稳定性。
    cluster_vector = [] #初始化列表，用于存储所有样本的聚类预测结果
    feature_vector = [] #初始化列表，用于存储所有样本的潜在特征表示
    for step, x in enumerate(loader): #遍历数据加载器loader。step是索引，x是输入的数据张量【【【【【为什么和我的15.py不一样？为什么读出来的是索引,张量而不是其他？
        x = x.float().to(device) # 将输入数据转换为32位浮点型张量 (Float Tensor)，并移动到指定的计算设备 (GPU/CPU) 上
        with torch.no_grad(): #关闭梯度计算。因为推断阶段不需要反向传播
            c,h = model.forward_cluster(x) #调用network.py里自定义的前向传播方法，将x传入模型，获取聚类分配概率 c 和特征向量 h。【【【【【这玩意还能自定义？看看怎么实现的？
        c = c.detach() # 将张量 c 从计算图中分离 (Detach)，使其不再具有梯度历史。【【【【【这是干嘛的？
        h = h.detach() # 将张量 h 从计算图中分离。
        cluster_vector.extend(c.cpu().detach().numpy()) # 将聚类结果移动到 CPU，转换为 NumPy 数组，并添加到结果列表中
        feature_vector.extend(h.cpu().detach().numpy()) # 将特征结果移动到 CPU，转换为 NumPy 数组，并添加到结果列表中
    cluster_vector = np.array(cluster_vector) # 将聚类结果列表转换为 NumPy 数组
    feature_vector = np.array(feature_vector) # 将特征结果列表转换为 NumPy 数组
    print("Features shape {}".format(feature_vector.shape)) # 打印特征矩阵的形状 (样本数, 特征维度)，用于检查数据维度是否符合预期
    return cluster_vector,feature_vector

# 函数功能：执行一个 Epoch (轮次) 的训练流程。返回当前 Epoch 的总损失值
def train():
    loss_epoch = 0 #初始化累加器，用于记录当前 Epoch 的总损失。
    for step, x in enumerate(DL): #遍历训练数据加载器 DL（全局变量） 中的每一个 step (批次)
        optimizer.zero_grad() #梯度清零
        # 数据增强：
        # 向原始数据 x 添加标准正态分布的噪声（高斯噪声）（均值0、标准差1），生成两个增强视图 x_i 和 x_j。这是对比学习常用的构建正样本对的方法。
        x_i = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        x_j = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
        #前向传播，将x_i、x_j传入模型
        # z_i, z_j: 实例级投影头 (Instance-level Projector) 的输出，用于实例对比学习
        # c_i, c_j: 聚类级投影头 (Cluster-level Projector) 的输出，即软聚类分配概率
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        batch = x_i.shape[0] #获取当前批次的样本数量(Batch Size)

        # 实例化损失函数：
        # DCL (Decoupled Contrastive Loss): 解耦对比损失，用于实例级对比学习，旨在拉近正样本对，推开负样本对
        criterion_instance = contrastive_loss.DCL(temperature=0.5, weight_fn=None)
        # ClusterLoss: 聚类损失，用于聚类级对比学习，确保聚类分配的一致性和多样性
        criterion_cluster = contrastive_loss.ClusterLoss(cluster_number, args.cluster_temperature, loss_device).to(loss_device)
        # 计算实例级损失：包含两个方向 (view i -> view j 和 view j -> view i) 的损失之和
        loss_instance = criterion_instance(z_i, z_j)+criterion_instance(z_j, z_i)
        # 计算聚类级损失：基于两个视图的聚类分配概率计算损失
        loss_cluster = criterion_cluster(c_i, c_j)
        # 总损失：将实例损失和聚类损失相加，用于联合优化 (Joint Optimization)
        loss = loss_instance + loss_cluster
        loss.backward() #反向传播
        optimizer.step() #更新model中所有需要学习的参数
        loss_epoch += loss.item() #累加损失
    return loss_epoch

# 绘制训练损失随 Epoch 变化的曲线图并保存。
def draw_fig(list,name,epoch):
    x1 = range(0, epoch+1) # 创建 x 轴数据，表示 Epoch 的序列，从 0 到当前 epoch
    print(x1) # 打印 x 轴序列
    y1 = list # y 轴数据为传入的损失值列表
    save_file = './results/' + name + 'Train_loss.png' # 定义图片保存路径
    plt.cla() # 清除当前活动的轴 (Clear Axes)，防止重叠绘图。
    plt.title('Train loss vs. epoch', fontsize=20) # 设置图表标题及字体大小
    plt.plot(x1, y1, '.-') # 绘制折线图，点线样式为 '.-'
    plt.xlabel('epoch', fontsize=20) # 设置 x 轴标签
    plt.ylabel('Train loss', fontsize=20) # 设置 y 轴标签
    plt.grid() # 显示网格线
    plt.savefig(save_file) # 将图像保存到指定文件
    plt.show() # 显示图像（窗口）。在非交互式环境中可能无效

if __name__ == "__main__":
    parser = argparse.ArgumentParser() #初始化命令行参数解析器。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #如果 CUDA 可用则使用 GPU，否则使用 CPU
    #添加命令行参数定义：
    parser.add_argument("--cancer_type", '-c', type=str, default="BRCA") #癌症类型，默认为 BRCA (乳腺癌) #这个模型并不是用来判断“这个人得的是不是乳腺癌（BRCA）”（这是癌症诊断/分类），而是用来判断“这个已经确诊乳腺癌的病人，具体属于哪一种乳腺癌”（这是癌症亚型鉴定）。
    parser.add_argument('--batch_size', type=int, default=64) # 批次大小 (Batch Size)，默认为 64
    parser.add_argument('--cluster_number', type=int,default=5) # 聚类数量 (Cluster Number)，默认为 5
    args = parser.parse_args() #参数实例化。解析命令行参数，将结果存储在 args 对象中【【【【【这玩意是个啥？
    #定义癌症类型到聚类数量的映射字典 (Domain Knowledge)
    cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
                   'LUAD': 3, 'PAAD': 2, 'SKCM': 4,
                   'STAD': 3, 'UCEC': 4, 'UVM': 4, 'GBM': 2}
    
    cluster_number = cancer_dict[args.cancer_type] #按照癌症种类选择。根据输入的癌症类型，从字典中获取对应的预设聚类数目
    print(cluster_number) #打印聚类数目以确认

    config = yaml_config_hook("config/config.yaml") #加载 YAML 配置文件，yaml_config_hook 是"Subtype-DCC\utils\yaml_config_hook.py"里的自定义函数，可能用于处理嵌套配置【【【【【用来处理什么？
    for k, v in config.items(): #遍历配置文件中的键值对
        parser.add_argument(f"--{k}", default=v, type=type(v)) # 将配置文件中的参数动态添加到 argparse 中【【【【【argparse是什么？
    args = parser.parse_args() #重新解析参数，合并命令行参数和配置文件参数
    model_path = './save/' + args.cancer_type #模型保存路径
    if not os.path.exists(model_path): #如果路径不存在，则创建该目录。
        os.makedirs(model_path)

    # 设置随机种子 (Random Seed) 以确保实验结果的可复现性 (Reproducibility)。
    torch.manual_seed(args.seed) # 设置 CPU 生成随机数的种子。
    torch.cuda.manual_seed_all(args.seed) # 为所有 GPU 设置随机种子。
    torch.cuda.manual_seed(args.seed) # 为当前 GPU 设置随机种子。
    np.random.seed(args.seed) # 设置 NumPy 的随机种子。

    logger = SummaryWriter(log_dir="./log") #初始化 SummaryWriter，用于将日志写入 TensorBoard 指定的目录
    
    #加载数据
    DL=get_feature(args.cancer_type, args.batch_size, True) #调用"Subtype-DCC\dataloader.py"里自定义的get_feature函数，把输入数据封装成数据加载器并返回【【【【【说的对吗？
    
    #初始化模型
    ae = ae.AE() #使用"Subtype-DCC\modules\ae.py"里自定义的AE类，【【【【【干了什么？
    model = network.Network(ae, args.feature_dim, cluster_number) #使用"Subtype-DCC\modules\network.py"里自定义的Network类，【【【【【干了什么？
    model = model.to(device)
    
    #初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) #初始化 Adam 优化器，用于更新模型参数。lr 为学习率，weight_decay 为权重衰减 (L2 正则化)【【【【【L2正则化是什么？

    loss_device = device #设置计算损失的设备
    
    #开始训练循环
    loss=[] #用于记录每个 Epoch 的损失值
    for epoch in range(args.start_epoch, args.epochs+1): # 遍历从开始 Epoch 到结束 Epoch【【【【【？？这个遍历是怎么个事？看了下"Subtype-DCC\config\config.yaml"，这个就是range(0,601)
        lr = optimizer.param_groups[0]["lr"] #获取当前的学习率 (不过lr获取后没有使用)
        loss_epoch = train() #调用前面自定义的train函数，执行一个 Epoch (轮次) 的训练流程。返回当前 Epoch 的总损失值
        loss.append(loss_epoch) #记录本轮损失，将本轮损失添加到列表
        logger.add_scalar("train loss", loss_epoch) # 将本轮损失写入 TensorBoard 日志。
        if epoch % 100 == 0: #每 100 个 Epoch 打印一次日志
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
    #训练结束

    save_model(model_path, model, optimizer, args.epochs) #保存模型的状态字典 (State Dict)、优化器状态、当前 Epoch【【【【【当前Epoch是怎么个事？
    draw_fig(loss,args.cancer_type,epoch) #绘制损失曲线图并保存
    
    #推断阶段
    dataloader=get_feature(args.cancer_type,args.batch_size,False) #调用"Subtype-DCC\dataloader.py"里自定义的get_feature函数，重新获取数据加载器，这次是 False (不打乱数据)，用于按顺序生成特征和预测
    model = network.Network(ae, args.feature_dim, cluster_number) #使用"Subtype-DCC\modules\network.py"里自定义的Network类，重新加载模型架构，重新初始化一个和保存时结构一致的模型结构
    model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.epochs)) #定义模型权重文件的路径
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net']) #加载保存的权重文件，map_location 确保加载到正确的设备，['net'] 提取模型部分的参数
    model.to(device)
    print("### Creating features from model ###")
    X,h = inference(dataloader, model, device) # 调用前面自定义的 inference 函数，得到推理结果，也就是聚类结果 X 和特征结果 h
    
    #保存聚类结果
    output = pd.DataFrame(columns=['sample_name', 'dcc']) #创建一个新的 Pandas DataFrame 用于存储结果
    fea_tmp_file = '../subtype_file/fea/' + args.cancer_type + '/rna.fea' #定义原始特征文件的路径，于是可以读取该文件里的样本名称【【【【【对吗？
    sample_name = list(pd.read_csv(fea_tmp_file).columns)[1:] #读取 CSV 列名，切片 [1:] 去除第一个元素，即索引列，获取样本名称列表
    output['sample_name'] = sample_name #将样本名称填入 DataFrame
    output['dcc'] = X+1 #填入聚类结果，+1 是为了将从 0 开始的索引转换为从 1 开始的类别标签
    out_file = './results/' + args.cancer_type +'.dcc' #定义聚类结果输出文件路径
    output.to_csv(out_file, index=False, sep='\t') # 将 DataFrame 保存为 TSV 文件，不包含索引
    #保存特征结果
    fea_out_file = './results/' + args.cancer_type +'.fea' # 定义特征结果输出文件路径
    fea = pd.DataFrame(data=h, index=sample_name, columns=map(lambda x: 'v' + str(x), range(h.shape[1]))) # 创建包含特征数据的DataFrame，行索引为样本名，列名为 v0, v1, ...
    fea.to_csv(fea_out_file, header=True, index=True, sep='\t') #保存特征文件，包含表头和索引