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

#对模型进行测试，返回推理结果
# loader: 数据加载器，用于批量提供测试数据。
# model: 训练好的神经网络模型。
def inference(loader, model, device):
    model.eval() #将模型设置为评估模式
    cluster_vector = [] #初始化列表，用于存储所有样本的聚类预测结果
    feature_vector = [] #初始化列表，用于存储所有样本的潜在特征表示
    for step, x in enumerate(loader): #遍历数据加载器loader。step是索引（表示当前是第几个Batch），x是测试用的张量
        x = x.float().to(device) # 将输入数据转换为32位浮点型张量，并移动到指定的计算设备 (GPU/CPU) 上
        with torch.no_grad(): #关闭梯度计算。因为推断阶段不需要反向传播
            c,h = model.forward_cluster(x) #调用network.py里自定义的前向传播方法，将x传入模型，获取聚类分配概率 c 和特征向量 h
        c = c.detach() #将张量c从计算图中分离，使其不再具有梯度历史。PyTorch规定，带有梯度历史（requires_grad=True）的Tensor是不能直接转换为NumPy数组的，所以必须先detach()
        h = h.detach() #将张量h从计算图中分离
        cluster_vector.extend(c.cpu().detach().numpy()) # 将聚类结果移动到 CPU，转换为 NumPy 数组，并添加到结果列表中
        feature_vector.extend(h.cpu().detach().numpy()) # 将特征结果移动到 CPU，转换为 NumPy 数组，并添加到结果列表中
    cluster_vector = np.array(cluster_vector) # 将聚类结果列表转换为 NumPy 数组
    feature_vector = np.array(feature_vector) # 将特征结果列表转换为 NumPy 数组
    print("Features shape {}".format(feature_vector.shape)) # 打印特征矩阵的形状 (样本数, 特征维度)，用于检查数据维度是否符合预期
    return cluster_vector,feature_vector

#对模型训练一个Epoch，返回当前Epoch的总损失
def train():
    loss_epoch = 0 #初始化累加器，用于记录当前Epoch的总损失
    for step,x in enumerate(DL): #遍历训练数据加载器DL（全局变量）。step是索引（表示当前是第几个Batch），x是训练用的张量。因为本项目是无监督学习，所以只有训练用的张量，没有标签。所以和MNIST不同，这里写的是 索引,训练用的张量 而不是 训练用的张量,对应的标签 
        optimizer.zero_grad() #梯度清零
        # 数据增强：
        # 向原始数据 x 添加标准正态分布的噪声（高斯噪声）（均值0、标准差1），生成两个增强视图 x_i 和 x_j。这是对比学习常用的构建正样本对的方法
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
        # 总损失：将实例损失和聚类损失相加，用于联合优化
        loss = loss_instance + loss_cluster
        loss.backward() #反向传播
        optimizer.step() #更新model中所有需要学习的参数
        loss_epoch += loss.item() #累加损失
    return loss_epoch

# 绘制训练损失随 Epoch 变化的曲线图并保存
def draw_fig(list,name,epoch):
    x1 = range(0, epoch+1) # 创建 x 轴数据，表示 Epoch 的序列，从 0 到当前 epoch
    print(x1) # 打印 x 轴序列
    y1 = list # y 轴数据为传入的损失值列表
    save_file = './results/' + name + 'Train_loss.png' # 定义图片保存路径
    plt.cla() # 清除当前活动的轴，防止重叠绘图
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
    parser.add_argument("--cancer_type", '-c', type=str, default="BRCA") #癌症类型，默认为BRCA #这个模型并不是用来判断“这个人得的是不是乳腺癌（BRCA）”（这是癌症诊断/分类），而是用来判断“这个已经确诊乳腺癌的病人，具体属于哪一种乳腺癌”（这是癌症亚型鉴定）。
    parser.add_argument('--batch_size', type=int, default=64) # 批次大小，默认为64
    parser.add_argument('--cluster_number', type=int,default=5) # 聚类数量，默认为5
    args=parser.parse_args() #把用户在cmd里输入的内容，按照上面三行定义，转换一下，并把结果保存在一个类似于QuickSay的全局对象config的对象args
    #定义癌症类型到聚类数量的映射字典
    cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4,
                   'LUAD': 3, 'PAAD': 2, 'SKCM': 4,
                   'STAD': 3, 'UCEC': 4, 'UVM': 4, 'GBM': 2}
    
    cluster_number = cancer_dict[args.cancer_type] #按照癌症种类选择。根据输入的癌症类型，从字典中获取对应的预设聚类数目
    print(cluster_number) #打印聚类数目以确认

    config=yaml_config_hook("config/config.yaml") #加载YAML配置文件，yaml_config_hook是“Subtype-DCC\utils\yaml_config_hook.py”里的自定义函数
    for k, v in config.items(): #遍历配置文件中的键值对
        parser.add_argument(f"--{k}", default=v, type=type(v)) # 将配置文件中的参数动态也添加到argparse
    args=parser.parse_args() #重新解析参数，于是这样就能实现把命令行参数和配置文件参数都保存在args对象
    model_path = './save/' + args.cancer_type #模型保存路径
    if not os.path.exists(model_path): #如果路径不存在，则创建该目录
        os.makedirs(model_path)

    #设置随机种子
    torch.manual_seed(args.seed) #设置CPU生成随机数的种子
    torch.cuda.manual_seed_all(args.seed) #为所有GPU设置随机种子
    torch.cuda.manual_seed(args.seed) #为当前GPU设置随机种子
    np.random.seed(args.seed) #设置NumPy的随机种子

    logger=SummaryWriter(log_dir="./log") #初始化SummaryWriter，它会在"./log"文件夹中创建特殊的事件文件，用于存储日志

    #加载数据
    DL=get_feature(args.cancer_type,args.batch_size,True) #调用"Subtype-DCC\dataloader.py"里自定义的get_feature函数，把对应癌症类型的训练数据封装成数据加载器并返回

    #初始化模型
    ae=ae.AE() #使用"Subtype-DCC\modules\ae.py"里自定义的AE类创建一个对象
    model=network.Network(ae,args.feature_dim,cluster_number) #使用"Subtype-DCC\modules\network.py"里自定义的Network类创建一个对象
    model=model.to(device)

    #初始化优化器
    optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) #初始化Adam优化器，用于更新模型参数。lr为学习率，weight_decay为权重衰减

    loss_device=device #设置计算损失的设备

    #开始训练循环
    loss=[] #用于记录每个Epoch的总损失
    for epoch in range(args.start_epoch,args.epochs+1): #训练 args.epochs+1 - args.start_epoch 轮次。如果你没有在“Subtype-DCC\config\config.yaml”修改，那么这里就是range(0,601)
        lr=optimizer.param_groups[0]["lr"] #获取当前的学习率（不过lr获取后没有使用，所以这句代码完全可以注释掉）
        loss_epoch=train() #调用我们刚才自定义的train函数，对模型训练一个Epoch，返回当前Epoch的总损失
        loss.append(loss_epoch) #将当前Epoch的总损失添加到列表loss
        logger.add_scalar("train loss",loss_epoch) #将当前Epoch的总损失写入TensorBoard日志
        if epoch%100==0: #每100个Epoch在控制台打印一次提示
            print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
    #训练结束

    save_model(model_path, model, optimizer, args.epochs) #保存模型参数、优化器状态、当前Epoch
    draw_fig(loss,args.cancer_type,epoch) #绘制损失曲线图并保存

    #推断阶段
    dataloader=get_feature(args.cancer_type,args.batch_size,False) #调用"Subtype-DCC\dataloader.py"里自定义的get_feature函数，重新获取数据加载器，这次是 False（不打乱数据）。所以本项目训练数据测试数据用的是同一份
    model = network.Network(ae, args.feature_dim, cluster_number) #使用"Subtype-DCC\modules\network.py"里自定义的Network类，重新加载模型架构，重新初始化一个和保存时结构一致的模型结构
    model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.epochs)) #定义模型权重文件的路径
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net']) #加载保存的权重文件，map_location 确保加载到正确的设备，['net'] 提取模型部分的参数
    model.to(device)
    print("### Creating features from model ###")
    X,h = inference(dataloader, model, device) # 调用前面自定义的 inference 函数，得到推理结果，也就是聚类结果 X 和特征结果 h
    
    #保存聚类结果
    output = pd.DataFrame(columns=['sample_name', 'dcc']) #创建一个新的 Pandas DataFrame 用于存储结果
    fea_tmp_file = '../subtype_file/fea/' + args.cancer_type + '/rna.fea' #定义训练数据文件的路径 #为什么这里只选了rna而没有四个全选？因为这四个文件的列名都是一样，随便选一个抄一下样本名称就行了
    sample_name = list(pd.read_csv(fea_tmp_file).columns)[1:] #读取 CSV 列名，切片 [1:] 去除第一个元素，即索引列，获取样本名称列表
    output['sample_name'] = sample_name #将样本名称填入 DataFrame
    output['dcc'] = X+1 #填入聚类结果，+1 是为了将从 0 开始的索引转换为从 1 开始的类别标签
    out_file = './results/' + args.cancer_type +'.dcc' #定义聚类结果输出文件路径
    output.to_csv(out_file, index=False, sep='\t') # 将 DataFrame 保存为 TSV 文件，不包含索引
    #保存特征结果
    fea_out_file = './results/' + args.cancer_type +'.fea' # 定义特征结果输出文件路径
    fea = pd.DataFrame(data=h, index=sample_name, columns=map(lambda x: 'v' + str(x), range(h.shape[1]))) # 创建包含特征数据的DataFrame，行索引为样本名，列名为 v0, v1, ...
    fea.to_csv(fea_out_file, header=True, index=True, sep='\t') #保存特征文件，包含表头和索引