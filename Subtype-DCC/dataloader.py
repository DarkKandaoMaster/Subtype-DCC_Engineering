import torch
import numpy as np
import pandas as pd
from os.path import splitext, basename
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import joblib
import os

# 定义特征获取函数
# cancer_type: 癌症类型字符串，用于定位文件路径
# batch_size: 批处理大小，决定每次迭代送入模型的样本数量
# training: 布尔值，指示当前是否为训练模式（影响数据加载时的 shuffle 行为）
def get_feature(cancer_type, batch_size, training):
    fea_CN_file = '../subtype_file/fea/' + cancer_type + '/CN.fea' #拷贝数变异 (Copy Number, CN) 特征文件的相对路径。
    fea_CN = pd.read_csv(fea_CN_file, header=0, index_col=0, sep=',') #使用 Pandas 读取 CSV 格式的特征文件。
    # header=0: 指定第一行作为列名（Column Names）。
    # index_col=0: 指定第一列作为行索引（Row Index，通常是基因名称或特征ID）。
    # sep=',': 指定文件分隔符为逗号。

    fea_meth_file = '../subtype_file/fea/' + cancer_type + '/meth.fea' #DNA 甲基化 (DNA Methylation) 特征文件的相对路径。
    fea_meth = pd.read_csv(fea_meth_file, header=0, index_col=0, sep=',')

    fea_mirna_file = '../subtype_file/fea/' + cancer_type + '/miRNA.fea' #miRNA (微小核糖核酸) 特征文件的相对路径。
    fea_mirna = pd.read_csv(fea_mirna_file, header=0, index_col=0, sep=',')

    fea_rna_file = '../subtype_file/fea/' + cancer_type + '/rna.fea' #mRNA (信使核糖核酸，即基因表达量) 特征文件的相对路径。
    fea_rna = pd.read_csv(fea_rna_file, header=0, index_col=0, sep=',')

    #此时我们就读取到了上面4个数据，也就是论文图1的那4个输入数据
    #接下来就是把它们concat起来
    feature = np.concatenate((fea_CN, fea_meth, fea_mirna, fea_rna), axis=0).T #拼接逻辑：它把 4 个 CSV 文件读进来，然后暴力拼接在一起。
    # np.concatenate: 将四个 DataFrame（自动转换为 NumPy 数组）沿 axis=0（行方向）进行拼接。
    # 这里的假设是：原始 CSV 文件中，“行”代表特征（如基因），“列”代表样本。因此 axis=0 拼接增加了特征的数量（Feature Stacking）。
    # axis=0 表示在垂直方向上堆叠数据，即将不同组学的特征在行方向上合并，样本数（列数）保持不变。
    # .T 执行转置操作 (Transpose)。将矩阵维度从 (总特征数, 样本数) 转换为 (样本数, 总特征数)。这是因为机器学习模型通常要求输入数据的形状为 (N_samples, N_features)。

    minmaxscaler=MinMaxScaler() #初始化最大最小归一化器，用于将特征缩放到[0,1]区间
    feature=minmaxscaler.fit_transform(feature)
    # fit_transform: 对特征矩阵先后执行 fit 和 transform 操作
    # fit: 计算每一列特征的统计量（最小值 Min 和 最大值 Max）。
    # transform: 根据计算出的 Min 和 Max，应用公式 (x - Min) / (Max - Min) 对数据进行归一化。
    # 注意：在工程实践中，通常应在训练集上 fit，在验证/测试集上仅 transform，此处代码对全部数据重新 fit 存在数据泄露或分布不一致风险。
    #我们更应该在训练集上得到最小值Min和最大值Max之后，就把它保存下来；测试、实际使用模型时就直接使用这个保存下来的最小值Min和最大值Max，而不是临时计算最小值Min和最大值Max。否则，比如实际使用模型时，fit之后就会最大值==最小值==输入数据，然后transform之后输入数据就无效了。所以这个最小值Min和最大值Max也是模型训练的一部分，得和模型一样保存下来

    #下面这步就是保存最小值Min和最大值Max
    if training:
        save_dir='./save'
        if not os.path.exists(save_dir): #检查该目录是否存在，如果不存在则调用makedirs创建
            os.makedirs(save_dir)
        scaler_path=os.path.join(save_dir, f'{cancer_type}_scaler.pkl') #这个就是保存下来的文件的路径。文件名例如BRCA_scaler.pkl
        joblib.dump(minmaxscaler,scaler_path) #调用joblib.dump将minmaxscaler对象序列化到磁盘
        #为什么存的是.pkl（二进制）而不是.txt/.json（文本）？
        #1.它存的不是2个数字，而是2万个数字（9844+9844），保存时文件会很大，保存速度也很慢
        #2.文本格式会“丢精度”，比如1/3，写入文本时，可能被迫存为0.333333
        #3.方便。这样可以把整个最大最小归一化器对象保存下来，需要用到时再直接恢复成一个可用的对象，然后直接使用，而不需要再初始化一个最大最小归一化器对象并赋值
        print(f"Pre-fitted scaler saved to {scaler_path}") #在控制台打印日志信息，提示minmaxscaler对象已成功保存

    feature = torch.tensor(feature)
    # 将 NumPy 数组转换为 PyTorch 的张量 (Tensor)。也就是把numpy.ndarray 转换为 torch.Tensor
    
    dataloader = DataLoader(feature, batch_size=batch_size, shuffle=training)
    # 构建数据加载器 DataLoader 对象。
    # feature: 数据集张量。
    # batch_size=batch_size: 每个批次包含的样本数（形参）
    # shuffle=training: 是否打乱数据（形参）。训练时通常设为 True 以打破样本相关性，提高模型泛化能力；推理时通常设为 False。

    return dataloader #返回封装好的 DataLoader 对象