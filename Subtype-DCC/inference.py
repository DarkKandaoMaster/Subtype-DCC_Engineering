import torch
import numpy as np
import joblib
import os
from modules import ae, network

# 定义 CancerSubtypePredictor 类
# 这是一个面向对象的封装，旨在将复杂的模型加载、数据预处理和推理逻辑隐藏在一个简单的接口后面
class CancerSubtypePredictor:
    def __init__(self, cancer_type, model_epoch=60, device="cpu"):
        """
        构造函数 (__init__)：在类实例化时自动执行。
        主要负责：
        1. 确定计算设备（CPU/GPU）。
        2. 定位并加载静态资源（归一化器、模型权重）。
        3. 重建模型的网络架构（计算图）。
        Args:
            cancer_type (str): 癌症类型，用于定位特定的模型文件和配置 (如'BRCA'代表乳腺癌的文件路径【【【【【对吗？)
            model_epoch (int): 指定加载哪一个模型权重。就是我们训练完模型之后，不是会把模型保存到本地嘛，于是我们现在就可以加载指定的模型。比如这里填60，就代表加载训练60轮之后的模型【【【【【对吗？
            device (str): 运行设备，默认CPU
        """
        self.device = torch.device(device) #设置device

        base_path = './save' #定义存储模型和辅助文件的根目录路径，对应代码库中的 ./save 文件夹
        scaler_path = os.path.join(base_path, f'{cancer_type}_scaler.pkl') #构建minmaxscaler对象的完整文件路径
        model_path = os.path.join(base_path, cancer_type, f'checkpoint_{model_epoch}.tar') #构建模型权重文件的完整文件路径。 #这些 .tar 文件包含了训练好的神经网络参数（权重和偏置）

        #1.加载minmaxscaler对象
        if not os.path.exists(scaler_path): #检查训练模型时保存在本地的minmaxscaler对象是否存在，如果不存在则抛出文件未找到异常
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        self.scaler=joblib.load(scaler_path) #使用joblib.load反序列化文件，把保存在本地的minmaxscaler对象直接恢复成一个可用的对象
        print(f"[System] Scaler loaded from {scaler_path}") #在控制台打印日志信息，提示minmaxscaler对象已成功加载

        #2. 重建模型架构
        #定义特征维度。这是模型中间层（投影头输出）的向量大小。
        #[cite_start]论文中指出实例级投影头的输出维度设置为 128 [cite: 798]
        feature_dim = 128

        #定义一个字典，映射癌症类型到其对应的亚型（聚类）数量 (K值)。
        #这些配置来源于 train.py 中的 cancer_dict
        cancer_dict = {'BRCA': 5, 'BLCA': 5, 'KIRC': 4, 'LUAD': 3, 'PAAD': 2,
                       'SKCM': 4, 'STAD': 3, 'UCEC': 4, 'UVM': 4, 'GBM': 2}

        cluster_number = cancer_dict.get(cancer_type, 5) #根据输入的 cancer_type 获取对应的聚类数量，如果未找到则默认设为 5

        #实例化自编码器对象。
        #[cite_start]对应论文中的 "Encoder" 部分，它是一个四层深度神经网络，用于从原始高维数据中提取低维特征 [cite: 454, 687]
        self.ae_net = ae.AE()

        #实例化主网络对象。
        #该类将自编码器与对比学习头整合在一起。
        #[cite_start]feature_dim 对应实例级特征，cluster_number 对应聚类级输出维度 [cite: 386]
        self.model = network.Network(self.ae_net, feature_dim, cluster_number)

        #3.加载模型
        #检查模型权重文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        #使用 torch.load 加载权重文件。
        #map_location=self.device 确保权重直接被加载到指定的设备内存中（如 CPU 或 GPU），避免显存错误
        checkpoint = torch.load(model_path, map_location=self.device)

        # 将参数字典填充到模型实例中
        #从加载的 checkpoint 字典中提取 'net' 键对应的值（即网络权重字典 state_dict）。
        #然后使用 load_state_dict 将这些训练好的参数应用到刚刚实例化的 self.model 对象中
        self.model.load_state_dict(checkpoint['net'])

        #将模型移动到self.device
        self.model.to(self.device)

        #4. 设置为评估模式 (Evaluation Mode)（冻结 Dropout 和 BatchNorm 层的更新）
        #调用 .eval() 方法。这会改变模型中特定层（如 Dropout 和 BatchNorm）的行为：
        #- Dropout 层停止随机丢弃神经元。
        #- BatchNorm 层使用训练时统计的全局均值/方差，而不是当前批次的统计量。
        #这是推理阶段必须执行的操作，以保证结果的确定性
        self.model.eval()

        #打印日志信息，确认模型已就绪
        print(f"[System] Model loaded from {model_path}")

    def preprocess(self, raw_data_list):
        """
        数据预处理函数：负责数据拼接、维度对齐和归一化，将输入的原始多组学特征转换为模型可接受的张量格式。
        Args:
            raw_data_list (list): 一个包含四个 NumPy 数组的列表，分别对应四种组学数据：
                                  [Copy Number, DNA Methylation, miRNA, mRNA]。
                                  [cite_start]这是论文提到的 "Early Integration"（早期融合）策略的体现 [cite: 327, 427]。
        Returns:
            torch.Tensor: 预处理完成后的 PyTorch 张量，形状为 (1, 9844)，可以直接送入模型。
                          如果出错则返回 None。
        """
        try:
            #数据拼接
            #使用 numpy.concatenate 将列表中的四个数组沿轴0拼接成一个长向量。拼接后的总维度不出错的话是9844
            combined_feature = np.concatenate(raw_data_list, axis=0)

            #调整形状
            #Scikit-learn的scaler和PyTorch模型期望输入通常是二维矩阵，即(样本数,特征数)。但我们拼接后的combined_feature目前只是一个形状为(9844,)的一维向量。所以我们要调整它的形状
            combined_feature=combined_feature.reshape(1,-1) #此时combined_feature的形状就变成了(1,9844)。也就是说它的batch_size为1
            #这里的 -1 是NumPy的语法糖，意思是自动推断剩余维度的长度（即自动填入9844）

            #使用minmaxscaler对象缩放数据
            #这里必须使用.transform()而不能使用.fit_transform()，否则fit之后就会最大值==最小值==输入数据，然后transform之后输入数据就无效了
            scaled_feature=self.scaler.transform(combined_feature) #此时数据就被缩放到了[0,1]区间

            #将NumPy数组转换为PyTorch张量，同时指定元素的数据类型为32位浮点数，然后移动到self.device
            tensor_data=torch.tensor(scaled_feature,dtype=torch.float32).to(self.device)

            #返回处理好的张量。注意这里返回的是张量而不是数据加载器DataLoader对象，原因：
            #1.DataLoader的核心功能是Batch切分和Shuffle，而我们在调整形状时已经通过reshape(1,-1)手动完成了Batch构造，因此不需要DataLoader
            #2.对于单条数据的API推理，相较于直接使用张量，DataLoader开销大、耗时长
            return tensor_data

        except Exception as e:
            print(f"[Error] Preprocessing failed: {e}")
            return None

    def predict(self, raw_data_list):
        """
        推理主方法：执行模型的前向传播以获取预测结果。
        
        Args:
            raw_data_list (list): 包含四种组学特征的列表。
            
        Returns:
            int: 预测出的癌症亚型类别ID。
        """

        #我们刚才不是自定义了一个preprocess方法嘛，现在我们来调用它
        tensor_data = self.preprocess(raw_data_list)

        # 如果预处理失败（返回 None），则直接返回错误代码 -1
        if tensor_data is None:
            return -1

        #关闭梯度计算
        with torch.no_grad():
            c, h = self.model.forward_cluster(tensor_data) #调用 Network 类的 forward_cluster 方法，也就是那个只在推理时使用的前向传播逻辑 #这个forward_cluster是论文代码里原有的函数

        # .item() 将单元素 Tensor 转换为 Python 标准数据类型 (int/float)
        # 提取预测结果。
        # c 是一个包含单个值的 PyTorch 张量。
        # .item() 方法将这个张量中的数值提取出来，转换为标准的 Python 标量 (如 int)，方便后续处理或 JSON 序列化。
        return c.item()

if __name__ == "__main__": #这是 Python 的标准入口判断。只有当这个文件被直接运行（而不是作为模块被导入）时，下面的代码才会执行
    np.random.seed(42) #设置随机种子
    print("Initializing Inference Engine...") #打印提示信息

    try:
        # 实例化预测器对象 (这里直接假设使用 BRCA 数据集，模型为第 60 轮的模型。因为这些代码的目的是测试能不能正常运行)
        predictor = CancerSubtypePredictor(cancer_type='BRCA', model_epoch=60)

        # 模拟生成一条病人的多组学数据
        # 维度必须与训练数据（也就是论文里提到的数据）严格一致：
        # CN（Copy Number）: 3105, Meth（DNA Methylation）: 3139, miRNA: 383, RNA（mRNA）: 3217 -> 总和: 9844。否则在 reshape 或网络输入层会报错。
        fake_cn = np.random.rand(3105) #使用 np.random.rand 生成 [0, 1) 之间的随机浮点数。
        fake_meth = np.random.rand(3139)
        fake_mirna = np.random.rand(383)
        fake_rna = np.random.rand(3217)

        data_packet = [fake_cn, fake_meth, fake_mirna, fake_rna] #将生成的四个特征向量打包成列表

        print("Running Prediction...") #打印提示信息
        result = predictor.predict(data_packet) #调用 predict 方法进行推理，获取预测的亚型 ID
        print("="*30)
        print(f"Prediction Result: Subtype {result}") #打印最终的预测结果，显示亚型 ID
        print("="*30)

    except Exception as e:
        print(f"Test Failed: {e}") #捕获整个测试过程中的任何异常，打印错误堆栈
        print("Tip: Ensure you have run 'python train.py -c BRCA --epochs 60' first.") #给出提示建议，因为常见的错误通常是缺少训练好的模型文件