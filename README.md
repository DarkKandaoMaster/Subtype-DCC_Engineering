# Subtype-DCC_Engineering
### 我把这个论文模型 https://github.com/zhaojingo/Subtype-DCC 工程化落地了！


## 相较于原代码，我改动了什么？
- 增加了大量注释！于是代码更适合学习和阅读。如果您也要学习Subtype-DCC这篇论文，那么可以来我这个仓库！因为注释实在是太详细了，应该能给您节省很多时间  
- 在dataloader.py增加了几句代码，实现：训练模型时，也将minmaxscaler对象序列化到磁盘  
- 新增inference.py。实现：原代码只能在给定的训练集测试集上运行，现在支持使用保存在本地的模型，对单个病人数据进行预测  
- 新增server.py。也就是说用FastAPI写了套接口，实现：运行server.py后，既可以双击index.html，上传test_patient.csv，得到结果；也可以运行test_request.py，得到结果  
- index.html、test_patient.csv、test_request.py也是我新增的。做得比较潦草，能用就行  


## 如何下载代码并使用
1. 下载我这个项目的代码  
2. 配置环境略。有报错就浏览器搜索，或者问AI，然后安装对应环境就行了  
3. 下载 https://github.com/haiyang1986/Subtype-GAN 这个项目的代码，里面的fea文件夹就是训练模型时要用到的训练数据  
4. 在我这个项目的根目录，新建一个名称为subtype_file的文件夹，把fea文件夹复制进去  
5. 打开命令提示符（cmd），cd到Subtype-DCC这个文件夹  
6. 激活你配置好的那个虚拟环境（如果你刚才配置的是虚拟环境的话）（比如我就是`conda activate myenv`）  
7. 输入`python train.py`，等待运行完毕。于是成功生成模型、minmaxscaler对象  
8. 输入`python server.py`，不要关闭这个命令提示符窗口，于是就可以双击index.html，上传test_patient.csv，得到结果；也可以运行test_request.py，得到结果  


## 常见问题
- index.html会报2个红色警告，这是正常的。它俩一个警告网页在手机上会很难用，一个警告对视障用户不友好  
- test_patient.csv这里面的数据是我随机生成的，如果想测试其他数据的话可以写个代码把训练集里的数据提取一份出来。注意训练集里的数据是一列一列的，但是server.py期望的csv文件是一行一行的  
- 有不懂就浏览器搜索，或者问AI，或者发议题（issue）问我，我肯定会回复的  


https://github.com/user-attachments/assets/a312541e-9598-4c5f-bacb-3a2087be783f  