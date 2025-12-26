# Subtype-DCC_Engineering
### 我把这个论文模型 https://github.com/zhaojingo/Subtype-DCC 工程化落地了！


## 相较于原代码，我改动了什么？
- 增加了大量注释！于是代码更适合学习和阅读  
- 在dataloader.py增加了几句代码，实现：训练模型时，也将minmaxscaler对象序列化到磁盘  
- 新增inference.py。实现：原代码只能在给定的训练集测试集上运行，现在支持使用保存在本地的模型，对单个病人数据进行预测  
- 新增server.py。也就是说用FastAPI写了套接口，实现：运行server.py后，既可以双击index.html，上传test_patient.csv，得到结果；也可以运行test_request.py，得到结果  


## 如何下载代码并使用
1. 配置环境步骤略，有报错就浏览器搜索，或者问AI，然后安装对应环境就行了  
2. 下载测试集训练集数据【【【【【这里记得补充说几句  
3. 打开命令提示符（cmd），cd到Subtype-DCC这个文件夹  
4. 激活你配置好的那个虚拟环境（如果你刚才配置的是虚拟环境的话）（比如我就是`conda activate myenv`）  
5. 首先生成模型、minmaxscaler对象：输入`python train.py`，等待运行完毕  
6. 输入`python server.py`，不要关闭这个命令提示符窗口，于是就可以双击index.html，上传test_patient.csv，得到结果；也可以运行test_request.py，得到结果  

https://github.com/user-attachments/assets/a312541e-9598-4c5f-bacb-3a2087be783f