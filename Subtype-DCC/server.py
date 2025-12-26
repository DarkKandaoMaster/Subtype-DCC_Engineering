import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
import pydantic
from typing import List
from inference import CancerSubtypePredictor
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

class PatientRecord(pydantic.BaseModel): #定义一个类，在这个类里声明四个变量，并且明确指定这四个变量的类型。以此实现确保输入数据符合类型要求（浮点数列表），否则就像C语言那样报错
    #这里的冒号使用的是 Python 的类型提示语法。 变量名: 类型 意思是声明一个名为 变量名 的变量，它的预期类型是 类型 
    #虽然在普通 Python 代码中类型提示通常只是像注释一样给人看的，但在 pydantic.BaseModel 中，冒号具有强制性。Pydantic 库会读取冒号后面的类型，并像C语言那样执行强制类型转换和验证
    cn_features: List[float] #对应 Copy Number Variation (拷贝数变异) 特征向量
    meth_features: List[float] #对应 DNA Methylation (DNA 甲基化) 特征向量
    mirna_features: List[float] #对应 miRNA (微小核糖核酸) 表达量特征向量
    rna_features: List[float] #对应 mRNA (信使核糖核酸) 基因表达量特征向量

#实例化 FastAPI 类，创建一个 Web 应用程序对象
app=FastAPI(
    title="强壮的Cancer Subtype Diagnostic API", #设置 API 文档的标题，方便前端开发者查看
    description="强壮的Based on Subtype-DCC Deep Learning Model", #设置 API 的描述信息
    version="1.0.0" #设置版本号，用于接口版本管理
)
#配置CORS(跨域资源共享)，允许前端页面访问此API。如果不加这段代码，浏览器会拦截前端对 8000 端口的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #允许所有来源访问（实际生产中应限制为特定域名）
    allow_credentials=True, #允许携带Cookie等凭证
    allow_methods=["*"], #允许所有HTTP方法（POST、GET等）
    allow_headers=["*"], #允许所有HTTP请求头
)

#一个细节：在服务启动时（server.py运行时）加载模型，而不是在每次请求时加载。这样可以避免重复的磁盘 I/O 和 GPU/CPU 内存分配
print("[Server] Initializing Global Model...") #在控制台打印日志，表明模型正在加载
try:
    GLOBAL_PREDICTOR=CancerSubtypePredictor(cancer_type='BRCA',model_epoch=600) #使用我们在inference.py里定义的类创建一个全局对象。这里我们使用的是第600轮的BRCA模型【【【【【600轮？这里记得改一下
    print("[Server] Model loaded successfully!")
except Exception:
    GLOBAL_PREDICTOR=None
    print(f"[Server] Critical Error: Failed to load model. {Exception}")

#@app.post 是一个装饰器 (Decorator)。它的作用是将下面的 python 函数“注册”到 Web 服务器的路由表中。当用户以 POST 方法访问 "/predict" 这个网址时，服务器会自动调用下面的 predict_subtype 函数来处理请求。
@app.post("/predict")
def predict_subtype(record: PatientRecord): #FastAPI 会自动读取 HTTP 请求体中的 JSON 数据，并将其作为参数传递给 record，实现接收输入数据 #参数record的类型是PatientRecord，就是我们刚才定义的那个类
    if GLOBAL_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model service unavailable") #主动抛出一个 HTTP 异常。这相当于在 C 语言中 return error_code，告诉客户端“服务器暂时不可用”(状态码 503)。
    try:
        #首先把record里的数据挨个转换为numpy数组，然后用列表套起来
        input_packet=[
            np.array(record.cn_features),
            np.array(record.meth_features),
            np.array(record.mirna_features),
            np.array(record.rna_features)
        ]

        result_subtype=GLOBAL_PREDICTOR.predict(input_packet) #调用GLOBAL_PREDICTOR对象的predict方法，于是就得到模型推理结果了

        if result_subtype==-1: #如果predict为-1，说明预处理失败
            raise HTTPException(status_code=400, detail="Data Preprocessing Failed. Check input dimensions.") #返回 400 Bad Request 错误，告诉客户端“你发的数据格式不对，我处理不了”

        # 函数直接返回一个 Python 字典
        # FastAPI 框架会自动将这个字典序列化成 JSON 字符串，并通过网络发送给客户端。
        # 客户端收到的就是形如 {"status": "success", ...} 的 JSON 数据。
        return {
            "status": "success",
            "model_version": "BRCA_v1",
            "prediction": {
                "subtype_id": result_subtype,
                "description": f"Predicted Cancer Subtype {result_subtype}"
            }
        }

    except Exception: #防止代码中出现未预料的错误导致服务器崩溃
        raise HTTPException(status_code=500, detail=str(Exception)) #将错误信息转为字符串并封装在 500 错误中返回，方便调试

@app.post("/predict_file") #该接口作用：接收CSV文件，自动解析并调用模型
async def predict_file(file: UploadFile = File(...)): #使用UploadFil类型接收文件流
    if GLOBAL_PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model service unavailable")
    #检查文件扩展名是不是.csv就不用了

    try:
        # 1. 读取文件内容
        contents = await file.read() #await file.read()意思是异步读取上传文件的全部二进制内容

        # 2. 将二进制内容转换为字符串，并使用 pandas 读取 CSV
        # io.StringIO 将字符串包装成类似文件的对象，以便 pd.read_csv 读取
        # header=None 假设 CSV 没有表头，只有一行数据。如果用户上传的有表头，这里需要调整
        # --- 2. 解析 CSV ---
        # io.BytesIO(contents): 将二进制数据包装成内存中的文件流对象，让 pandas 以为它在读一个本地文件
        # header=None: 假设 CSV 没有表头，直接读数据。如果有表头请改为 header=0
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), header=None)

        # 3. 检查数据维度
        # 根据 dataloader.py 和 inference.py，模型输入的总特征数为 9844
        # 具体组成: CN(3105) + Meth(3139) + miRNA(383) + mRNA(3217) = 9844
        if df.shape[1] != 9844:
            raise HTTPException(status_code=400, 
                                detail=f"Dimension Error: Expected 9844 columns, but got {df.shape[1]}.")

        # 4. 提取第一行数据并转换为 numpy 数组
        # .iloc[0] 获取第一行，.values 获取其数值数组
        # 取第一行数据作为当前病人的特征 (假设用户只传了一个病人的数据)
        # .values 属性将 pandas Series 转换为 numpy 数组
        full_features = df.iloc[0].values 

        # # 简单校验维度 (BRCA 数据集总共有 9844 个特征)
        # total_dims = 9844
        # if len(patient_data) != total_dims:
        #     raise HTTPException(status_code=400, detail=f"Expected {total_dims} features, but got {len(patient_data)}")


# --- 3. 数据切片 (Data Slicing) ---
        # 根据 inference.py 中的维度定义，将长向量切分为 4 个组学特征向量
        # CN: 3105, Meth: 3139, miRNA: 383, RNA: 3217
        # 5. 切分数据 (Data Slicing)
        # 按照 dataloader.py 中 concatenate 的顺序进行逆向切分
        # 必须严格遵守该顺序，否则特征含义错位会导致预测完全错误
        
        idx1 = 3105
        idx2 = idx1 + 3139
        idx3 = idx2 + 383
        cn_features = full_features[0:idx1]
        meth_features = full_features[idx1:idx2]
        mirna_features = full_features[idx2:idx3]
        rna_features = full_features[idx3:]

# 组装输入包
        # 6. 构造 input_packet
        # 这是一个列表，符合 inference.py 中 predict 方法的输入要求
        input_packet = [cn_features, meth_features, mirna_features, rna_features]

        # 7. 调用模型预测
        result_subtype = GLOBAL_PREDICTOR.predict(input_packet)

        if result_subtype == -1:
            raise HTTPException(status_code=400, detail="Prediction failed during preprocessing.")

        # 8. 返回结果 (与 /predict 接口保持一致)
        return {
            "status": "success",
            "model_version": "BRCA_v1",
            "prediction": {
                "subtype_id": result_subtype,
                "description": f"Predicted Cancer Subtype {result_subtype}"
            }
        }

    except Exception as e:
        # 捕获我们自己抛出的 HTTP 异常、 Pandas 解析错误或其他未知错误
        raise HTTPException(status_code=500, detail=f"Processing Error: {str(e)}")
# ===

if __name__ == "__main__": #这是 Python 的标准入口判断。只有当这个文件被直接运行（而不是作为模块被导入）时，下面的代码才会执行
    #启动 Uvicorn 服务器
    #我们刚才不是实例化了一个app对象嘛，现在我们将app作为参数传给Uvicorn服务器，于是当服务器收到请求时，可以找到并调用对应的@app.post("/predict")修饰的函数
    uvicorn.run(app, host="0.0.0.0", port=8000) #这句代码的意思就是让Uvicorn服务器加载app这个对象，并且在所有网卡（0.0.0.0）上监听 8000 端口，准备随时接收请求
    #host="0.0.0.0"对应底层 Socket 编程中的 INADDR_ANY 宏，意思是监听本机“所有”网卡接口。也就是说允许外部网络访问本服务
    #一旦执行这句代码，主线程将进入一个无限循环，持续挂起以监听网络端口。也就是说这之后的代码都执行不了了，除非进程被信号终止。
    # http://127.0.0.1:8000/redoc【【【【【
    # http://127.0.0.1:8000/doc【【【【【
    # http://127.0.0.1:8000/docs【【【【【