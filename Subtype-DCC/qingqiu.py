# client_test.py (新建这个文件用来模拟前端)
import requests
import numpy as np

# 造假数据 (维度必须正确)
fake_data = {
    "cn_features": np.random.rand(3105).tolist(),
    "meth_features": np.random.rand(3139).tolist(),
    "mirna_features": np.random.rand(383).tolist(),
    "rna_features": np.random.rand(3217).tolist()
}

print("正在发送请求给服务器...")
try:
    # 发送 POST 请求
    response = requests.post("http://127.0.0.1:8000/predict", json=fake_data)
    
    # 打印结果
    print("状态码:", response.status_code)
    print("返回结果:", response.json())
except Exception as e:
    print("请求失败:", e)