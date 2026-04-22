from langsmith import Client
from dotenv import load_dotenv
import os

# 加载当前目录下的 .env 文件
load_dotenv()  # 默认查找当前目录的 .env 文件

# 初始化 LangSmith 客户端
client = Client()

# 创建数据集
dataset_name = "qa-dataset"
dataset = client.create_dataset(dataset_name=dataset_name)

# 添加数据点
client.create_examples(
    inputs=[
        {"question": "中国的首都是哪里？"},
        {"question": "1+1等于多少？"}
    ],
    outputs=[
        {"answer": "北京"},
        {"answer": "2"}
    ],
    dataset_id=dataset.id
)