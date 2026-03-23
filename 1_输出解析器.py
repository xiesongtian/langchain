import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_deepseek.chat_models import ChatDeepSeek

# 加载当前目录下的 .env 文件
load_dotenv()  # 默认查找当前目录的 .env 文件

# 从环境变量读取 API Key
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# 可选：检查是否成功获取，如果没有则提示并退出
if not deepseek_api_key:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")

messages = [
    SystemMessage(content="你是一名咨询助理。"),
    HumanMessage(content="请写一段关于华为的简介。")
]

llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    api_key=deepseek_api_key
)

response = llm.invoke(messages)
print(response.content)