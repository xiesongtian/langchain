import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate


from langchain_core.output_parsers import StrOutputParser

# 加载当前目录下的 .env 文件
load_dotenv()  # 默认查找当前目录的 .env 文件

# 从环境变量读取 API Key
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

model = ChatDeepSeek(
    api_key=deepseek_api_key,
    model="deepseek-chat",  # 或其他支持的模型名称
    temperature=0
)

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

parser = StrOutputParser()


chain = prompt_template | model | parser

result = chain.invoke({"language": "china", "text": "i will be rich man"})

print(result)


