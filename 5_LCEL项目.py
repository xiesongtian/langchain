from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.chains import LLMChain
from langserve import add_routes
from dotenv import load_dotenv
import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""
# os.environ["LANGCHAIN_PROJECT"] = "my-llm-project"

# 加载当前目录下的 .env 文件
load_dotenv()  # 默认查找当前目录的 .env 文件

# 从环境变量读取 API Key
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# 2. Create model
model = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    api_key=deepseek_api_key
)

prompt = PromptTemplate.from_template("翻译这句话到英文：{sentence}")
chain = LLMChain(llm=model, prompt=prompt)

# 运行 Chain，LangSmith 会自动记录
result = chain.invoke({"sentence": "今天是星期五"})
print(result)