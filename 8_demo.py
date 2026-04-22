from langchain_openai import ChatOpenAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from dotenv import load_dotenv
import os

load_dotenv()

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# 定义llm
llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    api_key=deepseek_api_key
)

tools = load_tools(["serpapi","llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

print(agent.run("请问今天日历?历史上今天的大事件都有什么"))