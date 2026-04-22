import yaml
from langchain_community.agent_toolkits import JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec
from langchain_deepseek import ChatDeepSeek  # ✅ 换成 DeepSeek
from dotenv import load_dotenv
load_dotenv()
import os

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)

with open("openai_openapi.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

json_agent_executor = create_json_agent(
    llm=llm, toolkit=json_toolkit, verbose=True, handle_parsing_errors=True
)

output= json_agent_executor.run(
    "What are the required parameters in the request body to the /completions endpoint?"
)
print(output)