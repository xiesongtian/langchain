from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from dotenv import load_dotenv
import requests
import os

load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


# ✅ 自定义图片生成工具，调用 Pollinations.ai（免费，无需 Key）
def generate_image(prompt: str) -> str:
    # 把空格换成短横线，拼接成 URL
    encoded_prompt = prompt.replace(" ", "-")
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    # 验证图片是否生成成功
    response = requests.get(image_url)
    if response.status_code == 200:
        # 保存到本地
        with open("output.jpg", "wb") as f:
            f.write(response.content)
        return f"图片已生成并保存为 output.jpg，在线地址：{image_url}"
    else:
        return "图片生成失败，请重试"


image_tool = Tool(
    name="ImageGenerator",
    func=generate_image,
    description="根据英文描述生成图片，输入必须是英文 prompt"
)

agent = initialize_agent(
    [image_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

prompt = """
请根据以下中文描述生成一张图片，先将描述翻译成英文，再调用 ImageGenerator 工具：

暮色将至，山脊上最后一缕阳光把云层烧成了橘红色。远处的山峦一层叠着一层，近的墨绿，远的靛蓝，最远的几乎与天空融为一体，分不清边界。
山脚下有一条河，水流不急，透过水面能看见河底的卵石，每一块都被打磨得圆润光滑。偶尔有风吹过，水面漾起细碎的波纹，把倒映其中的山影揉碎，又缓缓拼回原样。
林子里很安静。松树的气息混着潮湿的泥土味，一阵一阵地涌过来。脚下的落叶厚厚地铺了一层，踩上去软而无声，像是整片森林都屏住了呼吸。
天色再暗一些，第一颗星出现在东边的天空，孤零零的，却格外亮。
"""

output = agent.run(prompt)
print(output)