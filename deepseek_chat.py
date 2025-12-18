from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

chat = AzureChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    azure_endpoint=os.getenv("DEEPSEEK_ENDPOINT"),
    deployment_name=os.getenv("DEEPSEEK_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("DEEPSEEK_API_VERSION"),
    temperature=0.7,
)

response = chat.invoke([
    HumanMessage(content="Hi DeepSeek! What's your specialty?")
])

print(response.content)
