from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
import os

load_dotenv()

chat = AzureChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0.7,
)

response = chat([
    HumanMessage(content="Hello! What can you do?")
])

print(response.content)




chat2 = AzureChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0.7,
)

response2 = chat2.invoke([
    HumanMessage(content="Hey GPT-4o! What can you do?")
])

print(response2.content)
