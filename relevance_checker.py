# relevance_checker.py
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def check_relevance(answer: str, question: str, job_title: str) -> str:
    prompt = f"""
You are a strict interviewer for the {job_title} role.
Classify the candidate’s answer as: relevant, partially relevant, irrelevant, misbehave, time wasting, or nonsense.
Answer with exactly one word.
Question: "{question}"
Answer: "{answer}"
"""
    chat = AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        temperature=0.0,
    )
    resp = chat.invoke([HumanMessage(content=prompt)])
    return resp.content.strip().lower()