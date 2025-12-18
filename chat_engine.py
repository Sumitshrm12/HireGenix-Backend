# chat_engine.py
import os
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def generate_question(
    history: List[dict], job_title: str, total_q: int
) -> str:
    """Based on conversation history, return the next question or completion message."""
    qn = sum(1 for m in history if m["role"]=="assistant") + 1
    if qn > total_q:
        return f"Thank you for participating in this {job_title} interview. We've asked {total_q} questions. Good luck!"
    if qn == 1:
        system = f"""You are a friendly interviewer for a {job_title} role.
Greet the candidate and ask them to introduce themselves."""
    else:
        system = f"""You are an interviewer for {job_title}. Ask one precise question. only ask one question at a time and just one or two lines of question, donot include candidate answer from the previous question in any question."""
    chat = AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        temperature=0.7,
    )
    messages = [HumanMessage(role=m["role"], content=m["content"]) for m in history]
    messages.insert(0, HumanMessage(role="system", content=system))
    resp = chat.invoke(messages)
    return resp.content.strip()

def analyze_answer_and_next(answer: str, last_question: str, job_title: str) -> dict:
    """
    Returns:
      - relevance: one of [relevant,...]
      - warning (optional)
      - rationale (optional)
    """
    from relevance_checker import check_relevance
    rel = check_relevance(answer, last_question, job_title)
    warning = None
    if rel not in ("relevant","partially relevant"):
        warnings = {
            "misbehave": "Please stay respectful.",
            "time wasting": "Please answer concisely.",
            "irrelevant": "Please be on-topic.",
            "nonsense": "That doesn’t make sense."
        }
        warning = warnings.get(rel, "Please answer properly.")
    return { "relevance": rel, "warning": warning }


def generate_summary_and_feedback(history: List[dict], job_title: str) -> str:
    """
    Generate a rational summary and feedback after the interview session.
    """
    system = f"""You are an experienced recruiter for a {job_title} role.
Given the interview conversation, summarize the candidate's performance.
Mention their strengths and areas to improve. Be concise, rational, and constructive."""

    chat = AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        temperature=0.5,
    )

    messages = [HumanMessage(role=m["role"], content=m["content"]) for m in history]
    messages.insert(0, HumanMessage(role="system", content=system))
    resp = chat.invoke(messages)
    return resp.content.strip()