# utils/azure_openai.py
from typing import Optional
import time
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ..config import settings

# Initialize Azure OpenAI chat model
def get_azure_chat():
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_DEPLOYMENT_NAME,
        openai_api_version=settings.AZURE_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.7,
    )

# Create a simple chain for text completion using LCEL (new syntax)
def create_completion_chain(template: str, input_variables: list):
    """
    Creates a completion chain using LangChain Expression Language (LCEL).
    This replaces the deprecated LLMChain.
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=input_variables
    )
    llm = get_azure_chat()
    # LCEL chain: prompt | llm | output_parser
    chain = prompt | llm | StrOutputParser()
    return chain

# Function to call Azure OpenAI with retry logic
async def call_azure_openai_with_retries(prompt: str, max_tokens: int, retries: int = 3, delay: int = 1000) -> str:
    llm = get_azure_chat()
    
    # Set max tokens if needed
    if max_tokens:
        llm.max_tokens = max_tokens
    
    for attempt in range(retries):
        try:
            # Use LangChain to make the call
            return llm.predict(prompt)
        except Exception as e:
            if attempt < retries - 1:
                # Exponential backoff
                wait_time = delay * (2 ** attempt) / 1000  # Convert to seconds
                print(f"Retrying after {wait_time}s due to error: {str(e)}")
                time.sleep(wait_time)
            else:
                raise e
