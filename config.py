# config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    AZURE_DEPLOYMENT_NAME: str = os.getenv("AZURE_DEPLOYMENT_NAME", "")
    AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION", "2024-08-01-preview")
    
    # Database connection (if using direct DB access)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Next.js API endpoints (for calling back to your Next.js app)
    NEXTJS_API_BASE_URL: str = os.getenv("NEXTJS_API_BASE_URL", "http://localhost:3000/api")
    
    # File storage settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: list = ["pdf", "docx", "doc", "txt", "csv", "xlsx", "xls"]

settings = Settings()