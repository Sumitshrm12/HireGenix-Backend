# services/resume/parser.py
from typing import Dict, Any, Tuple
import json
from ...utils.azure_openai import call_azure_openai_with_retries
from ...api.models.schemas import ResumeData, ParseResult
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from ...config import settings

# Initialize LangChain model
def get_parser_model():
    return AzureChatOpenAI(
        azure_deployment=settings.AZURE_DEPLOYMENT_NAME,
        openai_api_version=settings.AZURE_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.3,  # Lower temperature for parsing tasks
    )

# Function to create AI prompt for resume parsing
def create_resume_parse_prompt(text: str) -> str:
    return f"""
Extract and structure the following resume information into JSON format. Follow these rules:
1. Only extract information explicitly present in the text
2. Format all dates as MM/YYYY
3. For missing fields, use null
4. Categorize skills appropriately
5. Calculate total years of experience from dates

Required JSON structure:
{{
  "name": string,
  "email": string|null,
  "phone": string|null,
  "experience": [{{
    "title": string,
    "company": string,
    "startDate": string,
    "endDate": string,
    "description": string
  }}],
  "education": [{{
    "degree": string,
    "institution": string,
    "startDate": string,
    "endDate": string
  }}],
  "skills": {{
    "technical": string[],
    "soft": string[],
    "languages": string[]
  }},
  "yearsOfExperience": number
}}

Resume text to analyze:
{text[:15000]}

Return ONLY valid JSON with no additional commentary or formatting.
"""

# LangChain-based Parser using Pydantic output parsing
async def parse_resume_with_langchain(text: str) -> ResumeData:
    """Parse resume text using LangChain and Pydantic parser"""
    parser = PydanticOutputParser(pydantic_object=ResumeData)
    
    prompt = PromptTemplate(
        template="""
        Extract information from the following resume and return it in a structured format.
        
        Resume:
        {text}
        
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    model = get_parser_model()
    chain = LLMChain(llm=model, prompt=prompt)
    
    try:
        # Two different approaches - try the first, and if it fails, try the second
        response = await chain.arun(text=text[:15000])
        parsed_data = parser.parse(response)
        return parsed_data
    except Exception as e:
        # Fallback to original approach
        return await parse_resume_text(text)

# Function to parse resume text using Azure OpenAI
async def parse_resume_text(text: str) -> ResumeData:
    prompt = create_resume_parse_prompt(text)
    
    try:
        ai_response = await call_azure_openai_with_retries(prompt, 2000)
        
        # Clean and parse AI response
        json_string = ai_response.replace('```json', '').replace('```', '').strip()
        parsed_data = json.loads(json_string)
        
        # Convert to Pydantic model
        resume_data = ResumeData(**parsed_data)
        return resume_data
        
    except Exception as e:
        print(f"Error parsing resume with AI: {str(e)}")
        # Return minimal parsed data using fallback
        return extract_basic_info_fallback(text)

# Fallback function for basic info extraction
def extract_basic_info_fallback(text: str) -> ResumeData:
    """Extract basic info using regex as a fallback"""
    import re
    
    # Basic regex patterns
    name_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    phone_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    
    # Simple experience detection
    experience_section = re.search(r'(experience|work history|employment)[:\s]*([\s\S]*?)(?=(education|skills|$))', text, re.IGNORECASE)
    date_matches = re.findall(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b', 
                             experience_section.group(2) if experience_section else "", 
                             re.IGNORECASE)
    
    # Create empty ResumeData with fallback values
    return ResumeData(
        name=name_match.group(0).strip() if name_match else "Unknown",
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0) if phone_match else None,
        yearsOfExperience=max(0, len(date_matches) // 2)
    )

# Main function to process resume file
async def process_resume_file(text: str, file_name: str, file_type: str) -> ParseResult:
    """Process resume text and return structured data"""
    try:
        # Parse the resume text using LangChain
        parsed_data = await parse_resume_with_langchain(text)
        
        # Create result
        return ParseResult(
            **parsed_data.dict(),
            fileName=file_name,
            fileType=file_type,
            parseDate=datetime.now().isoformat(),
            rawText=text[:500] + ("..." if len(text) > 500 else ""),
            textLength=len(text),
            parseSuccess=True,
            needsReview=False
        )
        
    except Exception as e:
        print(f"Error processing resume {file_name}: {str(e)}")
        # Return error result
        return ParseResult(
            fileName=file_name,
            fileType=file_type,
            parseDate=datetime.now().isoformat(),
            rawText=text[:500] + ("..." if len(text) > 500 else ""),
            textLength=len(text),
            parseSuccess=False,
            needsReview=True,
            error=str(e)
        )