# services/resume/analyzer.py
from typing import Dict, Any, Optional
from ...api.models.schemas import ParseResult, JobData
from ...utils.azure_openai import call_azure_openai_with_retries
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from ...config import settings
import httpx

# Function to get job details from Next.js API
async def get_job_details(job_id: str) -> Optional[JobData]:
    """Fetch job details from Next.js API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.NEXTJS_API_BASE_URL}/jobs/{job_id}",
                headers={"Content-Type": "application/json"}
            )
            
        if response.status_code == 200:
            return JobData(**response.json())
        else:
            print(f"Error fetching job details. Status: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching job details: {str(e)}")
        return None

# LangChain-based job match analysis
async def analyze_resume_match_langchain(parsed_content: ParseResult, job: JobData) -> float:
    """Analyze resume match using LangChain"""
    llm = AzureChatOpenAI(
        azure_deployment=settings.AZURE_DEPLOYMENT_NAME,
        openai_api_version=settings.AZURE_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.2,  # Lower temperature for analysis
    )
    
    template = """Analyze how well this resume matches the job posting and provide a match score between 0-100.
    Consider the following factors:
    - Skills match (technical skills, languages)
    - Experience relevance
    - Education requirements
    - Years of experience

    Job Title: {job_title}
    Job Description: {job_description}
    Job Requirements: {job_requirements}
    Required Skills: {job_skills}
    Salary Range: {salary_info}

    Resume Details:
    - Candidate Name: {candidate_name}
    - Skills: {candidate_skills}
    - Languages: {candidate_languages}
    - Experience: {candidate_experience}
    - Education: {candidate_education}
    - Years of Experience: {years_of_experience}

    Return ONLY the numeric match score (0-100) with no additional text or explanation.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "job_title", "job_description", "job_requirements", "job_skills", 
            "salary_info", "candidate_name", "candidate_skills", "candidate_languages", 
            "candidate_experience", "candidate_education", "years_of_experience"
        ]
    )
    
    # Format the inputs
    salary_info = f"{job.salary_currency or ''} {job.salary_min or ''}-{job.salary_max or ''}" if (job.salary_min or job.salary_max) else "Salary not specified"
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = await chain.arun(
        job_title=job.title,
        job_description=job.description,
        job_requirements=job.requirements or job.description,
        job_skills=", ".join(job.skills) if job.skills else "Not specified",
        salary_info=salary_info,
        candidate_name=parsed_content.name or "Not provided",
        candidate_skills=", ".join(parsed_content.skills.technical) if parsed_content.skills.technical else "None listed",
        candidate_languages=", ".join(parsed_content.skills.languages) if parsed_content.skills.languages else "None listed",
        candidate_experience=f"{len(parsed_content.experience) if parsed_content.experience else 0} positions",
        candidate_education=", ".join([e.degree for e in parsed_content.education]) if parsed_content.education else "None listed",
        years_of_experience=parsed_content.yearsOfExperience or 0
    )
    
    # Extract score from response
    try:
        match_score = float(response.strip())
        return min(100, max(0, match_score))  # Ensure between 0-100
    except:
        print(f"Error parsing match score: {response}")
        return 0

# Analyze resume match
async def analyze_resume_match(parsed_content: ParseResult, job_id: str) -> float:
    try:
        # Get job details
        job = await get_job_details(job_id)
        if not job:
            print(f"Job not found for match analysis: {job_id}")
            return 0
            
        # Use LangChain-based analysis
        return await analyze_resume_match_langchain(parsed_content, job)
        
    except Exception as e:
        print(f"Error in resume match analysis: {str(e)}")
        return 0