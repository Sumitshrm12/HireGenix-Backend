# main.py
# Suppress urllib3 LibreSSL warning before any imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from resume_processor import extract_text, analyze_and_extract
from job_generator import JobDescriptionParams, generate_job_description
from interview_conductor import InterviewEvaluationRequest, evaluate_interview_agentic
from question_generator import QuestionGenerationRequest, generate_questions_agentic
from document_verifier import DocumentVerificationRequest, verify_document_agentic
from ranking_calculator import RankingRequest, calculate_ranking_agentic

# Import new agentic job modules
from agentic_job_processor import (
    get_job_processor,
    JobUploadFile,
    AgenticJobProcessor
)
from agentic_job_description_generator import (
    get_job_description_generator,
    JobDescriptionRequest,
    AgenticJobDescriptionGenerator
)
# Optional imports with graceful degradation
try:
    from proctoring_agent import analyze_proctoring_frame
    PROCTORING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Proctoring agent unavailable (missing dependencies): {e}")
    PROCTORING_AVAILABLE = False
    analyze_proctoring_frame = None

from conversational_interview_agent import get_conversational_interview_agent
from context_service import get_context_service
from deep_resume_analyzer import get_deep_resume_analyzer
from semantic_screening import get_semantic_screening_agent
from adaptive_assessment import get_assessment_engine
from hiring_committee import get_hiring_committee
from market_intelligence import get_market_intelligence_service, get_agentic_market_orchestrator, ResearchDepth
from crawl4ai_service import get_company_intelligence_orchestrator, AgenticCompanyResearchRequest, CrawlStrategy
from outreach_agent import get_outreach_agent
from sourcing_agent import get_sourcing_agent
from roleplay_engine import get_roleplay_engine, ScenarioType
from negotiator_agent import get_negotiator_agent
from onboarding_agent import get_onboarding_agent
from bias_guardian import get_bias_guardian
from langchain_openai import AzureChatOpenAI
from typing import List, Any, Dict
from pydantic import BaseModel
from time import time as current_time
import time
# Import pre-screening routes
from prescreening.api import router as prescreening_router
# Import Crawl4AI service
from crawl4ai_service import router as crawl4ai_router
# Import Agentic Interview API for RealtimeInterviewer
from routes.agentic_interview_api import router as agentic_interview_router
from routes.proctoring_api import router as proctoring_router
from routes.telephonic_screening_api import router as telephonic_screening_router
import base64
import os
import traceback
import json
import re
import asyncio
from langchain_core.messages import HumanMessage
from utils.token_usage import get_token_tracker
import cv2
import numpy as np

# Lazy MediaPipe loading - will be initialized on first use
_mp_module = None
_mp_face_mesh_instance = None
_mp_init_attempted = False

def _get_mediapipe():
    """Lazy load MediaPipe module with proper error handling"""
    global _mp_module, _mp_init_attempted
    if _mp_module is not None:
        return _mp_module
    if _mp_init_attempted:
        return None
    _mp_init_attempted = True
    try:
        import mediapipe as mp
        _mp_module = mp
        return mp
    except Exception:
        return None

def _get_face_mesh():
    """Lazy load FaceMesh with proper error handling"""
    global _mp_face_mesh_instance
    if _mp_face_mesh_instance is not None:
        return _mp_face_mesh_instance
    mp = _get_mediapipe()
    if mp is None:
        return None
    try:
        _mp_face_mesh_instance = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
        return _mp_face_mesh_instance
    except Exception:
        return None
from fastapi import WebSocket

import json
from enum import Enum
import time
import random

import socketio


import psycopg2
import psycopg2.extras
from decouple import config
from dotenv import load_dotenv
import uvicorn

from typing import List, Optional
from uuid import uuid4
from datetime import datetime
from contextlib import asynccontextmanager
# YOLO import is lazy-loaded to prevent startup issues

# Load .env manually if decouple fails (optional fallback)
load_dotenv()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Emotion mapping to match client expectations
EMOTION_MAP = {
    "happy": "positive",
    "surprise": "positive",
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sad": "negative",
    "neutral": "neutral"
}
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include pre-screening router
app.include_router(prescreening_router, prefix="/api/prescreening", tags=["prescreening"])

# Include Crawl4AI router
app.include_router(crawl4ai_router, prefix="/api/crawl4ai", tags=["crawl4ai"])

# Include Agentic Interview API router (for RealtimeInterviewer integration)
app.include_router(agentic_interview_router)
app.include_router(proctoring_router)

# Include Telephonic Screening API router
app.include_router(telephonic_screening_router, prefix="/api", tags=["Telephonic Screening"])

# Configure Azure OpenAI
llm =   AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        temperature=0.7,
        max_tokens=2500,
     
    )


class InterviewEvaluationRequest(BaseModel):
    formatted_input: List[Dict[str, str]]
    candidate_name: str
    candidate_email: str
    # proctoring_data_analysis: Dict[str, Any]
    # detected_face: bool

def clean_response(text: str) -> str:
    """Clean the AI response by removing markdown code blocks."""
    return re.sub(r'```json|```', '', text).strip()
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.ping_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.ping_tasks[client_id] = asyncio.create_task(self.send_ping(websocket))
        return client_id

    async def disconnect(self, client_id: str):
        if client_id in self.ping_tasks:
            self.ping_tasks[client_id].cancel()
            del self.ping_tasks[client_id]
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_ping(self, websocket: WebSocket):
        
        try:
            while True:
                await asyncio.sleep(25)
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": int(time.time() * 1000)
                    })
                except Exception as e:
                    print(f"Ping failed: {e}")
                    break
        except asyncio.CancelledError:
            pass

manager = ConnectionManager()



@app.post("/job-description")
async def job_description_endpoint(params: JobDescriptionParams):
    """
    LEGACY ENDPOINT - DEPRECATED
    Use /api/jobs/agentic/generate-description for new agentic system
    """
    try:
        print("⚠️ Using legacy job description generator...")
        job_desc = await generate_job_description(params)
        
        return {
            "success": True,
            "jobDescription": job_desc.get('job_description'),
            "analysis": job_desc.get('analysis'),
            "token_usage": job_desc.get('token_usage'),
            "agent_metadata": job_desc.get('agent_metadata')
        }
    except Exception as e:
        print(f"❌ Error generating job description: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate job description: {str(e)}")


@app.post("/api/jobs/agentic/generate-description")
async def agentic_job_description_endpoint(request: JobDescriptionRequest):
    """
    AGENTIC AI: Advanced Job Description Generation with Company Intelligence
    
    Features:
    - 5-agent multi-step workflow (Company Intelligence, Market Research, Drafting, Refinement, QA)
    - 100% company-specific content using company intelligence
    - Market insights and industry trends integration
    - Quality scoring and validation
    - RAG storage for continuous learning
    - Deep company context weaving
    
    Returns comprehensive job description with quality metrics
    """
    try:
        print(f"🤖 [AGENTIC] Generating job description for: {request.jobTitle}")
        
        # Get the agentic generator
        generator = get_job_description_generator()
        
        # Generate with company intelligence if available
        result = await generator.generate_description(
            request=request,
            company_intelligence=None  # Will be fetched by agent
        )
        
        print(f"✅ [AGENTIC] Job description generated successfully")
        print(f"📊 Quality Score: {result['quality_score']:.2%}")
        print(f"🏢 Company-Specific: {result['agent_metadata']['company_specific']}")
        
        return result
        
    except Exception as e:
        print(f"❌ [AGENTIC] Error generating job description: {str(e)}")
        import traceback
        print(f"ERROR details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate job description: {str(e)}"
        )


@app.post("/api/jobs/agentic/process-upload")
async def agentic_job_upload_endpoint(
    files: List[UploadFile] = File(...),
    company_id: str = Form(...),
    user_id: str = Form(...),
    company_intelligence: Optional[str] = Form(None)
):
    """
    AGENTIC AI: Advanced Job File Processing with Multi-Agent System
    
    Features:
    - 5-agent workflow (Parser, Context Enrichment, Quality Enhancement, Validation, Storage)
    - Intelligent file parsing with context awareness
    - Company intelligence integration for 100% authentic jobs
    - Quality scoring and validation
    - Automatic enhancement and optimization
    - RAG storage for historical learning
    
    Supports: CSV, Excel, PDF, Word, Text files
    Returns: Processed jobs with quality metrics
    """
    try:
        print(f"🤖 [AGENTIC] Processing {len(files)} job files...")
        print(f"📂 Files received: {[f.filename for f in files]}")
        
        # Parse company intelligence if provided
        company_intel_data = None
        if company_intelligence:
            try:
                company_intel_data = json.loads(company_intelligence)
                print(f"🏢 Company intelligence parsed successfully")
            except:
                print("⚠️ Could not parse company intelligence, proceeding without it")
        
        # Read and prepare files
        job_files = []
        for file in files:
            print(f"\n📄 Processing file: {file.filename}")
            print(f"   Content-Type: {file.content_type}")
            
            # Check file size first
            file.file.seek(0, 2)  # Seek to end
            file_size = file.file.tell()
            file.file.seek(0)  # Reset to beginning
            print(f"   File size: {file_size} bytes")
            
            if file_size == 0:
                print(f"   ❌ ERROR: File is empty!")
                continue
            
            # Determine file type and extract text
            file_extension = file.filename.split('.')[-1].lower()
            print(f"   Extension: {file_extension}")
            
            # Reset file position and use existing extract_text utility
            text_content = ""
            try:
                file.file.seek(0)  # Reset file position before reading
                text_content = extract_text(file)
                print(f"   ✅ Extracted text: {len(text_content)} chars")
                if text_content:
                    print(f"   📝 Preview: {text_content[:200]}...")
            except Exception as e:
                print(f"   ⚠️ extract_text failed: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                # Fallback to raw content
                file.file.seek(0)
                content = await file.read()
                text_content = content.decode('utf-8', errors='ignore')
                print(f"   📝 Fallback extraction: {len(text_content)} chars")
            
            if not text_content or len(text_content.strip()) < 10:
                print(f"   ⚠️ WARNING: Very little text extracted ({len(text_content.strip())} chars)")
            
            job_files.append(JobUploadFile(
                filename=file.filename,
                content=text_content,
                file_type=file.content_type or f"application/{file_extension}"
            ))
            print(f"   ✅ File added to processing queue")
        
        print(f"\n📊 Total files to process: {len(job_files)}")
        
        if len(job_files) == 0:
            print("❌ No valid files to process!")
            return {
                "success": False,
                "processed_jobs": [],
                "total_processed": 0,
                "total_errors": 1,
                "errors": [{"error": "No valid files could be processed"}],
                "token_usage": {},
                "agent_metadata": {
                    "agents_deployed": 0,
                    "company_intelligence_used": False,
                    "avg_quality_score": 0
                }
            }
        
        # Get the agentic processor
        processor = get_job_processor()
        
        # Process jobs
        result = await processor.process_jobs(
            files=job_files,
            company_id=company_id,
            user_id=user_id,
            company_intelligence=company_intel_data
        )
        
        print(f"✅ [AGENTIC] Processed {result['total_processed']} jobs")
        print(f"📊 Average Quality Score: {result['agent_metadata']['avg_quality_score']:.2%}")
        print(f"🏢 Company Intelligence Used: {result['agent_metadata']['company_intelligence_used']}")
        
        return result
        
    except Exception as e:
        print(f"❌ [AGENTIC] Error processing job files: {str(e)}")
        import traceback
        print(f"ERROR details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process job files: {str(e)}"
        )


@app.post("/score-resumes")
async def score_resumes_endpoint(
    job_description: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    AGENTIC AI: Resume analysis with DSPy, RAG, and Chain-of-Thought
    
    Features:
    - Multi-step reasoning workflow
    - Context-aware scoring with RAG
    - Persistent learning in Qdrant
    - Structured candidate data extraction
    """
    results: List[Any] = []
    tokenUsage: List[Any] = []
    
    print(f"🤖 Using ResumeAnalyzerAgent for {len(files)} resumes...")

    for file in files:
        try:
            file.file.seek(0)
            resume_text = extract_text(file)
            
            # Use agentic AI version
            data = await analyze_and_extract(resume_text, job_description)
            
            tokenUsage.append(data.get('token_usage'))
            results.append(data.get('data'))
            
            print(f"✅ Analyzed {file.filename} - Score: {data['data'].get('match_score', 'N/A')}/100")
            
        except Exception as e:
            print(f"❌ Error analyzing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "candidate": {"name": "Error", "email": ""}
            })

    return {
        "success": True,
        "results": results,
        "tokenUsage": tokenUsage,
        "agent_metadata": {
            "agent": "ResumeAnalyzerAgent",
            "files_processed": len(files),
            "successful": len([r for r in results if 'error' not in r])
        }
    }


@app.post("/api/candidates/agentic-process")
async def agentic_candidate_process_endpoint(
    resume: UploadFile = File(...),
    job_id: str = Form(...),
    company_id: str = Form(...),
    use_agentic: str = Form("true"),
    user_id: str = Form(None)
):
    """
    🤖 AGENTIC AI: Advanced Candidate Processing with Multi-Agent System
    
    Features:
    - 7-agent LangGraph workflow
    - Company Intelligence integration
    - Job Description Intelligence matching
    - Market Trends analysis via Tavily
    - Career Trajectory prediction
    - Redis Vector storage
    - DSPy reasoning modules
    - Intelligent multi-dimensional scoring
    
    Returns: Comprehensive candidate intelligence with scores and recommendations
    """
    try:
        print(f"🤖 [AGENTIC] Processing candidate for job: {job_id}")
        
        # Extract resume text
        resume.file.seek(0)
        resume_text = extract_text(resume)
        
        if not resume_text or len(resume_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from resume"
            )
        
        # Import and use agentic processor
        from agentic_candidate_processor import get_candidate_processor
        
        processor = get_candidate_processor()
        
        # Process through multi-agent system
        result = await processor.process_candidate(
            resume_text=resume_text,
            job_id=job_id,
            company_id=company_id,
            user_id=user_id
        )
        
        if result['success']:
            print(f"✅ [AGENTIC] Candidate processed successfully")
            print(f"📊 Overall Score: {result['scores']['overall']:.2%}")
            print(f"🎯 Recommendations: {len(result['recommendations'])}")
            
            return {
                "success": True,
                "candidate_id": result['candidate_id'],
                "profile": result['profile'],
                "scores": result['scores'],
                "recommendations": result['recommendations'],
                "career_analysis": result['career_analysis'],
                "market_insights": result['market_insights'],
                "company_fit": result['company_fit'],
                "metadata": result['metadata'],
                "agent_metadata": {
                    "system": "agentic_ai",
                    "agents_executed": 7,
                    "workflow": "langgraph",
                    "reasoning": "dspy",
                    "storage": ["redis_vector", "qdrant"]
                }
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Agentic processing failed')
            )
        
    except Exception as e:
        print(f"❌ [AGENTIC] Error: {str(e)}")
        import traceback
        print(f"ERROR details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Agentic candidate processing failed: {str(e)}"
        )


@app.post("/api/parse-resume")
async def parse_resume_endpoint(file: UploadFile = File(...), deep_analysis: bool = Form(False)):
    """
    Parse resume to extract candidate information.
    Optional: deep_analysis=True triggers AI-powered semantic extraction.
    
    Supports: PDF, DOCX
    Returns: Extracted candidate data (name, email, phone, skills, etc.)
    """
    try:
        print(f"📄 Parsing resume: {file.filename} (Deep Analysis: {deep_analysis})")
        
        # Read file content
        file_bytes = await file.read()
        
        if deep_analysis:
            # Use Deep Resume Analyzer
            analyzer = get_deep_resume_analyzer()
            parsed_data = await analyzer.analyze(file_bytes, file.filename)
        else:
            # Use Standard Regex Parser
            from resume_parser import ResumeParser
            parser = ResumeParser()
            parsed_data = parser.parse(file_bytes, file.filename)
        
        print(f"✅ Resume parsed successfully")
        print(f"📊 Extracted: {parsed_data.get('name', 'N/A')}, {parsed_data.get('email', 'N/A')}")
        
        return {
            "success": True,
            **parsed_data
        }
        
    except Exception as e:
        print(f"❌ Error parsing resume: {str(e)}")
        import traceback
        print(f"ERROR details: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse resume: {str(e)}"
        )

@app.post("/api/screen-candidate")
async def screen_candidate_endpoint(data: dict):
    """
    AGENTIC AI: Semantic Screening
    Matches candidate profile against job description using AI reasoning.
    """
    try:
        candidate_profile = data.get('candidate_profile')
        job_description = data.get('job_description')
        company_context = data.get('company_context')
        
        if not candidate_profile or not job_description:
            raise HTTPException(status_code=400, detail="candidate_profile and job_description are required")
            
        agent = get_semantic_screening_agent()
        result = await agent.screen_candidate(candidate_profile, job_description, company_context)
        
        return {"success": True, "screening_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")

@app.post("/api/generate-assessment")
async def generate_assessment_endpoint(data: dict):
    """
    AGENTIC AI: Adaptive Assessment Generation
    Creates a real-world challenge based on skill and difficulty.
    """
    try:
        skill = data.get('skill')
        difficulty = data.get('difficulty', 'medium')
        context = data.get('context')
        
        if not skill:
            raise HTTPException(status_code=400, detail="skill is required")
            
        engine = get_assessment_engine()
        challenge = await engine.generate_challenge(skill, difficulty, context)
        
        return {"success": True, "challenge": challenge}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment generation failed: {str(e)}")

@app.post("/api/evaluate-assessment")
async def evaluate_assessment_endpoint(data: dict):
    """
    AGENTIC AI: Assessment Evaluation
    Evaluates candidate code submission.
    """
    try:
        challenge = data.get('challenge')
        submission = data.get('submission')
        language = data.get('language', 'python')
        
        if not challenge or not submission:
            raise HTTPException(status_code=400, detail="challenge and submission are required")
            
        engine = get_assessment_engine()
        evaluation = await engine.evaluate_submission(challenge, submission, language)
        
        return {"success": True, "evaluation": evaluation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/api/hiring-committee-review")
async def hiring_committee_review_endpoint(data: dict):
    """
    AGENTIC AI: Hiring Committee Consensus
    Simulates a multi-stakeholder debate to reach a final hiring decision.
    """
    try:
        candidate_name = data.get('candidate_name')
        job_role = data.get('job_role')
        interview_transcript = data.get('interview_transcript')
        assessment_results = data.get('assessment_results')
        resume_analysis = data.get('resume_analysis')
        
        if not all([candidate_name, job_role, interview_transcript]):
            raise HTTPException(status_code=400, detail="Missing required fields for committee review")
            
        committee = get_hiring_committee()
        decision = await committee.conduct_review(
            candidate_name,
            job_role,
            interview_transcript,
            assessment_results or {},
            resume_analysis or {}
        )
        
        return {"success": True, "committee_decision": decision}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Committee review failed: {str(e)}")

@app.post("/api/company/reputation")
async def company_reputation_endpoint(data: dict):
    """
    AGENTIC AI: 360° Company Reputation Analysis
    Analyzes market sentiment, employee reviews, and news to generate a reputation score.
    """
    try:
        company_name = data.get('company_name')
        if not company_name:
            raise HTTPException(status_code=400, detail="company_name is required")
            
        service = get_market_intelligence_service()
        reputation = await service.analyze_company_reputation(company_name)
        
        return {"success": True, "reputation": reputation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reputation analysis failed: {str(e)}")


# ============================================================================
# ENHANCED MARKET INTELLIGENCE ENDPOINTS v2.0
# ============================================================================

@app.post("/api/market-intelligence/salary-trends")
async def salary_trends_endpoint(data: dict):
    """
    AGENTIC AI: Real-Time Salary Trends Analysis
    
    Fetches live salary benchmarks for job roles using Tavily search and AI synthesis.
    
    Request Body:
    - job_title: (required) Job title to analyze (e.g., "Senior React Developer")
    - location: (required) Location for salary data (e.g., "Bangalore", "India", "Remote")
    - experience_level: (optional) Experience level (e.g., "Senior", "Mid", "Junior")
    
    Returns:
    - salary_range: {min, max, median, currency}
    - market_positioning: below_market | competitive | above_market
    - remote_premium: percentage for remote roles
    - yoy_change: year-over-year percentage change
    - hot_skills_premium: skills that command higher salary
    - recommendation: salary recommendation for employer
    """
    try:
        job_title = data.get('job_title')
        location = data.get('location')
        experience_level = data.get('experience_level')
        
        if not job_title or not location:
            raise HTTPException(status_code=400, detail="job_title and location are required")
        
        print(f"💰 Salary trends request: {job_title} in {location}")
        
        service = get_market_intelligence_service()
        salary_data = await service.fetch_salary_trends(job_title, location, experience_level)
        
        return {
            "success": True,
            "job_title": job_title,
            "location": location,
            "experience_level": experience_level,
            "data": salary_data
        }
    except Exception as e:
        print(f"❌ Salary trends error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Salary trends analysis failed: {str(e)}")


@app.post("/api/market-intelligence/skills-demand")
async def skills_demand_endpoint(data: dict):
    """
    AGENTIC AI: Skills Demand & Trend Analysis
    
    Analyzes real-time demand for tech skills using Tavily search.
    
    Request Body:
    - tech_stack: (required) List of technologies to analyze (e.g., ["React", "Python", "AWS"])
    - industry: (optional) Industry context (e.g., "Fintech", "Healthcare")
    
    Returns:
    - skills_analysis: demand level and trend for each skill
    - trending_up: skills with increasing demand
    - trending_down: skills with decreasing demand
    - emerging_skills: new skills gaining traction
    - skill_combinations: powerful skill combinations employers want
    - hiring_difficulty: overall hiring difficulty and time estimates
    - recommendations: strategic hiring recommendations
    """
    try:
        tech_stack = data.get('tech_stack', [])
        industry = data.get('industry')
        
        if not tech_stack:
            raise HTTPException(status_code=400, detail="tech_stack is required (list of technologies)")
        
        print(f"📈 Skills demand request: {', '.join(tech_stack[:5])}")
        
        service = get_market_intelligence_service()
        skills_data = await service.fetch_skills_demand(tech_stack, industry)
        
        return {
            "success": True,
            "tech_stack": tech_stack,
            "industry": industry,
            "data": skills_data
        }
    except Exception as e:
        print(f"❌ Skills demand error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Skills demand analysis failed: {str(e)}")


@app.post("/api/market-intelligence/competitor-hiring")
async def competitor_hiring_endpoint(data: dict):
    """
    AGENTIC AI: Competitor Hiring Activity Intelligence
    
    Analyzes competitor hiring patterns and strategies using Tavily search.
    
    Request Body:
    - company_name: (required) Your company name (for comparison)
    - competitors: (required) List of competitor company names
    - job_roles: (optional) Specific roles to track
    
    Returns:
    - competitor_analysis: hiring intensity and focus areas per competitor
    - talent_competition_level: high | medium | low
    - competitor_advantages: what competitors are offering
    - your_opportunities: strategies to compete for talent
    - poaching_risk: risk level with explanation
    - recommendations: strategic hiring recommendations
    """
    try:
        company_name = data.get('company_name')
        competitors = data.get('competitors', [])
        job_roles = data.get('job_roles')
        
        if not company_name:
            raise HTTPException(status_code=400, detail="company_name is required")
        if not competitors:
            raise HTTPException(status_code=400, detail="competitors list is required")
        
        print(f"🏢 Competitor hiring request: {company_name} vs {', '.join(competitors[:3])}")
        
        service = get_market_intelligence_service()
        competitor_data = await service.fetch_competitor_hiring(company_name, competitors, job_roles)
        
        return {
            "success": True,
            "company_name": company_name,
            "competitors_analyzed": competitors,
            "data": competitor_data
        }
    except Exception as e:
        print(f"❌ Competitor hiring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Competitor hiring analysis failed: {str(e)}")


@app.post("/api/market-intelligence/company-news")
async def company_news_endpoint(data: dict):
    """
    AGENTIC AI: Company News & Sentiment Analysis
    
    Fetches recent news about a company using Tavily search.
    
    Request Body:
    - company_name: (required) Company name to search news for
    - topics: (optional) Specific topics to focus on (e.g., ["funding", "product launch"])
    
    Returns:
    - recent_news: list of news items with title, summary, date, sentiment, category
    - overall_sentiment: positive | neutral | negative
    - key_themes: recurring themes in the news
    - growth_signals: positive indicators
    - risk_signals: concerning indicators
    - last_major_event: most significant recent event
    """
    try:
        company_name = data.get('company_name')
        topics = data.get('topics')
        
        if not company_name:
            raise HTTPException(status_code=400, detail="company_name is required")
        
        print(f"📰 Company news request: {company_name}")
        
        service = get_market_intelligence_service()
        news_data = await service.fetch_company_news(company_name, topics)
        
        return {
            "success": True,
            "company_name": company_name,
            "data": news_data
        }
    except Exception as e:
        print(f"❌ Company news error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Company news analysis failed: {str(e)}")


@app.post("/api/market-intelligence/industry-insights")
async def industry_insights_endpoint(data: dict):
    """
    AGENTIC AI: Industry-Wide Insights & Trends
    
    Fetches comprehensive industry analysis using Tavily search.
    
    Request Body:
    - industry: (required) Industry to analyze (e.g., "Fintech", "SaaS", "Healthcare Tech")
    - focus_areas: (optional) Specific areas (e.g., ["hiring trends", "remote work"])
    
    Returns:
    - industry_overview: current state summary
    - key_trends: list of trends with impact and timeline
    - workforce_dynamics: talent availability, remote adoption, time to hire, turnover
    - technology_adoption: technologies being adopted
    - challenges: major hiring challenges
    - opportunities: opportunities for employers
    - predictions_2025: predictions for next year
    """
    try:
        industry = data.get('industry')
        focus_areas = data.get('focus_areas')
        
        if not industry:
            raise HTTPException(status_code=400, detail="industry is required")
        
        print(f"🏭 Industry insights request: {industry}")
        
        service = get_market_intelligence_service()
        industry_data = await service.fetch_industry_insights(industry, focus_areas)
        
        return {
            "success": True,
            "industry": industry,
            "data": industry_data
        }
    except Exception as e:
        print(f"❌ Industry insights error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Industry insights analysis failed: {str(e)}")


@app.post("/api/market-intelligence/dashboard")
async def market_intelligence_dashboard_endpoint(data: dict):
    """
    AGENTIC AI: Comprehensive Market Intelligence Dashboard
    
    Fetches all market intelligence data in parallel for the enterprise hiring dashboard.
    
    Request Body:
    - company_name: (required) Your company name
    - industry: (required) Your industry
    - job_title: (required) Job title for salary analysis
    - location: (required) Location for salary analysis
    - tech_stack: (required) List of technologies
    - competitors: (required) List of competitor companies
    
    Returns comprehensive dashboard with:
    - salary_intelligence: salary trends and recommendations
    - skills_demand: skills analysis and market outlook
    - competitor_hiring: competitor activity and strategies
    - company_news: recent news and sentiment
    - industry_insights: industry trends and predictions
    - summary: quick overview metrics
    """
    try:
        company_name = data.get('company_name')
        industry = data.get('industry')
        job_title = data.get('job_title')
        location = data.get('location')
        tech_stack = data.get('tech_stack', [])
        competitors = data.get('competitors', [])
        
        if not all([company_name, industry, job_title, location]):
            raise HTTPException(
                status_code=400,
                detail="company_name, industry, job_title, and location are required"
            )
        
        print(f"🚀 Market Intelligence Dashboard request for {company_name}")
        
        service = get_market_intelligence_service()
        dashboard = await service.fetch_comprehensive_market_dashboard(
            company_name=company_name,
            industry=industry,
            job_title=job_title,
            location=location,
            tech_stack=tech_stack,
            competitors=competitors
        )
        
        return {
            "success": True,
            **dashboard
        }
    except Exception as e:
        print(f"❌ Market intelligence dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Market intelligence dashboard failed: {str(e)}")

# ============================================================================
# AGENTIC AI MARKET INTELLIGENCE v3.0 ENDPOINTS
# ============================================================================

@app.post("/api/agentic/market-research")
async def agentic_market_research_endpoint(data: dict):
    """
    🤖 AGENTIC AI v3.0: Advanced Market Research with Multi-Agent Orchestration
    
    Features:
    - 5-agent autonomous system (Planner, Searcher, Enricher, Analyzer, Validator)
    - Intelligent query generation and expansion
    - Parallel Tavily search execution with batching
    - Cross-source data correlation
    - Real-time confidence scoring
    - Memory and caching layer
    
    Request Body:
    - job_title: (required) Job title to research
    - tech_stack: (optional) List of technologies
    - industry: (optional) Industry sector
    - location: (optional) Geographic location
    - depth: (optional) Research depth: "quick" | "standard" | "deep" | "enterprise"
    
    Returns comprehensive market intelligence with agent reports.
    """
    try:
        job_title = data.get('job_title')
        tech_stack = data.get('tech_stack', [])
        industry = data.get('industry')
        location = data.get('location')
        depth_str = data.get('depth', 'standard')
        
        if not job_title:
            raise HTTPException(status_code=400, detail="job_title is required")
        
        # Map depth string to enum
        depth_map = {
            'quick': ResearchDepth.QUICK,
            'standard': ResearchDepth.STANDARD,
            'deep': ResearchDepth.DEEP,
            'enterprise': ResearchDepth.ENTERPRISE
        }
        depth = depth_map.get(depth_str, ResearchDepth.STANDARD)
        
        print(f"🤖 [AGENTIC v3.0] Market Research: {job_title}, Depth: {depth.value}")
        
        orchestrator = get_agentic_market_orchestrator()
        result = await orchestrator.orchestrate_market_research(
            job_title=job_title,
            tech_stack=tech_stack,
            industry=industry,
            location=location,
            depth=depth
        )
        
        print(f"✅ [AGENTIC v3.0] Research completed: {result.get('summary', {}).get('confidence', 0):.0%} confidence")
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        print(f"❌ [AGENTIC v3.0] Market research error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agentic market research failed: {str(e)}")


@app.post("/api/agentic/company-research")
async def agentic_company_research_endpoint(data: dict):
    """
    🤖 AGENTIC AI v3.0: Advanced Company Intelligence with Multi-Agent Orchestration
    
    Features:
    - 6-agent autonomous system (Discovery, Strategy, Extraction, Analysis, CrossRef, Validation)
    - Deep Crawl4AI extraction with JS rendering
    - Intelligent URL discovery and prioritization
    - Cross-source data validation
    - Comprehensive product/service extraction
    - Real-time confidence scoring
    
    Request Body:
    - company_name: (required) Company name to research
    - website: (optional) Company website URL
    - industry: (optional) Industry sector
    - research_depth: (optional) "fast" | "standard" | "deep" | "intelligent" | "stealth"
    - max_sources: (optional) Maximum sources to crawl (5-100)
    - include_competitors: (optional) Include competitor analysis
    - include_news: (optional) Include recent news
    - include_social: (optional) Include social media analysis
    - parallel_agents: (optional) Number of parallel crawl agents (1-10)
    
    Returns comprehensive company intelligence with agent reports.
    """
    try:
        company_name = data.get('company_name')
        website = data.get('website')
        industry = data.get('industry')
        depth_str = data.get('research_depth', 'intelligent')
        max_sources = data.get('max_sources', 30)
        include_competitors = data.get('include_competitors', True)
        include_news = data.get('include_news', True)
        include_social = data.get('include_social', True)
        parallel_agents = data.get('parallel_agents', 5)
        
        if not company_name:
            raise HTTPException(status_code=400, detail="company_name is required")
        
        # Map depth string to enum
        depth_map = {
            'fast': CrawlStrategy.FAST,
            'standard': CrawlStrategy.STANDARD,
            'deep': CrawlStrategy.DEEP,
            'intelligent': CrawlStrategy.INTELLIGENT,
            'stealth': CrawlStrategy.STEALTH
        }
        research_depth = depth_map.get(depth_str, CrawlStrategy.INTELLIGENT)
        
        print(f"🤖 [AGENTIC v3.0] Company Research: {company_name}, Depth: {research_depth.value}")
        
        # Create request object
        request = AgenticCompanyResearchRequest(
            company_name=company_name,
            website=website,
            industry=industry,
            research_depth=research_depth,
            max_sources=max_sources,
            include_competitors=include_competitors,
            include_news=include_news,
            include_social=include_social,
            parallel_agents=parallel_agents
        )
        
        orchestrator = get_company_intelligence_orchestrator()
        result = await orchestrator.orchestrate_research(request)
        
        print(f"✅ [AGENTIC v3.0] Company research completed: {result.confidence_score:.0%} confidence")
        
        return {
            "success": result.success,
            "company_name": result.company_name,
            "research_id": result.research_id,
            "agents_deployed": result.agents_deployed,
            "total_sources_crawled": result.total_sources_crawled,
            "data_quality": result.data_quality.value,
            "confidence_score": result.confidence_score,
            "processing_time_seconds": result.processing_time_seconds,
            "intelligence_data": result.intelligence_data,
            "agent_reports": result.agent_reports,
            "recommendations": result.recommendations
        }
        
    except Exception as e:
        print(f"❌ [AGENTIC v3.0] Company research error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agentic company research failed: {str(e)}")


@app.post("/api/agentic/combined-intelligence")
async def agentic_combined_intelligence_endpoint(data: dict):
    """
    🤖 AGENTIC AI v3.0: Combined Market + Company Intelligence
    
    Runs both market research and company research in parallel for comprehensive hiring intelligence.
    
    Request Body:
    - company_name: (required) Company name
    - website: (optional) Company website URL
    - job_title: (required) Job title to research
    - industry: (optional) Industry sector
    - location: (optional) Geographic location
    - tech_stack: (optional) List of technologies
    - competitors: (optional) List of competitor companies
    - depth: (optional) "quick" | "standard" | "deep" | "enterprise"
    
    Returns combined market and company intelligence.
    """
    try:
        company_name = data.get('company_name')
        website = data.get('website')
        job_title = data.get('job_title')
        industry = data.get('industry')
        location = data.get('location')
        tech_stack = data.get('tech_stack', [])
        competitors = data.get('competitors', [])
        depth_str = data.get('depth', 'standard')
        
        if not company_name or not job_title:
            raise HTTPException(status_code=400, detail="company_name and job_title are required")
        
        print(f"🤖 [AGENTIC v3.0] Combined Intelligence: {company_name} + {job_title}")
        
        # Map depth
        depth_map = {
            'quick': ResearchDepth.QUICK,
            'standard': ResearchDepth.STANDARD,
            'deep': ResearchDepth.DEEP,
            'enterprise': ResearchDepth.ENTERPRISE
        }
        market_depth = depth_map.get(depth_str, ResearchDepth.STANDARD)
        
        crawl_depth_map = {
            'quick': CrawlStrategy.FAST,
            'standard': CrawlStrategy.STANDARD,
            'deep': CrawlStrategy.DEEP,
            'enterprise': CrawlStrategy.INTELLIGENT
        }
        company_depth = crawl_depth_map.get(depth_str, CrawlStrategy.INTELLIGENT)
        
        # Run both in parallel
        import asyncio
        
        market_orchestrator = get_agentic_market_orchestrator()
        company_orchestrator = get_company_intelligence_orchestrator()
        
        # Create company request
        company_request = AgenticCompanyResearchRequest(
            company_name=company_name,
            website=website,
            industry=industry,
            research_depth=company_depth,
            max_sources=20,
            include_competitors=len(competitors) > 0,
            include_news=True,
            include_social=True,
            parallel_agents=5
        )
        
        # Run in parallel
        market_task = market_orchestrator.orchestrate_market_research(
            job_title=job_title,
            tech_stack=tech_stack,
            industry=industry,
            location=location,
            depth=market_depth
        )
        company_task = company_orchestrator.orchestrate_research(company_request)
        
        market_result, company_result = await asyncio.gather(
            market_task,
            company_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(market_result, Exception):
            market_result = {"error": str(market_result)}
        if isinstance(company_result, Exception):
            company_result_dict = {"error": str(company_result)}
        else:
            company_result_dict = {
                "success": company_result.success,
                "intelligence_data": company_result.intelligence_data,
                "confidence_score": company_result.confidence_score,
                "data_quality": company_result.data_quality.value,
                "recommendations": company_result.recommendations,
                "agent_reports": company_result.agent_reports
            }
        
        # Calculate combined confidence
        market_confidence = market_result.get("summary", {}).get("confidence", 0) if isinstance(market_result, dict) else 0
        company_confidence = company_result_dict.get("confidence_score", 0) if isinstance(company_result_dict, dict) else 0
        combined_confidence = (market_confidence + company_confidence) / 2
        
        print(f"✅ [AGENTIC v3.0] Combined intelligence completed: {combined_confidence:.0%} confidence")
        
        return {
            "success": True,
            "company_name": company_name,
            "job_title": job_title,
            "combined_confidence": combined_confidence,
            "market_intelligence": market_result,
            "company_intelligence": company_result_dict,
            "summary": {
                "market_outlook": market_result.get("market_intelligence", {}).get("market_overview", {}).get("growth_trajectory", "unknown") if isinstance(market_result, dict) else "unknown",
                "company_products": len(company_result_dict.get("intelligence_data", {}).get("products_services", [])) if isinstance(company_result_dict, dict) else 0,
                "hiring_recommendations": market_result.get("market_intelligence", {}).get("hiring_recommendations", [])[:3] if isinstance(market_result, dict) else [],
                "company_tech_stack": company_result_dict.get("intelligence_data", {}).get("technology", {}).get("tech_stack", [])[:5] if isinstance(company_result_dict, dict) else []
            }
        }
        
    except Exception as e:
        print(f"❌ [AGENTIC v3.0] Combined intelligence error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Combined intelligence failed: {str(e)}")


@app.post("/api/outreach/generate")
async def generate_outreach_endpoint(data: dict):
    """
    AGENTIC AI: Hyper-Personalized Outreach
    Generates a headhunter-style email using deep resume analysis and company context.
    """
    try:
        candidate_profile = data.get('candidate_profile')
        job_details = data.get('job_details')
        company_context = data.get('company_context')
        sender_name = data.get('sender_name', 'Hiring Team')
        
        if not all([candidate_profile, job_details, company_context]):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        agent = get_outreach_agent()
        email = await agent.generate_outreach_email(
            candidate_profile,
            job_details,
            company_context,
            sender_name
        )
        
        return {"success": True, "email": email}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outreach generation failed: {str(e)}")

@app.post("/api/sourcing/start")
async def start_sourcing_endpoint(data: dict):
    """
    AGENTIC AI: Deep Sourcing Agent
    Autonomous headhunter that finds, evaluates, and engages candidates using real-time market data.
    """
    try:
        job_role = data.get('job_role')
        job_description = data.get('job_description')
        company_name = data.get('company_name')
        company_context = data.get('company_context', {})
        
        if not all([job_role, job_description, company_name]):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        agent = get_sourcing_agent()
        result = await agent.start_sourcing_mission(
            job_role,
            job_description,
            company_name,
            company_context
        )
        
        return {"success": True, "sourcing_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sourcing mission failed: {str(e)}")

@app.post("/api/roleplay/start")
async def start_roleplay_endpoint(data: dict):
    """
    AGENTIC AI: Roleplay Simulation
    Starts a real-world scenario (e.g., Crisis Management) with an AI persona.
    """
    try:
        scenario_type = data.get('scenario_type', 'conflict_resolution')
        candidate_name = data.get('candidate_name', 'Candidate')
        role = data.get('role', 'Employee')
        
        engine = get_roleplay_engine()
        state = await engine.start_simulation(ScenarioType(scenario_type), candidate_name, role)
        
        return {
            "success": True,
            "scenario_description": state.scenario_description,
            "ai_persona": state.ai_persona,
            "initial_message": state.last_ai_response,
            "state": state.__dict__
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Roleplay start failed: {str(e)}")

@app.post("/api/roleplay/message")
async def process_roleplay_message_endpoint(data: dict):
    """
    AGENTIC AI: Process Roleplay Message
    """
    try:
        message = data.get('message')
        state_data = data.get('state')
        
        if not message or not state_data:
            raise HTTPException(status_code=400, detail="message and state are required")
            
        # Reconstruct state (simplified for demo)
        from roleplay_engine import RoleplayState
        state = RoleplayState(**state_data)
        
        engine = get_roleplay_engine()
        new_state = await engine.process_message(state, message)
        
        return {
            "success": True,
            "ai_response": new_state.last_ai_response,
            "ai_mood": new_state.ai_mood,
            "status": new_state.status,
            "state": new_state.__dict__
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Roleplay processing failed: {str(e)}")

@app.post("/api/negotiation/start")
async def start_negotiation_endpoint(data: dict):
    """
    AGENTIC AI: Start Salary Negotiation
    Initiates a negotiation session with an initial offer based on market data.
    """
    try:
        candidate_name = data.get('candidate_name')
        role = data.get('role')
        budget_max = data.get('budget_max')
        
        if not all([candidate_name, role, budget_max]):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        agent = get_negotiator_agent()
        state = await agent.start_negotiation(candidate_name, role, float(budget_max))
        
        return {
            "success": True,
            "initial_offer": state.current_offer,
            "opening_message": state.history[-1]['content'],
            "state": state.__dict__
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Negotiation start failed: {str(e)}")

@app.post("/api/negotiation/respond")
async def process_negotiation_response_endpoint(data: dict):
    """
    AGENTIC AI: Process Negotiation Response
    Handles candidate counter-offers and generates AI response.
    """
    try:
        message = data.get('message')
        state_data = data.get('state')
        
        if not message or not state_data:
            raise HTTPException(status_code=400, detail="message and state are required")
            
        # Reconstruct state
        from negotiator_agent import NegotiationState
        state = NegotiationState(**state_data)
        
        agent = get_negotiator_agent()
        result = await agent.process_candidate_response(state, message)
        
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Negotiation processing failed: {str(e)}")

@app.post("/api/onboarding/generate")
async def generate_onboarding_plan_endpoint(data: dict):
    """
    AGENTIC AI: Generate Onboarding Plan
    Creates a personalized 30-60-90 day plan based on interview insights.
    """
    try:
        candidate_name = data.get('candidate_name')
        role = data.get('role')
        interview_feedback = data.get('interview_feedback', {})
        assessment_results = data.get('assessment_results', {})
        
        if not candidate_name or not role:
            raise HTTPException(status_code=400, detail="candidate_name and role are required")
            
        agent = get_onboarding_agent()
        plan = await agent.generate_plan(candidate_name, role, interview_feedback, assessment_results)
        
        return {"success": True, "plan": plan.__dict__}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Onboarding generation failed: {str(e)}")

@app.post("/api/bias/check")
async def check_bias_endpoint(data: dict):
    """
    AGENTIC AI: Real-time Bias Check
    Analyzes interview questions or feedback for potential bias.
    """
    try:
        text = data.get('text')
        context = data.get('context', '')
        
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
            
        guardian = get_bias_guardian()
        report = await guardian.monitor_interaction(text, context)
        
        return {"success": True, "report": report.__dict__}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bias check failed: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/evaluate-interview")
async def evaluate_interview_endpoint(request: InterviewEvaluationRequest):
    """
    AGENTIC AI: Interview evaluation with LangGraph workflow
    
    Features:
    - 7-step multi-agent evaluation
    - Technical & soft skills analysis with Chain-of-Thought
    - Proctoring integration & behavioral pattern detection
    - RAG storage for continuous learning
    - Comprehensive structured output
    """
    try:
        print(f"🤖 Using InterviewConductorAgent for {request.candidate_name}...")
        print(f"📝 Evaluating {len(request.formatted_input)} Q&A pairs")
        
        # Use agentic AI version
        result = await evaluate_interview_agentic(request)
        
        print(f"✅ Interview evaluation complete")
        print(f"📊 Overall score: {result['structured_evaluation']['interviewResults'][0]['score']}")
        print(f"📈 Status: {result['structured_evaluation']['interviewResults'][0]['status']}")
        
        # Return in format expected by frontend
        return {
            "raw_response": result['raw_response'],
            "token_usage": result['token_usage'],
            "agent_metadata": result['agent_metadata'],
            "analysis_breakdown": result.get('analysis_breakdown')  # Additional details for advanced users
        }
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in evaluate_interview: {str(e)}")
        print(f"ERROR type: {type(e).__name__}")
        print(f"ERROR details: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Interview evaluation failed: {str(e)}")
    
    
    
@app.post("/generate-questions")
async def generate_questions_endpoint(request: QuestionGenerationRequest):
    """
    AGENTIC AI: Question generation with zero repetition
    
    Features:
    - Real-time web search (Tavily API)
    - 85% similarity threshold for uniqueness
    - Current affairs integration (1+ day rule)
    - Multi-language support (English/Hindi)
    - Exam-specific formatting (UPSC, JEE, NEET, SSB, etc.)
    - RAG knowledge base integration
    - Chain-of-Thought question generation
    """
    try:
        print(f"🎯 Question generation request: {request.count} questions for {request.subject}/{request.topic}")
        print(f"📋 Exam: {request.exam_category}, Difficulty: {request.difficulty}, Type: {request.question_type}")
        
        # Use agentic AI version
        result = await generate_questions_agentic(request)
        
        if result['success']:
            print(f"✅ Generated {result['count']}/{request.count} unique questions")
            return {
                "success": True,
                "questions": result['questions'],
                "count": result['count'],
                "token_usage": result['token_usage'],
                "agent_metadata": result['agent_metadata']
            }
        else:
            raise HTTPException(status_code=500, detail="Question generation failed")
            
    except Exception as e:
        print(f"❌ Error generating questions: {str(e)}")
        print(f"ERROR details: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")
    
    
@app.post("/verify-document")
async def verify_document_endpoint(request: DocumentVerificationRequest):
    """
    AGENTIC AI: Document verification with Azure Computer Vision and Face API
    
    Features:
    - Azure Computer Vision OCR extraction
    - Aadhar/PAN validation with regex patterns
    - Azure Face API for photo verification
    - Signature verification using AI
    - Certificate authenticity checks
    - AI-powered fraud detection
    - Chain-of-Thought analysis
    """
    try:
        print(f"📄 Document verification request: {request.documentType} for application {request.applicationId}")
        
        # Use agentic AI version
        result = await verify_document_agentic(request)
        
        if result['success']:
            print(f"✅ Verification complete: {result['status']} (confidence: {result['confidence']:.2%})")
            return result
        else:
            raise HTTPException(status_code=500, detail="Document verification failed")
            
    except Exception as e:
        print(f"❌ Error verifying document: {str(e)}")
        print(f"ERROR details: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Document verification failed: {str(e)}")
    
@app.post("/calculate-ranking")
async def calculate_ranking_endpoint(request: RankingRequest, results_data: List[Dict] = None):
    """
    AGENTIC AI: Ranking calculation with score normalization and AI cutoff prediction
    
    Features:
    - Equipercentile score normalization across multiple sessions
    - Overall and category-wise ranking
    - Percentile calculation
    - AI-powered cutoff prediction using historical data
    - Reservation policy application (India-specific)
    - Merit list generation with tie-breaking rules
    """
    try:
        print(f"🎯 Ranking calculation request for examination: {request.examination_id}")
        
        # If results_data not provided, fetch from database (mock for now)
        if not results_data:
            results_data = []
            # TODO: Fetch from database
            # results_data = await fetch_exam_results(request.examination_id)
        
        if len(results_data) == 0:
            raise HTTPException(status_code=400, detail="No results data provided or found")
        
        # Use agentic AI version
        result = await calculate_ranking_agentic(request, results_data)
        
        if result['success']:
            print(f"✅ Ranking calculated: {len(results_data)} candidates, {result['qualified_candidates']} qualified")
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Ranking calculation failed'))
            
    except Exception as e:
        print(f"❌ Error calculating ranking: {str(e)}")
        print(f"ERROR details: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ranking calculation failed: {str(e)}")


@app.post("/analyze-proctoring")
async def analyze_proctoring_endpoint(data: dict):
    """
    AGENTIC AI: Advanced proctoring with real-time object detection
    
    Features:
    - YOLO v8 object detection (phones, earphones, smartwatches, bluetooth devices)
    - MediaPipe gaze tracking and facial landmarks
    - DeepFace emotion and face recognition
    - Multi-face detection for impersonation prevention
    - Audio anomaly detection (background voices, phone rings)
    - Motion detection between frames
    - AI-powered behavioral pattern analysis
    - LangGraph workflow for decision making
    - Risk scoring and violation tracking
    
    Note: Requires additional computer vision dependencies (deepface, tf-keras, etc.)
    """
    if not PROCTORING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Advanced proctoring is currently unavailable. Missing required dependencies (deepface, tf-keras). Please install: pip install deepface tf-keras torchaudio librosa soundfile"
        )
    
    try:
        candidate_id = data.get('candidate_id', 'unknown')
        session_id = data.get('session_id', 'unknown')
        
        # Decode frame data
        frame_b64 = data.get('frame_data')
        if not frame_b64:
            raise HTTPException(status_code=400, detail="frame_data is required")
        
        frame_data = base64.b64decode(frame_b64)
        
        # Optional audio and previous frame
        audio_data = None
        if data.get('audio_data'):
            audio_data = base64.b64decode(data['audio_data'])
        
        previous_frame = None
        if data.get('previous_frame'):
            previous_frame = base64.b64decode(data['previous_frame'])
        
        screen_activity = data.get('screen_activity')
        
        print(f"🔍 Proctoring analysis for {candidate_id} in session {session_id}")
        
        # Use advanced proctoring agent
        result = await analyze_proctoring_frame(
            candidate_id=candidate_id,
            session_id=session_id,
            frame_data=frame_data,
            audio_data=audio_data,
            previous_frame=previous_frame,
            screen_activity=screen_activity
        )
        
        if result['success']:
            violation_count = len(result['violations'])
            print(f"✅ Proctoring analysis complete: {violation_count} violations, risk score: {result['risk_assessment']['risk_score']:.2f}")
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Proctoring analysis failed'))
            
    except Exception as e:
        print(f"❌ Error in proctoring analysis: {str(e)}")
        print(f"ERROR details: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Proctoring analysis failed: {str(e)}")


@app.post("/interview/start")
async def start_conversational_interview_endpoint(data: dict):
    """
    AGENTIC AI: Start conversational interview session
    
    Features:
    - Multi-phase interview (introduction, technical, behavioral, situational, closing)
    - Context-aware question generation
    - Adaptive difficulty adjustment
    - Real-time relevance checking
    - Skill-level assessment
    - Behavioral pattern detection
    """
    try:
        candidate_id = data.get('candidate_id')
        session_id = data.get('session_id')
        job_title = data.get('job_title')
        job_description = data.get('job_description', '')
        required_skills = data.get('required_skills', [])
        total_questions = data.get('total_questions', 10)
        company_name = data.get('company_name', 'HireGenix') # Default if not provided
        language = data.get('language', 'English')
        
        if not all([candidate_id, session_id, job_title]):
            raise HTTPException(status_code=400, detail="candidate_id, session_id, and job_title are required")
        
        print(f"🎤 Starting conversational interview: {job_title} for {candidate_id} (Language: {language})")
        
        # 1. Build Context (The "Brain")
        context_service = get_context_service()
        # In a real scenario, we would fetch resume bytes from DB/Storage using candidate_id
        # For now, we proceed with available data, context service handles missing resume gracefully
        interview_context = await context_service.build_interview_context(
            candidate_id=candidate_id,
            job_title=job_title,
            job_description=job_description,
            company_name=company_name
        )
        
        # Get conversational agent
        agent = get_conversational_interview_agent()
        
        # Start interview with RICH CONTEXT
        result = await agent.start_interview(
            candidate_id=candidate_id,
            session_id=session_id,
            job_title=job_title,
            job_description=job_description,
            required_skills=required_skills,
            total_questions=total_questions,
            context=interview_context, # Inject the brain
            persona=data.get('persona', 'professional'),
            language=language
        )
        
        if result['success']:
            print(f"✅ Interview started: First question generated")
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Failed to start interview'))
            
    except Exception as e:
        print(f"❌ Error starting interview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")


@app.post("/interview/answer")
async def process_interview_answer_endpoint(data: dict):
    """
    AGENTIC AI: Process candidate's answer and generate next question
    
    Features:
    - Real-time answer relevance checking
    - Answer quality analysis (0-10 scale)
    - Skill-level assessment per answer
    - Behavioral pattern detection (Deep Sensing)
    - Adaptive difficulty adjustment
    - Warning generation for problematic answers
    - Comprehensive interview summary on completion
    """
    try:
        session_id = data.get('session_id')
        answer = data.get('answer')
        state_data = data.get('state')
        signals = data.get('signals') # New: Deep Sensing signals
        
        if not all([session_id, answer, state_data]):
            raise HTTPException(status_code=400, detail="session_id, answer, and state are required")
        
        print(f"💬 Processing answer for session {session_id}")
        
        # Reconstruct state
        from conversational_interview_agent import ConversationalInterviewState
        state = ConversationalInterviewState(**state_data)
        
        # Get conversational agent
        agent = get_conversational_interview_agent()
        
        # Process answer with signals
        result = await agent.process_answer(
            session_id=session_id,
            answer=answer,
            state=state,
            signals=signals
        )
        
        if result['success']:
            if result.get('completed'):
                print(f"✅ Interview completed with rating: {result['summary']['overall_rating']}/10")
            else:
                print(f"✅ Answer processed, next question generated")
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Failed to process answer'))
            
    except Exception as e:
        print(f"❌ Error processing answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")

    
    # Database configuration

DB_USER = config('DB_USER', default=os.getenv("DB_USER", "postgres"))
DB_PASSWORD = config('DB_PASSWORD', default=os.getenv("DB_PASSWORD", ""))
DB_HOST = config('DB_HOST', default=os.getenv("DB_HOST", "localhost"))
DB_PORT = config('DB_PORT', default=os.getenv("DB_PORT", "5432"))
DB_NAME = config('DB_NAME', default=os.getenv("DB_NAME", "HireGenix-Latest"))

# Database connection function
def get_db_connection():
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME
    )
    conn.autocommit = True  # Set to False if you want to manage transactions manually
    return conn

# Check database connection and create tables if needed
def setup_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create messages table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR(255) PRIMARY KEY,
                sender_id VARCHAR(255) NOT NULL,
                receiver_id VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                job_id VARCHAR(255)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sender_id ON messages (sender_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_receiver_id ON messages (receiver_id)")
        
        cursor.close()
        conn.close()
        print("Database setup completed successfully")
        return True
    except Exception as e:
        print(f"Database setup error: {e}")
        return False

# Pydantic models for API
class MessageCreate(BaseModel):
    sender_id: str
    receiver_id: str
    content: str
    job_id: Optional[str] = None

class MessageResponse(BaseModel):
    id: str
    sender_id: str
    receiver_id: str
    content: str
    timestamp: datetime
    job_id: Optional[str] = None

class ContactResponse(BaseModel):
    id: str
    name: str
    avatar: Optional[str] = None
    lastMessage: str
    timestamp: datetime
    unread: int
    type: str
    userId: str
    jobId: Optional[str] = None
    jobTitle: str

# Initialize Socket.IO server
sio = socketio.AsyncServer(
    cors_allowed_origins=[
        "https://myhiregenix.ai",
        "https://api.myhiregenix.ai",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    async_mode="asgi",
    logger=True,
    engineio_logger=True
)

# Add CORS middleware

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Store connected users { user_id: socket_id }
user_to_socket = {}
socket_to_user = {}

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def join_room(sid, data):
    user_id = data['user_id']

    # Clean up existing connection
    if user_id in user_to_socket:
        old_sid = user_to_socket[user_id]
        if old_sid in socket_to_user:
            del socket_to_user[old_sid]
        del user_to_socket[user_id]
        print(f"Cleaned up old connection for user {user_id}")

    user_to_socket[user_id] = sid
    socket_to_user[sid] = user_id

    print(f"User {user_id} joined with socket {sid}")
    print(f"Current users: {user_to_socket}")

@sio.event
async def send_message(sid, data):
    # Ensure messages table exists before inserting
    setup_database()
    print(f"Message data: {data}")

    sender_id = socket_to_user.get(sid)
    print(f"Sender ID: {sender_id}")

    if not sender_id:
        print(f"No user found for socket {sid}")
        return

    receiver_id = data['to']
    message_content = data['message']
    job_id = data.get('jobId')
    message_id = str(uuid4())
    timestamp = datetime.now()

    # Save to database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (id, sender_id, receiver_id, content, timestamp, job_id) VALUES (%s, %s, %s, %s, %s, %s)",
            (message_id, sender_id, receiver_id, message_content, timestamp, job_id)
        )
        cursor.close()
        conn.close()

        # Emit to receiver
        if receiver_id in user_to_socket:
            receiver_sid = user_to_socket[receiver_id]
            await sio.emit('receive_message', {
                'from': sender_id,
                'message': message_content,
                'timestamp': timestamp.isoformat(),
                'jobId': job_id
            }, to=receiver_sid)
            print(f"Message sent to {receiver_id} at socket {receiver_sid}")
        else:
            print(f"Receiver {receiver_id} not connected")

        # Confirm to sender
        await sio.emit('message_sent', {
            'to': receiver_id,
            'message': message_content,
            'timestamp': timestamp.isoformat(),
            'delivered': receiver_id in user_to_socket,
            'jobId': job_id
        }, to=sid)
    except Exception as e:
        print(f"Error saving message: {e}")
        await sio.emit('message_error', {'error': str(e)}, to=sid)

@sio.event
async def disconnect(sid):
    if sid in socket_to_user:
        user_id = socket_to_user[sid]
        del socket_to_user[sid]
        if user_id in user_to_socket and user_to_socket[user_id] == sid:
            del user_to_socket[user_id]
        print(f"User {user_id} disconnected from socket {sid}")
    else:
        print(f"Unknown socket {sid} disconnected")

# API endpoint to emit agent stage updates
@app.post("/emit-agent-stage")
async def emit_agent_stage(data: dict):
    """
    Emit agent stage updates to connected clients via Socket.IO
    Used by Next.js frontend to broadcast agent processing progress
    """
    try:
        user_id = data.get('userId')
        company_id = data.get('companyId')
        stage = data.get('stage')
        status = data.get('status')
        message = data.get('message', '')
        progress = data.get('progress', 0)
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        event_data = {
            'stage': stage,
            'status': status,
            'message': message,
            'progress': progress,
            'timestamp': timestamp
        }
        
        # Emit to user if connected
        if user_id and user_id in user_to_socket:
            user_sid = user_to_socket[user_id]
            await sio.emit('agent-stage-update', event_data, to=user_sid)
            print(f"✅ Emitted agent stage update to user {user_id}: {stage} - {status}")
        
        # Also emit to company room (broadcast to all connected users of that company)
        if company_id:
            await sio.emit('agent-stage-update', event_data, room=f"company-{company_id}")
            print(f"✅ Emitted agent stage update to company {company_id}: {stage} - {status}")
        
        return {"success": True, "message": "Stage update emitted"}
        
    except Exception as e:
        print(f"❌ Error emitting agent stage: {str(e)}")
        return {"success": False, "error": str(e)}

# API endpoint to fetch message history
@app.get("/messages/{user_id}/{contact_id}", response_model=List[MessageResponse])
async def get_messages(user_id: str, contact_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute("""
            SELECT * FROM messages 
            WHERE (sender_id = %s AND receiver_id = %s) OR (sender_id = %s AND receiver_id = %s)
            ORDER BY timestamp ASC
        """, (user_id, contact_id, contact_id, user_id))
        
        messages = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [
            MessageResponse(
                id=msg['id'],
                sender_id=msg['sender_id'],
                receiver_id=msg['receiver_id'],
                content=msg['content'],
                timestamp=msg['timestamp'],
                job_id=msg['job_id']
            ) for msg in messages
        ]
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return []

# API endpoint to fetch contacts for a user
@app.get("/contacts/{user_id}", response_model=List[ContactResponse])
async def get_contacts(user_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # Get unique message threads with most recent message
        cursor.execute("""
            WITH ranked_messages AS (
                SELECT 
                    CASE 
                        WHEN sender_id = %s THEN receiver_id 
                        ELSE sender_id 
                    END AS contact_id,
                    content,
                    timestamp,
                    job_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY 
                            CASE 
                                WHEN sender_id = %s THEN receiver_id 
                                ELSE sender_id 
                            END,
                            COALESCE(job_id, 'no-job')
                        ORDER BY timestamp DESC
                    ) AS rn
                FROM messages
                WHERE sender_id = %s OR receiver_id = %s
            )
            SELECT 
                contact_id,
                content AS last_message,
                timestamp AS latest_timestamp,
                job_id
            FROM ranked_messages
            WHERE rn = 1
            ORDER BY latest_timestamp DESC
        """, (user_id, user_id, user_id, user_id))

        message_threads = cursor.fetchall()
        
        if not message_threads:
            cursor.close()
            conn.close()
            return []

        # Prepare IDs
        candidate_ids = [str(thread['contact_id']) for thread in message_threads]
        job_ids = [str(thread['job_id']) for thread in message_threads if thread['job_id']]

        # Fetch candidate details
        candidates_data = {}
        if candidate_ids:
            cursor.execute("""
                SELECT 
                    c.id, 
                    c.name, 
                    c."lastName", 
                  
                    c.email,
                    c.phone,
                    j.id as job_id,
                    j.title as job_title,
                    co.name as company_name
                FROM public."Candidate" c
                LEFT JOIN public."Job" j ON c."jobId" = j.id
                LEFT JOIN public."Company" co ON j."companyId" = co.id
                WHERE c.id = ANY(%s)
            """, (candidate_ids,))
            candidates_data = {str(row['id']): row for row in cursor.fetchall()}

        # Fetch job details
        job_details = {}
        if job_ids:
            cursor.execute("""
                SELECT 
                    j.id,
                    j.title,
                    co.name as company_name
                FROM public."Job" j
                LEFT JOIN public."Company" co ON j."companyId" = co.id
                WHERE j.id = ANY(%s)
            """, (job_ids,))
            job_details = {str(row['id']): row for row in cursor.fetchall()}

        cursor.close()
        conn.close()

        # Build response with unique IDs
        response = []
        for thread in message_threads:
            candidate_id = str(thread['contact_id'])
            candidate = candidates_data.get(candidate_id, {})
            job_id = str(thread['job_id']) if thread['job_id'] else None

            # Create unique contact ID
            contact_id = f"candidate-{candidate_id}-{job_id or 'no-job'}-{int(thread['latest_timestamp'].timestamp())}"

            full_name = f"{candidate.get('name', '')} {candidate.get('lastName', '')}".strip()
            if not full_name:
                full_name = candidate.get('email', f"Candidate {candidate_id}")

            job_info = job_details.get(job_id) if job_id else None
            job_title = job_info.get('title') if job_info else candidate.get('job_title', 'General Inquiry')

            response.append(ContactResponse(
                id=contact_id,
                name=full_name,
                avatar=candidate.get('avatar'),
                lastMessage=thread['last_message'],
                timestamp=thread['latest_timestamp'],
                unread=0,
                type="candidate",
                userId=candidate_id,
                jobId=job_id,
                jobTitle=job_title,
                email=candidate.get('email'),
                phone=candidate.get('phone'),
                companyName=job_info.get('company_name') if job_info else candidate.get('company_name')
            ))

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching contacts: {str(e)}")
@app.on_event("startup")
async def startup_event():
    # Setup database on startup
    if not setup_database():
        print("Warning: Database setup failed")






# Computer Vision initialization
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# FaceMesh is now lazy-loaded via _get_face_mesh() function
# This avoids MediaPipe initialization errors at startup

# Lazy YOLO model initialization
_yolo_model = None
_yolo_init_attempted = False

def _get_yolo_model():
    """Lazy load YOLO model with proper error handling"""
    global _yolo_model, _yolo_init_attempted
    if _yolo_model is not None:
        return _yolo_model
    if _yolo_init_attempted:
        return None
    _yolo_init_attempted = True
    try:
        print("🔄 Loading YOLO model...")
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded successfully")
        return _yolo_model
    except Exception as e:
        print(f"⚠️ YOLO model unavailable: {e}")
        return None

def detect_smartphone(frame):
    """Detect smartphones using YOLO model"""
    try:
        model = _get_yolo_model()
        if model is None:
            print("⚠️ YOLO model not available, skipping phone detection")
            return False, None
        results = model(frame)[0]
        phone_detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            if label in ["cell phone", "mobile phone", "phone"]:  # Depending on YOLO class names
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                phone_detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": (x1, y1, x2 - x1, y2 - y1)
                })

        # Sort by confidence
        phone_detections.sort(key=lambda x: x['confidence'], reverse=True)

        if phone_detections:
            best = phone_detections[0]
            print(f"✅ SUCCESS: Phone detected by YOLO! Box: {best['bbox']}")
            return True, best['bbox']
        print("📵 No phone detected by YOLO")
        return False, None
    except Exception as e:
        print(f"Error in detect_smartphone: {str(e)}")
        return False, None

def detect_objects(frame):
    """Detect objects (specifically smartphones) using YOLO"""
    detected_objects = []
    try:
        phone_detected, phone_box = detect_smartphone(frame)
        if phone_detected:
            detected_objects.append("smartphone")
            print(f"📱 PHONE DETECTED! Box: {phone_box}")
        else:
            print("📵 No objects detected")
        return detected_objects, phone_box
    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")
        return detected_objects, None

def detect_gaze(frame):
    """Detect gaze direction using MediaPipe FaceMesh (lazy loaded)"""
    face_mesh = _get_face_mesh()
    if face_mesh is None:
        return "unknown"
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                left_eye = np.mean([(face_landmarks.landmark[362].x * w, face_landmarks.landmark[362].y * h),
                                    (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h)], axis=0)
                right_eye = np.mean([(face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
                                     (face_landmarks.landmark[133].x * w, face_landmarks.landmark[133].y * h)], axis=0)
                nose_tip = (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h)
                
                eye_center_y = (left_eye[1] + right_eye[1]) / 2
                if eye_center_y > nose_tip[1] + 20:
                    print("Gaze: Downward (suspicious)")
                    return "downward"
                print("Gaze: Forward")
                return "forward"
        print("Gaze: No face detected")
        return "unknown"
    except Exception as e:
        print(f"Error in detect_gaze: {str(e)}")
        return "unknown"

def estimate_distance(box_width, focal_length=700, known_width=7.0):
    """Estimate distance (cm) from camera based on box width"""
    if box_width == 0:
        return float('inf')
    return (known_width * focal_length) / box_width

@app.post("/process-frame")
async def process_frame(data: dict):
    try:
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Failed to decode image. Check base64 encoding.")
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        cv2.imwrite("input_frame.jpg", frame)
        print(f"Input frame saved as input_frame.jpg, shape: {frame.shape}")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        print(f"Detected {len(faces)} faces")
        
        gaze_direction = detect_gaze(frame)
        
        detected_objects, phone_box = detect_objects(frame)
        
        phone_distance = None
        if phone_box:
            _, _, w, _ = phone_box
            phone_distance = estimate_distance(w, focal_length=700, known_width=7.0)
            print(f"Estimated phone distance: {phone_distance:.2f} cm")
        
        suspicious_activity = len(detected_objects) > 0 and (phone_distance is None or phone_distance < 100) or gaze_direction == "downward"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "face_count": len(faces),
            "emotions": {
                "positive": random.randint(20, 60),
                "neutral": random.randint(20, 60),
                "negative": random.randint(0, 20)
            },
            "detected_objects": detected_objects,
            "gaze_direction": gaze_direction,
            "phone_distance_cm": round(phone_distance, 2) if phone_distance else None,
            "analysis": {
                "attention_score": 80 if gaze_direction == "forward" else 20,
                "suspicious_activity": suspicious_activity
            }
        }
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    # Ensure database is set up before starting the server
    setup_database()
    uvicorn.run(socket_app, host='0.0.0.0', port=8000)
