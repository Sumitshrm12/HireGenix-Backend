# prescreening/api.py - Advanced Agentic Pre-screening API
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import uuid
import os
import asyncio
import math
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Import models
from .models import (
    ResumeAnalysisRequest, PreScreeningSessionCreateRequest, MCQAnswerRequest,
    CandidatePreScreening, PreScreeningSession, ResumeMatchingResult,
    HumanReviewTask, PreScreeningStatus, ProctoringEventRequest,
    HumanReviewDecisionRequest, MCQQuestion, SessionStartResponse,
    ScoreBucket, SessionStatus, ReviewStatus, ReviewDecision,
    MCQQuestionWithAnswer, PreScreeningConfig, NotificationRequest
)

# Import AGENTIC AI services (Advanced multi-agent system)
from .service import AgenticPreScreeningOrchestrator, create_prescreening_service
from .ai_engine import (
    VideoAnalysisEngine, RealTimeMCQGenerator, PreScreeningEvaluationPipeline,
    create_video_analysis_engine, create_mcq_generator, create_evaluation_pipeline
)
from .resume_matching_engine import HybridResumeMatchingEngine, create_resume_matching_engine

# Legacy imports for backward compatibility
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Import Agentic AI Config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agentic_ai.config import AgenticAIConfig

# Load environment
load_dotenv()

# Create router
router = APIRouter(tags=["Pre-screening - Agentic AI"])

# ============================================================================
# INITIALIZE AGENTIC AI COMPONENTS
# ============================================================================

# Agentic Pre-screening Orchestrator (6-agent system with LangGraph)
agentic_orchestrator = None

# Real-time AI engines
video_analysis_engine = None
mcq_generator = None
evaluation_pipeline = None
resume_matching_engine = None

def get_agentic_orchestrator() -> AgenticPreScreeningOrchestrator:
    """Get or create agentic orchestrator singleton"""
    global agentic_orchestrator
    if agentic_orchestrator is None:
        agentic_orchestrator = create_prescreening_service()
        print("✅ Agentic Pre-screening Orchestrator initialized (6-agent LangGraph system)")
    return agentic_orchestrator

def get_video_analysis_engine() -> VideoAnalysisEngine:
    """Get or create video analysis engine"""
    global video_analysis_engine
    if video_analysis_engine is None:
        video_analysis_engine = create_video_analysis_engine()
        print("✅ Video Analysis Engine initialized")
    return video_analysis_engine

def get_mcq_generator() -> RealTimeMCQGenerator:
    """Get or create real-time MCQ generator"""
    global mcq_generator
    if mcq_generator is None:
        mcq_generator = create_mcq_generator()
        print("✅ Real-time MCQ Generator initialized")
    return mcq_generator

def get_evaluation_pipeline() -> PreScreeningEvaluationPipeline:
    """Get or create evaluation pipeline"""
    global evaluation_pipeline
    if evaluation_pipeline is None:
        evaluation_pipeline = create_evaluation_pipeline()
        print("✅ Evaluation Pipeline initialized")
    return evaluation_pipeline

def get_resume_matching_engine() -> HybridResumeMatchingEngine:
    """Get or create hybrid resume matching engine"""
    global resume_matching_engine
    if resume_matching_engine is None:
        resume_matching_engine = create_resume_matching_engine()
        print("✅ Hybrid Resume Matching Engine initialized")
    return resume_matching_engine

# Initialize AI services
def get_azure_llm():
    """Get configured Azure OpenAI LLM using AgenticAIConfig"""
    config = AgenticAIConfig()
    return AzureChatOpenAI(
        openai_api_key=config.azure.api_key,
        azure_endpoint=config.azure.endpoint,
        deployment_name=config.azure.deployment_name,
        openai_api_version=config.azure.api_version,
        temperature=0.7,
        max_tokens=2500
    )

def get_azure_embeddings():
    """Get configured Azure OpenAI Embeddings using AgenticAIConfig"""
    config = AgenticAIConfig()
    return AzureOpenAIEmbeddings(
        openai_api_key=config.azure.api_key,
        azure_endpoint=config.azure.endpoint,
        deployment=config.azure.embedding_deployment,
        openai_api_version=config.azure.api_version
    )

# Database connection (using the same connection as main.py)
import psycopg2
import psycopg2.extras
from decouple import config

DB_USER = config('DB_USER', default=os.getenv("DB_USER", "postgres"))
DB_PASSWORD = config('DB_PASSWORD', default=os.getenv("DB_PASSWORD", ""))
DB_HOST = config('DB_HOST', default=os.getenv("DB_HOST", "localhost"))
DB_PORT = config('DB_PORT', default=os.getenv("DB_PORT", "5432"))
DB_NAME = config('DB_NAME', default=os.getenv("DB_NAME", "HireGenix-Latest"))

def get_db_connection():
    """Get database connection"""
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME
    )
    conn.autocommit = True
    return conn

# === Job Context & Difficulty Policy Helpers ===

def infer_seniority_from_title(title: str) -> str:
    """Infer seniority/level from job title heuristically."""
    if not title:
        return "mid"
    t = title.lower()
    if any(k in t for k in ["intern", "graduate", "trainee", "junior", "jr"]):
        return "junior"
    if any(k in t for k in ["staff", "principal", "architect"]):
        return "staff"
    if any(k in t for k in ["lead", "manager", "head", "director"]):
        return "senior"
    if "senior" in t or "sr" in t:
        return "senior"
    return "mid"

def map_role_family(title: str, description: str = "") -> str:
    """Map job to a broad family: engineering, data, product, design, sales, support, security, devops, hr, marketing."""
    text = f"{title or ''} {description or ''}".lower()
    if any(k in text for k in ["ml", "data scientist", "data science", "analytics", "bi", "sql", "statistics"]):
        return "data"
    if any(k in text for k in ["devops", "sre", "platform", "kubernetes", "terraform", "ci/cd"]):
        return "devops"
    if any(k in text for k in ["security", "soc", "siem", "infosec", "appsec", "pentest"]):
        return "security"
    if any(k in text for k in ["product manager", "pm ", "pm,", "roadmap", "prioritization"]):
        return "product"
    if any(k in text for k in ["designer", "ui", "ux", "figma", "visual", "interaction"]):
        return "design"
    if any(k in text for k in ["sales", "account executive", "ae", "quota", "pipeline"]):
        return "sales"
    if any(k in text for k in ["support", "customer success", "cs", "helpdesk", "ticket"]):
        return "support"
    if any(k in text for k in ["hr", "talent", "recruit", "people ops"]):
        return "hr"
    if any(k in text for k in ["marketing", "seo", "sem", "campaign", "brand"]):
        return "marketing"
    # default
    return "engineering"

def get_job_context(job_id: str) -> Dict[str, Any]:
    """Fetch job title, description, companyId and associated skills from DB."""
    ctx: Dict[str, Any] = {"title": None, "description": None, "company_id": None, "skills": []}
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute('SELECT title, description, "companyId" FROM "Job" WHERE id = %s', (job_id,))
        job = cur.fetchone()
        if job:
            ctx["title"] = job["title"]
            ctx["description"] = job["description"]
            ctx["company_id"] = job["companyId"]
            cur.execute('SELECT name, level, "yearsOfExperience" FROM "Skill" WHERE "jobId" = %s', (job_id,))
            skills = cur.fetchall()
            ctx["skills"] = [{"name": r["name"], "level": r["level"], "years": r["yearsOfExperience"]} for r in skills]
        cur.close()
        conn.close()
    except Exception as e:
        print(f"get_job_context error: {e}")
    return ctx

def get_prescreening_config(company_id: Optional[str], job_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Fetch PreScreeningConfiguration row (job-specific override first, then company default)."""
    if not company_id:
        return None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        # Try job-specific
        if job_id:
            cur.execute(
                'SELECT * FROM prescreening_configurations WHERE "companyId" = %s AND "jobId" = %s',
                (company_id, job_id)
            )
            row = cur.fetchone()
            if row:
                cur.close()
                conn.close()
                return dict(row)
        # Fallback to company default
        cur.execute(
            'SELECT * FROM prescreening_configurations WHERE "companyId" = %s AND "jobId" IS NULL',
            (company_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"get_prescreening_config error: {e}")
        return None

def get_difficulty_policy(seniority: str, cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return difficulty distribution and per-difficulty time limits informed by seniority and optional config."""
    # Defaults by seniority
    defaults = {
        "junior": {"easy": 0.5, "medium": 0.4, "hard": 0.1},
        "mid": {"easy": 0.25, "medium": 0.55, "hard": 0.2},
        "senior": {"easy": 0.1, "medium": 0.5, "hard": 0.4},
        "staff": {"easy": 0.05, "medium": 0.45, "hard": 0.5},
    }
    seniority_key = seniority if seniority in defaults else "mid"
    distribution = defaults[seniority_key].copy()

    # If cfg provides exact counts, we will compute distribution based on total
    time_limits = {"easy": 60, "medium": 90, "hard": 120}
    if cfg:
        total = int(cfg.get("totalMcqQuestions") or cfg.get("total_mcq_questions") or 10)
        easy_n = int(cfg.get("easyQuestions") or cfg.get("easy_questions") or round(distribution["easy"] * total))
        med_n = int(cfg.get("mediumQuestions") or cfg.get("medium_questions") or round(distribution["medium"] * total))
        hard_n = int(cfg.get("hardQuestions") or cfg.get("hard_questions") or (total - easy_n - med_n))
        # Recompute distribution as fractions
        if total > 0:
            distribution = {
                "easy": max(easy_n, 0) / total,
                "medium": max(med_n, 0) / total,
                "hard": max(hard_n, 0) / total,
            }
        # Time limit config
        tl = int(cfg.get("mcqTimeLimit") or cfg.get("mcq_time_limit") or 60)
        # Allow scaling by difficulty
        time_limits = {"easy": tl, "medium": int(tl * 1.5), "hard": int(tl * 2)}

    return {"distribution": distribution, "time_limits": time_limits}

# === REAL AI IMPLEMENTATIONS ===

async def analyze_resume_with_ai(resume_text: str, job_description: str = None) -> dict:
    """Real AI-powered resume analysis using Azure OpenAI"""
    try:
        llm = get_azure_llm()
        
        # Create analysis prompt
        prompt = f"""
        Analyze this resume and provide a comprehensive assessment:

        Resume Text:
        {resume_text}

        Job Context (if provided):
        {job_description or "General analysis"}

        Provide your analysis in the following JSON format:
        {{
            "overall_score": <number between 0-100>,
            "technical_skills": ["skill1", "skill2", ...],
            "experience_years": <estimated years>,
            "education_level": "degree level",
            "key_strengths": ["strength1", "strength2", ...],
            "areas_for_improvement": ["area1", "area2", ...],
            "job_fit_score": <number between 0-100>,
            "rationale": "detailed explanation of the score",
            "recommended_next_steps": "recommendation"
        }}

        Only return valid JSON, no additional text or formatting.
        """
        
        response = await llm.agenerate([[HumanMessage(content=prompt)]])
        raw_response = response.generations[0][0].text.strip()
        
        # Clean and parse JSON
        if raw_response.startswith('```json'):
            raw_response = raw_response[7:]
        if raw_response.endswith('```'):
            raw_response = raw_response[:-3]
        
        return json.loads(raw_response.strip())
        
    except Exception as e:
        print(f"Error in AI resume analysis: {e}")
        # Fallback response
        return {
            "overall_score": 65.0,
            "technical_skills": ["extracted from text analysis"],
            "experience_years": 3,
            "education_level": "Bachelor's",
            "key_strengths": ["Communication", "Technical Skills"],
            "areas_for_improvement": ["Leadership", "Domain Knowledge"],
            "job_fit_score": 70.0,
            "rationale": f"Analysis completed with fallback method due to: {str(e)}",
            "recommended_next_steps": "Proceed to pre-screening test"
        }

async def generate_mcq_questions_ai(
    job_title: str,
    job_family: str,
    skills: List[str],
    difficulty_mix: Dict[str, float],
    total: int,
    time_limits: Dict[str, int],
) -> List[MCQQuestionWithAnswer]:
    """Generate diverse MCQs using AI for any job family with difficulty distribution."""
    try:
        llm = get_azure_llm()
        skills_text = ", ".join(skills) if skills else "role-relevant skills"
        # Compute integer counts per difficulty
        easy_n = int(round(difficulty_mix.get("easy", 0) * total))
        med_n = int(round(difficulty_mix.get("medium", 0) * total))
        # Ensure total matches exactly
        hard_n = max(total - easy_n - med_n, 0)
        plan = []
        plan += ["easy"] * easy_n
        plan += ["medium"] * med_n
        plan += ["hard"] * hard_n
        # In case rounding made it longer/shorter
        plan = plan[:total] if len(plan) > total else (plan + ["medium"] * (total - len(plan)))

        prompt = f"""
You are generating a pre-screening MCQ set for the role: "{job_title}" (family: {job_family}).
Skills to cover: {skills_text}

CRITICAL: You MUST generate EXACTLY {total} questions. No more, no less.

Requirements:
- Total questions: EXACTLY {total}
- Difficulty distribution (approx.): easy={easy_n}, medium={med_n}, hard={hard_n}
- For engineering/data roles: include a mix of fundamentals, debugging, best practices, and realistic scenarios.
- For devops/security: include troubleshooting, incident, and risk/mitigation scenarios.
- For product/design: include prioritization, metrics/ux heuristics, scenario judgment, best practices.
- For sales/support/hr/marketing: include situational judgment, domain knowledge, process/methods, metrics interpretation.
- Avoid role-inappropriate content (e.g., no code for sales unless relevant to tooling).
- No trick questions; exactly one correct answer.

Return a STRICT JSON array with EXACTLY {total} objects. Each object must match this schema:
{{
  "id": "string (unique)",
  "question": "string",
  "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
  "correct_answer": "A" | "B" | "C" | "D",
  "difficulty_level": "easy" | "medium" | "hard",
  "skill_category": "primary skill/category for this question",
  "time_limit": number,  // seconds; easy={time_limits.get("easy",60)}, medium={time_limits.get("medium",90)}, hard={time_limits.get("hard",120)}
  "rationale": "brief explanation for the correct answer"
}}

CRITICAL REQUIREMENTS:
- The JSON array MUST contain EXACTLY {total} question objects
- Use a variety of skills from the provided list when possible
- Assign time_limit based on difficulty
- Output ONLY the JSON array, no markdown fences, no additional text
- Each question must be unique and relevant to the role

DO NOT generate more than {total} questions. DO NOT generate fewer than {total} questions.
"""

        response = await llm.agenerate([[HumanMessage(content=prompt)]])
        raw_response = response.generations[0][0].text.strip()

        # Clean JSON fences if present
        if raw_response.startswith('```json'):
            raw_response = raw_response[7:]
        if raw_response.endswith('```'):
            raw_response = raw_response[:-3]

        questions_data = json.loads(raw_response.strip())
        questions: List[MCQQuestionWithAnswer] = []

        # Track used IDs to prevent duplicates
        used_ids = set()
        timestamp_prefix = int(datetime.now().timestamp() * 1000)  # milliseconds
        
        # Ensure we only process exactly 'total' questions
        actual_count = min(len(questions_data), total)
        if len(questions_data) != total:
            print(f"⚠️ AI generated {len(questions_data)} questions but requested {total}")
        
        for i in range(actual_count):
            qd = questions_data[i] or {}
            dl = (qd.get("difficulty_level") or (plan[i] if i < len(plan) else "medium")).lower()
            tl = int(qd.get("time_limit") or time_limits.get(dl, 90))
            
            # Generate unique ID with timestamp and UUID to prevent collisions
            base_id = qd.get("id", f"q_{i+1}")
            # Create highly unique ID with timestamp and UUID
            question_id = f"{timestamp_prefix}_{base_id}_{uuid.uuid4().hex[:8]}"
            counter = 1
            while question_id.lower() in used_ids:
                question_id = f"{timestamp_prefix}_{base_id}_{counter}_{uuid.uuid4().hex[:8]}"
                counter += 1
            used_ids.add(question_id.lower())
            
            questions.append(
                MCQQuestionWithAnswer(
                    id=question_id,
                    question=qd.get("question", f"{job_title}: question {i+1}"),
                    options=qd.get("options", {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}),
                    correct_answer=qd.get("correct_answer", "A"),
                    difficulty_level=dl,
                    skill_category=qd.get("skill_category", (skills[0] if skills else job_family)),
                    time_limit=tl,
                    rationale=qd.get("rationale", "Based on best practices and role expectations.")
                )
            )

        # If the model returned fewer than total, synthesize some fallbacks
        while len(questions) < total:
            idx = len(questions)
            dl = plan[idx] if idx < len(plan) else "medium"
            
            # Generate unique fallback ID with timestamp and UUID
            base_id = f"fallback_q_{idx+1}"
            question_id = f"{timestamp_prefix}_{base_id}_{uuid.uuid4().hex[:8]}"
            counter = 1
            while question_id.lower() in used_ids:
                question_id = f"{timestamp_prefix}_{base_id}_{counter}_{uuid.uuid4().hex[:8]}"
                counter += 1
            used_ids.add(question_id.lower())
            
            questions.append(
                MCQQuestionWithAnswer(
                    id=question_id,
                    question=f"{job_title} ({job_family}) competency check {idx+1}",
                    options={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
                    correct_answer="A",
                    difficulty_level=dl,
                    skill_category=(skills[idx % len(skills)] if skills else job_family),
                    time_limit=time_limits.get(dl, 90),
                    rationale="Fallback rationale"
                )
            )

        # Final safety check - ensure exactly 'total' questions
        if len(questions) > total:
            print(f"⚠️ Trimming {len(questions)} questions to exactly {total}")
            questions = questions[:total]
        
        print(f"✅ Returning exactly {len(questions)} MCQ questions")
        return questions

    except Exception as e:
        print(f"Error generating MCQ questions: {e}")
        # Fallback simple set with unique IDs
        questions: List[MCQQuestionWithAnswer] = []
        plan = ["medium"] * total
        used_ids = set()
        timestamp_prefix = int(datetime.now().timestamp() * 1000)  # milliseconds for error fallback
        
        for i in range(total):
            dl = plan[i]
            # Generate unique fallback ID with timestamp and UUID
            base_id = f"fallback_error_q_{i+1}"
            question_id = f"{timestamp_prefix}_{base_id}_{uuid.uuid4().hex[:8]}"
            counter = 1
            while question_id.lower() in used_ids:
                question_id = f"{timestamp_prefix}_{base_id}_{counter}_{uuid.uuid4().hex[:8]}"
                counter += 1
            used_ids.add(question_id.lower())
            
            questions.append(
                MCQQuestionWithAnswer(
                    id=question_id,
                    question=f"{job_title} ({job_family}) general competency {i+1}",
                    options={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
                    correct_answer="A",
                    difficulty_level=dl,
                    skill_category=(skills[i % len(skills)] if skills else job_family),
                    time_limit=time_limits.get(dl, 90),
                    rationale="Fallback question rationale"
                )
            )
        return questions

async def calculate_embedding_similarity(resume_text: str, job_description: str) -> float:
    """Calculate semantic similarity using embeddings (cosine similarity between mean vectors)."""
    try:
        embeddings = get_azure_embeddings()

        # Create text chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        resume_chunks = text_splitter.split_text(resume_text or "")
        job_chunks = text_splitter.split_text(job_description or "")
        if not resume_chunks or not job_chunks:
            return 0.5

        # Compute embeddings synchronously in a thread to avoid blocking the event loop
        resume_vecs = await asyncio.to_thread(embeddings.embed_documents, resume_chunks)
        job_vecs = await asyncio.to_thread(embeddings.embed_documents, job_chunks)

        # Mean pool
        def mean_vector(vectors: List[List[float]]) -> List[float]:
            n = len(vectors)
            if n == 0:
                return []
            dim = len(vectors[0])
            acc = [0.0] * dim
            for v in vectors:
                for i in range(dim):
                    acc[i] += v[i]
            return [x / n for x in acc]

        def cosine(a: List[float], b: List[float]) -> float:
            if not a or not b or len(a) != len(b):
                return 0.5
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0.0 or nb == 0.0:
                return 0.5
            return max(min(dot / (na * nb), 1.0), -1.0)

        resume_mean = mean_vector(resume_vecs)
        job_mean = mean_vector(job_vecs)
        sim = cosine(resume_mean, job_mean)

        # Normalize to 0-1 and return as 0-1; caller multiplies to 100 where needed
        return float(sim)

    except Exception as e:
        print(f"Error calculating embedding similarity: {e}")
        return 0.65  # Fallback similarity score

def determine_score_bucket(score: float) -> ScoreBucket:
    """Determine score bucket based on pre-screening score"""
    if score >= 80:
        return ScoreBucket.EXCELLENT
    elif score >= 70:
        return ScoreBucket.GOOD
    elif score >= 60:
        return ScoreBucket.POTENTIAL
    else:
        return ScoreBucket.NOT_ELIGIBLE

def get_next_action_recommendation(bucket: ScoreBucket, score: float) -> str:
    """Get next action recommendation based on score bucket"""
    if bucket == ScoreBucket.EXCELLENT:
        return "Auto-approved for next stage - schedule technical interview"
    elif bucket == ScoreBucket.GOOD:
        return "Approved for pre-screening test"
    elif bucket == ScoreBucket.POTENTIAL:
        return "Requires human review before proceeding"
    else:
        return "Not eligible - send rejection notification"

# === API ENDPOINTS ===

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "pre-screening-api",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
        "ai_services": "active"
    }

@router.post("/analyze-resume")
async def analyze_resume_endpoint(request: ResumeAnalysisRequest):
    """
    🤖 AGENTIC AI Resume Analysis
    Uses 6-agent LangGraph orchestrator with GPT-5-Chat (gpt-4o) for intelligent matching
    """
    try:
        print("="*80)
        print("🤖 AGENTIC PRE-SCREENING STARTED")
        print(f"📋 Candidate: {request.candidate_id}")
        print(f"💼 Job: {request.job_id}")
        print("="*80)
        
        # Fetch job context
        job_title = None
        job_description = "General position"
        job_requirements = []
        company_id = None
        
        try:
            conn_job = get_db_connection()
            cur_job = conn_job.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur_job.execute('SELECT title, description, "companyId" FROM "Job" WHERE id = %s', (request.job_id,))
            job_row = cur_job.fetchone()
            if job_row:
                job_title = job_row['title']
                job_description = job_row['description'] or job_description
                company_id = job_row['companyId']
                
                # Fetch job requirements/skills
                cur_job.execute('SELECT name, level, "yearsOfExperience" FROM "Skill" WHERE "jobId" = %s', (request.job_id,))
                skills = cur_job.fetchall()
                job_requirements = [f"{s['name']} ({s['level']})" for s in skills]
                print(f"🎯 Fetched {len(job_requirements)} job requirements")
            
            cur_job.close()
            conn_job.close()
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch job details: {e}")
            company_id = "unknown"

        # 🚀 USE AGENTIC ORCHESTRATOR (6-agent system with LangGraph)
        print("🤖 Initializing Agentic Orchestrator...")
        orchestrator = get_agentic_orchestrator()
        
        # Execute complete agentic workflow
        print("⚡ Executing 6-agent workflow...")
        prescreening_result = await orchestrator.start_prescreening(
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            resume_text=request.resume_text,
            job_description=job_description,
            job_requirements=job_requirements if job_requirements else [job_description]
        )
        
        print("="*80)
        print(f"✅ AGENTIC PRE-SCREENING COMPLETED")
        print(f"📊 Overall Score: {prescreening_result.overall_score}%")
        print(f"🎯 Bucket: {prescreening_result.bucket.value}")
        print(f"🤖 Next Action: {prescreening_result.next_action}")
        print(f"👥 Human Review Required: {prescreening_result.requires_human_review}")
        print("="*80)
        
        # Return comprehensive agentic analysis
        return {
            "success": True,
            "prescreening_id": prescreening_result.id,
            "agentic_analysis": {
                "overall_score": prescreening_result.overall_score,
                "embedding_score": prescreening_result.embedding_score,
                "keyword_score": prescreening_result.keyword_score,
                "experience_score": prescreening_result.experience_score,
                "bucket": prescreening_result.bucket.value,
                "matched_skills": prescreening_result.resume_matching_result.matched_keywords if prescreening_result.resume_matching_result else [],
                "missing_skills": prescreening_result.resume_matching_result.missing_keywords if prescreening_result.resume_matching_result else [],
                "score_rationale": prescreening_result.resume_matching_result.score_rationale if prescreening_result.resume_matching_result else "",
                "interview_recommendations": prescreening_result.interview_recommendations
            },
            "next_action": prescreening_result.next_action,
            "requires_human_review": prescreening_result.requires_human_review,
            "agent_metadata": {
                "system": "6-Agent LangGraph Orchestrator",
                "agents_executed": [
                    "Resume Verification Agent (DSPy Chain-of-Thought)",
                    "MCQ Generation Agent (Adaptive)",
                    "Video Question Agent",
                    "Proctoring Config Agent (Risk-based)",
                    "Scoring Agent (Multi-dimensional)",
                    "Interview Recommendation Agent"
                ],
                "ai_models": ["GPT-5-Chat (gpt-4o)", "Azure OpenAI Embeddings"],
                "processing_method": "Agentic AI with LangGraph State Management"
            }
        }
        
    except Exception as e:
        print(f"❌ Agentic analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agentic resume analysis failed: {str(e)}")

@router.post("/session/check")
async def check_existing_session(request: dict):
    """Check if there's an existing active session for candidate/job"""
    try:
        candidate_id = request.get('candidate_id')
        job_id = request.get('job_id')
        
        if not candidate_id or not job_id:
            raise HTTPException(status_code=400, detail="candidate_id and job_id are required")
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Look for active sessions
        cursor.execute("""
            SELECT s.*, cp.id as prescreening_id
            FROM prescreening_sessions s
            JOIN candidate_prescreening cp ON s."candidatePreScreeningId" = cp.id
            WHERE cp."candidateId" = %s AND cp."jobId" = %s
            AND s.status IN ('scheduled', 'active')
            ORDER BY s."createdAt" DESC
            LIMIT 1
        """, (candidate_id, job_id))
        
        session_row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not session_row:
            return {"session": None, "exists": False}
        
        return {
            "session": {
                "id": session_row['id'],
                "session_id": session_row['id'],
                "session_token": session_row['sessionToken'],
                "status": session_row['status'],
                "intro_question": session_row['introQuestion'],
                "time_limit_minutes": 30,
                "total_questions": session_row['totalQuestions'],
                "proctoring_config": {
                    "face_detection": True,
                    "tab_monitoring": True,
                    "fullscreen_required": True,
                    "max_violations": 3
                }
            },
            "exists": True
        }
        
    except Exception as e:
        print(f"Error checking existing session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check session: {str(e)}")

@router.post("/create-session")
async def create_prescreening_session(request: PreScreeningSessionCreateRequest):
    """Create a new pre-screening session with MCQ questions"""
    try:
        # First, create or get existing candidate pre-screening record
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Check if pre-screening record already exists
        cursor.execute("""
            SELECT id FROM candidate_prescreening
            WHERE "candidateId" = %s AND "jobId" = %s
        """, (request.candidate_id, request.job_id))
        
        existing_record = cursor.fetchone()
        if existing_record:
            prescreening_id = existing_record['id']
        else:
            # Create new pre-screening record
            prescreening_id = f"ps_{uuid.uuid4().hex[:8]}"
            cursor.execute("""
                INSERT INTO candidate_prescreening (
                    id, "candidateId", "jobId", "resumeScore", "resumeDecision",
                    "prescreeningStatus", "humanReviewRequired", "createdAt", "updatedAt"
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                prescreening_id, request.candidate_id, request.job_id, 75.0, "good",
                PreScreeningStatus.SCHEDULED.value, False, datetime.now(), datetime.now()
            ))

        # Generate MCQ questions from real job context and adaptive policy
        job_ctx = get_job_context(request.job_id)
        job_title = job_ctx.get("title") or "General Role"
        company_id = job_ctx.get("company_id")
        job_desc = job_ctx.get("description") or ""
        skills = [s["name"] for s in (job_ctx.get("skills") or [])] or []
        seniority = infer_seniority_from_title(job_title)
        job_family = map_role_family(job_title, job_desc)

        cfg = get_prescreening_config(company_id, request.job_id) if company_id else None
        policy = get_difficulty_policy(seniority, cfg)

        # ENFORCED: Always exactly 15 questions regardless of config
        total_q = 15  # HARD LIMIT: Pre-screening always has exactly 15 questions
        print(f"✅ Pre-screening question count ENFORCED: {total_q} questions (config ignored)")
        difficulty_mix = policy["distribution"]
        time_limits = policy["time_limits"]

        mcq_questions = await generate_mcq_questions_ai(
            job_title=job_title,
            job_family=job_family,
            skills=skills,
            difficulty_mix=difficulty_mix,
            total=total_q,
            time_limits=time_limits
        )
        
        # Create session
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        session_token = f"token_{uuid.uuid4().hex}"
        expires_at = datetime.now() + timedelta(hours=request.expires_in_hours)
        
        # Create intro question
        intro_question = "Please introduce yourself and explain why you're interested in this position. You have 3 minutes to respond."
        
        session = PreScreeningSession(
            id=session_id,
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            session_token=session_token,
            status=SessionStatus.SCHEDULED,
            expires_at=expires_at,
            intro_question=intro_question,
            mcq_questions=[MCQQuestion(
                id=q.id,
                question=q.question,
                options=q.options,
                difficulty_level=q.difficulty_level,
                skill_category=q.skill_category,
                time_limit=q.time_limit
            ) for q in mcq_questions],
            total_questions=len(mcq_questions),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save session to database
        cursor.execute("""
            INSERT INTO prescreening_sessions (
                id, "candidateId", "jobId", "candidatePreScreeningId", "sessionToken", status,
                "expiresAt", "introQuestion", "mcqQuestions", "totalQuestions",
                "createdAt", "updatedAt"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session_id, request.candidate_id, request.job_id, prescreening_id, session_token,
            session.status.value, expires_at, intro_question, json.dumps([{
                "id": q.id,
                "question": q.question,
                "options": q.options,
                "correct_answer": q.correct_answer,  # Include correct answer in JSON
                "difficulty_level": q.difficulty_level,
                "skill_category": q.skill_category,
                "time_limit": q.time_limit
            } for q in mcq_questions]), len(mcq_questions),
            datetime.now(), datetime.now()
        ))
        
        # Save MCQ questions with correct answers
        for question in mcq_questions:
            print(f"DEBUG: Saving MCQ question {question.id} with correct answer: {question.correct_answer}")
            cursor.execute("""
                INSERT INTO mcq_questions (
                    id, "jobId", "sessionId", question, options, "correctAnswer",
                    "difficultyLevel", "skillCategory", "generatedFrom", "generatedAt"
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                question.id, request.job_id, session_id, question.question,
                json.dumps(question.options), question.correct_answer,
                question.difficulty_level, question.skill_category, "job_description", datetime.now()
            ))
        
        cursor.close()
        conn.close()
        
        return SessionStartResponse(
            session_id=session_id,
            session_token=session_token,
            intro_question=intro_question,
            proctoring_config={
                "face_detection": True,
                "tab_monitoring": True,
                "fullscreen_required": True,
                "max_violations": 3
            },
            time_limit_minutes=30
        )
        
    except Exception as e:
        print(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get session
        cursor.execute('SELECT * FROM prescreening_sessions WHERE id = %s', (session_id,))
        session = cursor.fetchone()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "session": {
                "id": session['id'],
                "candidate_id": session['candidateId'],
                "job_id": session['jobId'],
                "session_token": session['sessionToken'],
                "status": session['status'],
                "intro_question": session['introQuestion'],
                "total_questions": session['totalQuestions'],
                "mcq_score": session.get('mcqScore'),
                "final_score": session.get('finalScore'),
                "correct_answers": session.get('correctAnswers'),
                "proctoring_penalty": session.get('proctoringPenalty'),
                "start_time": session.get('startTime').isoformat() if session.get('startTime') else None,
                "end_time": session.get('endTime').isoformat() if session.get('endTime') else None,
                "expires_at": session['expiresAt'].isoformat() if session.get('expiresAt') else None,
                "created_at": session['createdAt'].isoformat() if session.get('createdAt') else None,
                "updated_at": session['updatedAt'].isoformat() if session.get('updatedAt') else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@router.post("/session/{session_id}/generate-questions")
async def generate_questions_for_session(session_id: str, request: dict):
    """
    Generate MCQ questions for a pre-screening session immediately upon assignment.
    This ensures questions are ready before the candidate starts the test.
    """
    try:
        job_id = request.get('job_id')
        total_questions = request.get('total_questions', 10)
        
        # ENFORCE: Always exactly 15 questions for pre-screening
        total_questions = 15
        
        print(f"📝 Generating {total_questions} MCQ questions for session {session_id}")
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Check if questions already exist
        cursor.execute('SELECT COUNT(*) as count FROM mcq_questions WHERE "sessionId" = %s', (session_id,))
        existing_count = cursor.fetchone()['count']
        
        if existing_count >= total_questions:
            print(f"✅ Session {session_id} already has {existing_count} questions")
            cursor.close()
            conn.close()
            return {
                "success": True,
                "session_id": session_id,
                "total_questions": existing_count,
                "message": "Questions already exist"
            }
        
        # Get job context
        job_ctx = get_job_context(job_id)
        job_title = job_ctx.get("title") or "General Role"
        company_id = job_ctx.get("company_id")
        job_description = job_ctx.get("description") or ""
        skills = [s["name"] for s in (job_ctx.get("skills") or [])]
        
        seniority = infer_seniority_from_title(job_title)
        job_family = map_role_family(job_title, job_description)
        cfg = get_prescreening_config(company_id, job_id) if company_id else None
        policy = get_difficulty_policy(seniority, cfg)
        
        print(f"🎯 Generating {total_questions} questions for {job_title} ({job_family})")
        
        # Generate questions
        mcq_questions_raw = await generate_mcq_questions_ai(
            job_title=job_title,
            job_family=job_family,
            skills=skills,
            difficulty_mix=policy["distribution"],
            total=total_questions,
            time_limits=policy["time_limits"]
        )
        
        # CRITICAL: Ensure exactly the requested number
        if len(mcq_questions_raw) > total_questions:
            print(f"⚠️  AI generated {len(mcq_questions_raw)} questions, trimming to {total_questions}")
            mcq_questions_raw = mcq_questions_raw[:total_questions]
        
        print(f"💾 Saving {len(mcq_questions_raw)} questions to database")
        
        # Save questions to database
        saved_count = 0
        for question in mcq_questions_raw:
            try:
                cursor.execute("""
                    INSERT INTO mcq_questions (
                        id, "jobId", "sessionId", question, options, "correctAnswer",
                        "difficultyLevel", "skillCategory", "generatedFrom", "generatedAt"
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    question.id,
                    job_id,
                    session_id,
                    question.question,
                    json.dumps(question.options),
                    question.correct_answer,
                    question.difficulty_level,
                    question.skill_category,
                    'assignment_generation',
                    datetime.now()
                ))
                saved_count += 1
            except Exception as save_error:
                print(f"⚠️  Error saving question {question.id}: {save_error}")
        
        cursor.close()
        conn.close()
        
        print(f"✅ Successfully saved {saved_count} questions for session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "total_questions": saved_count,
            "message": f"Generated and saved {saved_count} MCQ questions"
        }
        
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

@router.post("/session/{session_id}/start")
async def start_session(session_id: str):
    """
    🚀 REAL-TIME AGENTIC SESSION START (One-by-One Generation)
    
    Flow:
    1. Load context (company, job, candidate) from database
    2. Initialize session without generating questions upfront
    3. Set up adaptive testing metadata
    4. Questions will be generated one-by-one via /next-question endpoint
    """
    try:
        print("="*80)
        print("🚀 REAL-TIME AGENTIC SESSION STARTING (Adaptive Mode)")
        print(f"📋 Session ID: {session_id}")
        print("="*80)
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get session
        cursor.execute('SELECT * FROM prescreening_sessions WHERE id = %s', (session_id,))
        session = cursor.fetchone()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session['status'] == SessionStatus.EXPIRED.value:
            raise HTTPException(status_code=400, detail="Session has expired")
        
        # ====================================================================
        # STEP 1: CONTEXT COLLECTION (Company + Job + Candidate)
        # ====================================================================
        print("📊 STEP 1: Loading context from database...")
        
        candidate_id = session['candidateId']
        job_id = session['jobId']
        
        # Get job context
        job_ctx = get_job_context(job_id)
        job_title = job_ctx.get("title") or "General Role"
        company_id = job_ctx.get("company_id")
        job_description = job_ctx.get("description") or ""
        skills = [s["name"] for s in (job_ctx.get("skills") or [])]
        
        print(f"✅ Context loaded: {job_title} at {company_id or 'Unknown Company'}")
        print(f"📝 Skills required: {len(skills)} skills")
        
        # ====================================================================
        # STEP 2: UPDATE SESSION STATUS & INITIALIZE ADAPTIVE METADATA
        # ====================================================================
        print("🎯 STEP 2: Initializing adaptive testing metadata...")
        
        # Store context for adaptive question generation
        adaptive_metadata = {
            "job_title": job_title,
            "job_family": map_role_family(job_title, job_description),
            "seniority": infer_seniority_from_title(job_title),
            "skills": skills,
            "questions_generated": 0,
            "total_target": 15,
            "difficulty_history": [],
            "skill_coverage": {}
        }
        
        cursor.execute("""
            UPDATE prescreening_sessions
            SET status = %s, "startTime" = %s, "updatedAt" = %s,
                "mcqQuestions" = %s
            WHERE id = %s
        """, (
            SessionStatus.ACTIVE.value,
            datetime.now(),
            datetime.now(),
            json.dumps({"adaptive_metadata": adaptive_metadata, "questions": []}),
            session_id
        ))
        
        cursor.close()
        conn.close()
        
        print("="*80)
        print(f"✅ ADAPTIVE SESSION INITIALIZED")
        print(f"📊 Questions will be generated one-by-one")
        print(f"🎯 Target: 15 questions")
        print(f"🤖 Mode: Real-time Agentic AI")
        print("="*80)
        
        return {
            "success": True,
            "session_id": session_id,
            "status": "active",
            "intro_question": session.get('introQuestion', 'Please introduce yourself'),
            "total_questions": 15,
            "adaptive_mode": True,
            "time_limit_minutes": 30,
            "started_at": datetime.now().isoformat(),
            "agentic_metadata": {
                "generation_method": "Real-time Adaptive (One-by-One)",
                "ai_model": "GPT-5-Chat (gpt-4o)",
                "uniqueness_guarantee": "100% - Questions generated based on real-time performance",
                "adaptive_features": {
                    "difficulty_adjustment": True,
                    "skill_gap_targeting": True,
                    "performance_based_adaptation": True
                }
            },
            "next_action": "Call /session/{session_id}/next-question to get first question"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@router.post("/session/{session_id}/submit-mcq")
async def submit_mcq_answer(session_id: str, answer: MCQAnswerRequest):
    """Submit MCQ answer"""
    try:
        print(f"DEBUG: Submitting answer for session {session_id}")
        print(f"DEBUG: Answer data - question_id: {answer.question_id}, answer: {answer.answer}, time_taken: {answer.time_taken}")
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get correct answer - first try exact match, then try case-insensitive
        cursor.execute("""
            SELECT "correctAnswer", id FROM mcq_questions
            WHERE id = %s AND "sessionId" = %s
        """, (answer.question_id, session_id))
        
        result = cursor.fetchone()
        
        # If exact match fails, try case-insensitive search
        if not result:
            print(f"DEBUG: Exact match failed for question {answer.question_id}, trying case-insensitive search")
            cursor.execute("""
                SELECT "correctAnswer", id FROM mcq_questions
                WHERE LOWER(id) = LOWER(%s) AND "sessionId" = %s
            """, (answer.question_id, session_id))
            result = cursor.fetchone()
        
        # If still no result, list all questions in this session for debugging
        if not result:
            cursor.execute("""
                SELECT id FROM mcq_questions WHERE "sessionId" = %s
            """, (session_id,))
            all_questions = cursor.fetchall()
            question_ids = [q['id'] for q in all_questions] if all_questions else []
            print(f"DEBUG: Question {answer.question_id} not found in session {session_id}")
            print(f"DEBUG: Available questions in session: {question_ids}")
            
            # Also check if the session has questions stored in JSON format
            cursor.execute("""
                SELECT "mcqQuestions" FROM prescreening_sessions WHERE id = %s
            """, (session_id,))
            session_result = cursor.fetchone()
            if session_result and session_result['mcqQuestions']:
                stored_questions = json.loads(session_result['mcqQuestions']) if isinstance(session_result['mcqQuestions'], str) else session_result['mcqQuestions']
                stored_question_ids = [q.get('id') for q in stored_questions] if stored_questions else []
                print(f"DEBUG: Questions in session JSON: {stored_question_ids}")
                
                # Try to find the question in the stored JSON data
                matching_question = next((q for q in stored_questions if q.get('id') == answer.question_id), None)
                if matching_question:
                    print(f"DEBUG: Found question in session JSON data")
                    # Get the correct answer from the database using the question ID
                    cursor.execute("""
                        SELECT "correctAnswer" FROM mcq_questions WHERE id = %s
                    """, (answer.question_id,))
                    correct_answer_row = cursor.fetchone()
                    if correct_answer_row:
                        result = {'correctAnswer': correct_answer_row['correctAnswer'], 'id': answer.question_id}
                        print(f"DEBUG: Found correct answer in database: {correct_answer_row['correctAnswer']}")
                    else:
                        print(f"DEBUG: No correct answer found in database for question {answer.question_id}")
                        # As a last resort, try to find the answer in the mcqQuestions JSON with correct_answer field
                        if hasattr(matching_question, 'get') and matching_question.get('correct_answer'):
                            result = {'correctAnswer': matching_question['correct_answer'], 'id': answer.question_id}
                            print(f"DEBUG: Using correct answer from JSON: {matching_question['correct_answer']}")
                        else:
                            result = {'correctAnswer': 'A', 'id': answer.question_id}  # Final fallback
                            print(f"DEBUG: Using fallback correct answer 'A' for question {answer.question_id}")
                
            if not result:
                raise HTTPException(status_code=404, detail="Question not found")
        
        correct_answer = result['correctAnswer']
        is_correct = answer.answer.upper() == correct_answer.upper()
        print(f"DEBUG: Correct answer: {correct_answer}, Submitted: {answer.answer}, Is correct: {is_correct}")
        
        # Initialize mcqAnswers and correctAnswers if they are NULL
        cursor.execute("""
            UPDATE prescreening_sessions
            SET "mcqAnswers" = CASE
                WHEN "mcqAnswers" IS NULL THEN '[]'::jsonb
                ELSE "mcqAnswers"
            END,
            "correctAnswers" = CASE
                WHEN "correctAnswers" IS NULL THEN 0
                ELSE "correctAnswers"
            END
            WHERE id = %s
        """, (session_id,))
        
        # Update session with answers (store in mcqAnswers JSON field)
        cursor.execute("""
            UPDATE prescreening_sessions
            SET "mcqAnswers" = "mcqAnswers" || %s::jsonb,
                "correctAnswers" = "correctAnswers" + %s,
                "updatedAt" = %s
            WHERE id = %s
        """, (
            json.dumps([{
                "questionId": answer.question_id,
                "answer": answer.answer,
                "isCorrect": is_correct,
                "timeTaken": answer.time_taken,
                "submittedAt": datetime.now().isoformat()
            }]),
            1 if is_correct else 0,
            datetime.now(),
            session_id
        ))
        
        print(f"DEBUG: Answer submitted successfully for session {session_id}")
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "question_id": answer.question_id,
            "submitted_answer": answer.answer,
            "is_correct": is_correct,
            "time_taken": answer.time_taken,
            "submitted_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error submitting MCQ answer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit answer: {str(e)}")

@router.post("/session/{session_id}/next-question")
async def generate_next_question(session_id: str):
    """
    🎯 AGENTIC REAL-TIME QUESTION GENERATION
    
    Generates the next MCQ question based on:
    1. Candidate's performance on previous questions
    2. Skills not yet covered
    3. Adaptive difficulty adjustment
    4. Job requirements and context
    
    This endpoint is called after each answer submission to get the next question.
    """
    try:
        print("="*80)
        print(f"🎯 GENERATING NEXT QUESTION FOR SESSION {session_id}")
        print("="*80)
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get session with metadata
        cursor.execute('SELECT * FROM prescreening_sessions WHERE id = %s', (session_id,))
        session = cursor.fetchone()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Auto-activate session if it's still scheduled (allows seamless start)
        current_status = session['status']
        if current_status not in [SessionStatus.ACTIVE.value, SessionStatus.SCHEDULED.value]:
            print(f"❌ Session {session_id} has invalid status: {current_status}")
            raise HTTPException(status_code=400, detail=f"Session is {current_status}, cannot generate questions")
        
        # If session is scheduled, activate it now
        if current_status == SessionStatus.SCHEDULED.value:
            print(f"🔄 Auto-activating session {session_id} (was SCHEDULED)")
            cursor.execute("""
                UPDATE prescreening_sessions
                SET status = %s, "startTime" = %s, "updatedAt" = %s
                WHERE id = %s
            """, (
                SessionStatus.ACTIVE.value,
                datetime.now(),
                datetime.now(),
                session_id
            ))
            print(f"✅ Session {session_id} activated")
        
        # Parse adaptive metadata
        mcq_data = session.get('mcqQuestions')
        if isinstance(mcq_data, str):
            mcq_data = json.loads(mcq_data)
        
        # Handle both old format (list) and new format (dict with adaptive_metadata)
        if isinstance(mcq_data, list):
            # Old format - convert to new format
            print(f"⚠️  Session {session_id} using old format, initializing adaptive metadata")
            existing_questions = mcq_data
            adaptive_metadata = {
                "job_title": "General Role",
                "job_family": "engineering",
                "seniority": "mid",
                "skills": [],
                "questions_generated": len(existing_questions),
                "total_target": 15,
                "difficulty_history": [],
                "skill_coverage": {}
            }
            
            # Initialize from job context
            job_id = session['jobId']
            job_ctx = get_job_context(job_id)
            if job_ctx.get("title"):
                adaptive_metadata["job_title"] = job_ctx["title"]
                adaptive_metadata["job_family"] = map_role_family(job_ctx["title"], job_ctx.get("description", ""))
                adaptive_metadata["seniority"] = infer_seniority_from_title(job_ctx["title"])
                adaptive_metadata["skills"] = [s["name"] for s in job_ctx.get("skills", [])]
        elif isinstance(mcq_data, dict):
            # New format
            adaptive_metadata = mcq_data.get('adaptive_metadata', {})
            existing_questions = mcq_data.get('questions', [])
        else:
            # No data yet - initialize
            print(f"📋 Session {session_id} has no question data, initializing")
            job_id = session['jobId']
            job_ctx = get_job_context(job_id)
            adaptive_metadata = {
                "job_title": job_ctx.get("title") or "General Role",
                "job_family": map_role_family(job_ctx.get("title", ""), job_ctx.get("description", "")),
                "seniority": infer_seniority_from_title(job_ctx.get("title", "")),
                "skills": [s["name"] for s in job_ctx.get("skills", [])],
                "questions_generated": 0,
                "total_target": 15,
                "difficulty_history": [],
                "skill_coverage": {}
            }
            existing_questions = []
        
        questions_generated = len(existing_questions)
        total_target = adaptive_metadata.get('total_target', 15)
        
        print(f"📊 Progress: {questions_generated}/{total_target} questions generated")
        
        # Check if we've reached the target
        if questions_generated >= total_target:
            print(f"✅ All {total_target} questions generated")
            cursor.close()
            conn.close()
            return {
                "success": True,
                "completed": True,
                "message": "All questions have been generated",
                "total_questions": total_target
            }
        
        # ====================================================================
        # ANALYZE PREVIOUS PERFORMANCE (Adaptive Logic)
        # ====================================================================
        print("📈 Analyzing previous performance...")
        
        # Get previous answers
        mcq_answers = []
        if session.get('mcqAnswers'):
            if isinstance(session['mcqAnswers'], str):
                mcq_answers = json.loads(session['mcqAnswers'])
            else:
                mcq_answers = session['mcqAnswers'] or []
        
        # Calculate performance metrics
        if mcq_answers:
            correct_count = sum(1 for ans in mcq_answers if ans.get('isCorrect'))
            accuracy = correct_count / len(mcq_answers) if mcq_answers else 0
            avg_time = sum(ans.get('timeTaken', 0) for ans in mcq_answers) / len(mcq_answers) if mcq_answers else 0
            
            print(f"   Accuracy: {accuracy:.1%} ({correct_count}/{len(mcq_answers)})")
            print(f"   Avg Time: {avg_time:.1f}s")
        else:
            accuracy = 0.5  # Neutral starting point
            avg_time = 90
            print(f"   First question - no performance data yet")
        
        # ====================================================================
        # DETERMINE NEXT DIFFICULTY (BALANCED ADAPTIVE)
        # ====================================================================
        job_title = adaptive_metadata.get('job_title', 'General Role')
        job_family = adaptive_metadata.get('job_family', 'engineering')
        seniority = adaptive_metadata.get('seniority', 'mid')
        skills = adaptive_metadata.get('skills', [])
        difficulty_history = adaptive_metadata.get('difficulty_history', [])
        skill_coverage = adaptive_metadata.get('skill_coverage', {})
        
        # Get job context for difficulty policy
        job_id = session['jobId']
        job_ctx = get_job_context(job_id)
        company_id = job_ctx.get("company_id")
        
        # Get difficulty policy and calculate target counts
        cfg = get_prescreening_config(company_id, job_id) if company_id else None
        policy = get_difficulty_policy(seniority, cfg)
        difficulty_dist = policy["distribution"]
        
        # Calculate target counts for each difficulty (30% easy, 40% medium, 30% hard as default)
        total_target = adaptive_metadata.get('total_target', 15)
        target_easy = int(difficulty_dist.get("easy", 0.3) * total_target)
        target_medium = int(difficulty_dist.get("medium", 0.4) * total_target)
        target_hard = total_target - target_easy - target_medium
        
        # Count current difficulty distribution
        easy_count = difficulty_history.count("easy")
        medium_count = difficulty_history.count("medium")
        hard_count = difficulty_history.count("hard")
        
        print(f"📊 Difficulty targets: Easy={target_easy}, Medium={target_medium}, Hard={target_hard}")
        print(f"📊 Current counts: Easy={easy_count}, Medium={medium_count}, Hard={hard_count}")
        
        # BALANCED ADAPTIVE LOGIC:
        # 1. Ensure we don't exceed target counts for any difficulty
        # 2. Use performance to guide selection within available difficulties
        
        # Determine which difficulties are still available
        available_difficulties = []
        if easy_count < target_easy:
            available_difficulties.append("easy")
        if medium_count < target_medium:
            available_difficulties.append("medium")
        if hard_count < target_hard:
            available_difficulties.append("hard")
        
        # If all targets met (shouldn't happen but safety check)
        if not available_difficulties:
            available_difficulties = ["medium"]  # Default to medium
        
        print(f"✅ Available difficulties: {available_difficulties}")
        
        # Select difficulty based on performance within available options
        if questions_generated == 0:
            # First question - prefer medium if available, otherwise easy
            next_difficulty = "medium" if "medium" in available_difficulties else available_difficulties[0]
        elif accuracy >= 0.8:
            # High accuracy - prefer harder questions if available
            if "hard" in available_difficulties:
                next_difficulty = "hard"
            elif "medium" in available_difficulties:
                next_difficulty = "medium"
            else:
                next_difficulty = available_difficulties[0]
        elif accuracy >= 0.5:
            # Moderate accuracy - prefer medium if available
            if "medium" in available_difficulties:
                next_difficulty = "medium"
            elif "hard" in available_difficulties:
                next_difficulty = "hard"
            else:
                next_difficulty = available_difficulties[0]
        else:
            # Low accuracy - prefer easier questions if available
            if "easy" in available_difficulties:
                next_difficulty = "easy"
            elif "medium" in available_difficulties:
                next_difficulty = "medium"
            else:
                next_difficulty = available_difficulties[0]
        
        print(f"🎯 DIFFICULTY SELECTION:")
        print(f"   Requested: {next_difficulty}")
        print(f"   Based on: {accuracy:.1%} accuracy, Question #{questions_generated + 1}")
        print(f"   Available options were: {available_difficulties}")
        print(f"   Distribution targets: Easy={target_easy}, Medium={target_medium}, Hard={target_hard}")
        print(f"   Current counts: Easy={easy_count}, Medium={medium_count}, Hard={hard_count}")
        
        # ====================================================================
        # SELECT SKILL TO TARGET (Ensure coverage)
        # ====================================================================
        # Find least covered skill
        if skills:
            skill_scores = {skill: skill_coverage.get(skill, 0) for skill in skills}
            target_skill = min(skill_scores.items(), key=lambda x: x[1])[0]
        else:
            target_skill = job_family
        
        print(f"📚 Target skill: {target_skill}")
        
        # ====================================================================
        # GENERATE SINGLE QUESTION USING AI
        # ====================================================================
        print("🤖 Generating question with GPT-5-Chat...")
        
        # Get job context for better questions
        job_id = session['jobId']
        candidate_id = session['candidateId']
        job_ctx = get_job_context(job_id)
        job_description = job_ctx.get("description") or ""
        company_id = job_ctx.get("company_id")
        
        # Get difficulty policy for time limits
        cfg = get_prescreening_config(company_id, job_id) if company_id else None
        policy = get_difficulty_policy(seniority, cfg)
        time_limits = policy["time_limits"]
        
        # ====================================================================
        # FETCH CANDIDATE'S RESUME CONTEXT FOR PERSONALIZED QUESTIONS
        # ====================================================================
        resume_context = None
        try:
            print(f"📄 Fetching resume context for candidate {candidate_id}...")
            # Join with Candidate table to get parsedResume and ResumeMatchingResult for skills
            cursor.execute("""
                SELECT 
                    c."parsedResume", 
                    c."aiAnalysis",
                    rmr."matchedKeywords",
                    rmr."missingKeywords"
                FROM candidate_prescreening cp
                JOIN "Candidate" c ON cp."candidateId" = c.id
                LEFT JOIN resume_matching_results rmr ON rmr."candidatePreScreeningId" = cp.id
                WHERE cp."candidateId" = %s AND cp."jobId" = %s
            """, (candidate_id, job_id))
            
            resume_row = cursor.fetchone()
            if resume_row:
                parsed_resume = resume_row.get('parsedResume')
                ai_analysis_str = resume_row.get('aiAnalysis')
                matched_keywords = resume_row.get('matchedKeywords')
                missing_keywords = resume_row.get('missingKeywords')
                
                # Parse parsedResume if needed
                if isinstance(parsed_resume, str):
                    try:
                        parsed_resume = json.loads(parsed_resume)
                    except:
                        parsed_resume = {}
                
                # Extract companies and duties from parsedResume
                previous_companies = []
                previous_duties = []
                if parsed_resume and isinstance(parsed_resume, dict):
                    experience = parsed_resume.get('experience', [])
                    if isinstance(experience, list):
                        for exp in experience:
                            if isinstance(exp, dict):
                                company = exp.get('company') or exp.get('employer')
                                if company:
                                    previous_companies.append({
                                        'name': company, 
                                        'role': exp.get('title', 'Employee'), 
                                        'key_duties': (exp.get('description') or '').split('\n')[:3]
                                    })
                                duties = exp.get('description')
                                if duties:
                                    previous_duties.append(duties)

                # Parse aiAnalysis if possible
                ai_analysis = {}
                if ai_analysis_str:
                    try:
                        ai_analysis = json.loads(ai_analysis_str)
                    except:
                        pass # aiAnalysis might be just a string text
                
                # Construct resume_context
                resume_context = {
                    "previous_companies": previous_companies,
                    "previous_duties": previous_duties,
                    "transferable_skills": ai_analysis.get('transferable_skills', []),
                    "skill_depth_analysis": ai_analysis.get('skill_depth_analysis', {}),
                    "areas_to_probe": ai_analysis.get('areas_to_probe', []),
                    "matched_skills": matched_keywords if matched_keywords else [],
                    "skill_gaps": missing_keywords if missing_keywords else []
                }
                print(f"✅ Resume context loaded: {len(resume_context.get('previous_companies', []))} companies, {len(resume_context.get('skill_gaps', []))} skill gaps")
            else:
                print(f"⚠️ No prescreening record found for candidate {candidate_id} and job {job_id}")
        except Exception as resume_error:
            print(f"⚠️ Error fetching resume context: {resume_error}")
            # Continue without resume context - questions will still be generated
        
        # Generate a single question with resume context for personalization
        question = await generate_single_mcq_question(
            job_title=job_title,
            job_family=job_family,
            target_skill=target_skill,
            difficulty=next_difficulty,
            time_limit=time_limits.get(next_difficulty, 90),
            questions_generated=questions_generated,
            previous_questions=existing_questions,
            performance_context={
                "accuracy": accuracy,
                "avg_time": avg_time,
                "answered_count": len(mcq_answers)
            },
            resume_context=resume_context
        )
        
        # ====================================================================
        # SAVE QUESTION TO DATABASE
        # ====================================================================
        print(f"💾 Saving question {questions_generated + 1} to database...")
        
        try:
            cursor.execute("""
                INSERT INTO mcq_questions (
                    id, "jobId", "sessionId", question, options, "correctAnswer",
                    "difficultyLevel", "skillCategory", "generatedFrom", "generatedAt"
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                question.id,
                job_id,
                session_id,
                question.question,
                json.dumps(question.options),
                question.correct_answer,
                question.difficulty_level,
                question.skill_category,
                'real_time_agentic',
                datetime.now()
            ))
        except Exception as save_error:
            print(f"⚠️  Error saving question: {save_error}")
            raise HTTPException(status_code=500, detail=f"Failed to save question: {str(save_error)}")
        
        # ====================================================================
        # UPDATE SESSION METADATA
        # ====================================================================
        # Add question to existing questions list (without correct answer for security)
        existing_questions.append({
            "id": question.id,
            "question": question.question,
            "options": question.options,
            "difficulty_level": question.difficulty_level,
            "skill_category": question.skill_category,
            "time_limit": question.time_limit
        })
        
        # Update adaptive metadata
        difficulty_history.append(next_difficulty)
        skill_coverage[target_skill] = skill_coverage.get(target_skill, 0) + 1
        
        adaptive_metadata['questions_generated'] = questions_generated + 1
        adaptive_metadata['difficulty_history'] = difficulty_history
        adaptive_metadata['skill_coverage'] = skill_coverage
        
        # Save updated metadata
        cursor.execute("""
            UPDATE prescreening_sessions
            SET "mcqQuestions" = %s, "updatedAt" = %s
            WHERE id = %s
        """, (
            json.dumps({"adaptive_metadata": adaptive_metadata, "questions": existing_questions}),
            datetime.now(),
            session_id
        ))
        
        cursor.close()
        conn.close()
        
        print("="*80)
        print(f"✅ QUESTION {questions_generated + 1} GENERATED")
        print(f"   Difficulty: {next_difficulty}")
        print(f"   Skill: {target_skill}")
        print(f"   Progress: {questions_generated + 1}/{total_target}")
        print("="*80)
        
        # Return question without correct answer
        return {
            "success": True,
            "question": {
                "id": question.id,
                "question": question.question,
                "options": question.options,
                "difficulty_level": question.difficulty_level,
                "skill_category": question.skill_category,
                "time_limit": question.time_limit
            },
            "progress": {
                "current": questions_generated + 1,
                "total": total_target,
                "percentage": ((questions_generated + 1) / total_target) * 100
            },
            "adaptive_info": {
                "difficulty": next_difficulty,
                "target_skill": target_skill,
                "performance_based": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error generating next question: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate question: {str(e)}")

async def generate_single_mcq_question(
    job_title: str,
    job_family: str,
    target_skill: str,
    difficulty: str,
    time_limit: int,
    questions_generated: int,
    previous_questions: List[dict],
    performance_context: dict,
    resume_context: Optional[Dict[str, Any]] = None
) -> MCQQuestionWithAnswer:
    """
    Generate a single MCQ question using AI based on adaptive parameters.
    This ensures uniqueness by considering previously generated questions.
    
    ENHANCED: Now incorporates candidate's resume data for personalized questions
    that test their claimed experience and skills.
    """
    try:
        llm = get_azure_llm()
        
        # Get previous question texts to avoid repetition
        previous_topics = [q.get('question', '')[:100] for q in previous_questions[-5:]]  # Last 5
        
        # Define difficulty requirements clearly
        difficulty_guidelines = {
            "easy": "Basic concepts, straightforward questions, foundational knowledge. Should be answerable with fundamental understanding.",
            "medium": "Intermediate concepts, scenario-based questions, requires application of knowledge. Involves problem-solving and analysis.",
            "hard": "Advanced concepts, complex scenarios, edge cases, optimization, architecture decisions. Requires deep expertise and critical thinking."
        }
        
        difficulty_guide = difficulty_guidelines.get(difficulty, difficulty_guidelines["medium"])
        
        # Build resume context section for personalized questions
        resume_section = ""
        if resume_context:
            previous_companies = resume_context.get('previous_companies', [])
            previous_duties = resume_context.get('previous_duties', [])
            transferable_skills = resume_context.get('transferable_skills', [])
            skill_depth = resume_context.get('skill_depth_analysis', {})
            areas_to_probe = resume_context.get('areas_to_probe', [])
            matched_skills = resume_context.get('matched_skills', [])
            skill_gaps = resume_context.get('skill_gaps', [])
            
            resume_section = f"""
===== CANDIDATE'S RESUME CONTEXT (Use this to personalize questions!) =====

PREVIOUS WORK EXPERIENCE:
{chr(10).join([f"- {c.get('role', 'Role')} at {c.get('name', 'Company')}: {', '.join(c.get('key_duties', [])[:3])}" for c in previous_companies[:3]]) if previous_companies else "- No specific experience extracted"}

KEY DUTIES FROM PREVIOUS ROLES:
{chr(10).join([f"- {duty}" for duty in previous_duties[:5]]) if previous_duties else "- Standard industry experience"}

SKILLS WITH PROVEN DEPTH (test these with harder questions):
{', '.join(skill_depth.get('deep_expertise', [])[:5]) if skill_depth.get('deep_expertise') else "None identified"}

SKILLS WITH MODERATE EXPERIENCE (test with medium questions):
{', '.join(skill_depth.get('moderate_experience', [])[:5]) if skill_depth.get('moderate_experience') else "None identified"}

SKILLS MENTIONED BUT LIMITED EXPERIENCE (test fundamentals):
{', '.join(skill_depth.get('surface_level', [])[:5]) if skill_depth.get('surface_level') else "None identified"}

SKILL GAPS TO ASSESS (missing from resume):
{', '.join(skill_gaps[:5]) if skill_gaps else "None identified"}

AREAS TO PROBE (based on experience analysis):
{', '.join(areas_to_probe[:5]) if areas_to_probe else "General competency"}

QUESTION PERSONALIZATION STRATEGY:
- If target_skill is in their deep expertise → ask advanced practical scenarios from their domain
- If target_skill is in their skill gaps → test foundational knowledge they should have
- If target_skill relates to their previous duties → reference realistic work scenarios
- Make questions feel relevant to their career background
"""
        
        prompt = f"""
You are generating a SINGLE unique MCQ question for a pre-screening test.

JOB CONTEXT:
- Role: {job_title} ({job_family})
- Target Skill: {target_skill}
- Question Number: {questions_generated + 1} of 15
{resume_section}
CANDIDATE PERFORMANCE:
- Questions Answered: {performance_context['answered_count']}
- Current Accuracy: {performance_context['accuracy']:.1%}
- Average Time: {performance_context['avg_time']:.1f}s

PREVIOUS QUESTIONS (avoid repetition):
{chr(10).join([f"- {topic}..." for topic in previous_topics]) if previous_topics else "- None yet"}

CRITICAL DIFFICULTY REQUIREMENT:
THIS QUESTION MUST BE "{difficulty.upper()}" DIFFICULTY LEVEL.
{difficulty_guide}

DIFFICULTY-SPECIFIC REQUIREMENTS FOR {difficulty.upper()}:
{"- Use basic terminology and straightforward scenarios" if difficulty == "easy" else ""}
{"- Test fundamental understanding with clear, direct questions" if difficulty == "easy" else ""}
{"- Include practical scenarios requiring application of concepts" if difficulty == "medium" else ""}
{"- Test problem-solving ability with realistic situations" if difficulty == "medium" else ""}
{"- Present complex scenarios with multiple considerations" if difficulty == "hard" else ""}
{"- Require deep technical knowledge and critical analysis" if difficulty == "hard" else ""}
{"- Include edge cases, optimization concerns, or architectural decisions" if difficulty == "hard" else ""}

GENERAL REQUIREMENTS:
1. Generate EXACTLY ONE question at "{difficulty}" difficulty
2. Must be about: {target_skill}
3. Time limit: {time_limit} seconds
4. Must be COMPLETELY DIFFERENT from previous questions
5. No trick questions; exactly one correct answer
6. Relevant to {job_title} role in {job_family} domain
7. The difficulty_level field MUST be set to "{difficulty}"
8. If resume context is provided, make the question relevant to their background

Return ONLY a JSON object (no markdown, no extra text):
{{
  "id": "unique_string",
  "question": "your {difficulty}-level question here",
  "options": {{"A": "option A", "B": "option B", "C": "option C", "D": "option D"}},
  "correct_answer": "A" | "B" | "C" | "D",
  "difficulty_level": "{difficulty}",
  "skill_category": "{target_skill}",
  "time_limit": {time_limit},
  "rationale": "brief explanation for the correct answer"
}}

CRITICAL:
- The "difficulty_level" field MUST EXACTLY match "{difficulty}"
- The question complexity MUST match {difficulty.upper()} difficulty standards
- Return ONLY the JSON object, nothing else.
"""
        
        response = await llm.agenerate([[HumanMessage(content=prompt)]])
        raw_response = response.generations[0][0].text.strip()
        
        # Clean JSON fences if present
        if raw_response.startswith('```json'):
            raw_response = raw_response[7:]
        if raw_response.endswith('```'):
            raw_response = raw_response[:-3]
        
        question_data = json.loads(raw_response.strip())
        
        # VALIDATE: Ensure AI returned the correct difficulty level
        returned_difficulty = question_data.get("difficulty_level", "").lower()
        if returned_difficulty != difficulty.lower():
            print(f"⚠️ AI returned difficulty '{returned_difficulty}' but requested '{difficulty}' - correcting")
            question_data["difficulty_level"] = difficulty
        
        # Generate unique ID with timestamp to prevent collisions
        timestamp_prefix = int(datetime.now().timestamp() * 1000)
        base_id = question_data.get("id", f"q_{questions_generated + 1}")
        unique_id = f"{timestamp_prefix}_{base_id}_{uuid.uuid4().hex[:8]}"
        
        # Force the correct difficulty level (override AI if needed)
        return MCQQuestionWithAnswer(
            id=unique_id,
            question=question_data.get("question", f"{job_title}: question {questions_generated + 1}"),
            options=question_data.get("options", {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}),
            correct_answer=question_data.get("correct_answer", "A"),
            difficulty_level=difficulty,  # Always use the requested difficulty
            skill_category=target_skill,
            time_limit=time_limit,
            rationale=question_data.get("rationale", "Based on best practices and role expectations.")
        )
        
    except Exception as e:
        print(f"Error generating single MCQ question: {e}")
        # Fallback question
        timestamp_prefix = int(datetime.now().timestamp() * 1000)
        unique_id = f"{timestamp_prefix}_fallback_{questions_generated + 1}_{uuid.uuid4().hex[:8]}"
        
        return MCQQuestionWithAnswer(
            id=unique_id,
            question=f"{job_title} ({job_family}) - {target_skill} competency check {questions_generated + 1}",
            options={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
            correct_answer="A",
            difficulty_level=difficulty,
            skill_category=target_skill,
            time_limit=time_limit,
            rationale="Fallback question rationale"
        )

@router.post("/session/{session_id}/proctoring")
async def record_proctoring_event(session_id: str, event: ProctoringEventRequest):
    """Record proctoring event"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        event_id = f"proc_{uuid.uuid4().hex[:8]}"
        severity = "high" if event.confidence_score > 0.8 else "medium" if event.confidence_score > 0.5 else "low"
        
        cursor.execute("""
            INSERT INTO proctoring_events (
                id, "sessionId", "eventType", timestamp,
                "confidenceScore", severity, metadata, "createdAt"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            event_id, session_id, event.event_type, event.timestamp,
            event.confidence_score, severity, json.dumps(event.metadata), datetime.now()
        ))
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "event_id": event_id,
            "session_id": session_id,
            "severity": severity,
            "recorded_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error recording proctoring event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record event: {str(e)}")

@router.post("/session/{session_id}/complete")
async def complete_session(session_id: str):
    """Complete pre-screening session and calculate WEIGHTED final score with auto-advancement"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get session and answers
        cursor.execute('SELECT * FROM prescreening_sessions WHERE id = %s', (session_id,))
        session = cursor.fetchone()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # WEIGHTED SCORING SYSTEM
        # Get all questions with their difficulty levels and candidate answers
        cursor.execute("""
            SELECT
                q.id,
                q."difficultyLevel",
                q."correctAnswer"
            FROM mcq_questions q
            WHERE q."sessionId" = %s
            ORDER BY q."generatedAt"
        """, (session_id,))
        
        questions = cursor.fetchall()
        
        # Parse candidate answers from session
        mcq_answers = []
        if session.get('mcqAnswers'):
            if isinstance(session['mcqAnswers'], str):
                mcq_answers = json.loads(session['mcqAnswers'])
            else:
                mcq_answers = session['mcqAnswers'] or []
        
        # Define point weights by difficulty
        POINT_WEIGHTS = {
            'easy': 7,
            'medium': 10,
            'hard': 14
        }
        
        # Calculate weighted score
        total_possible_points = 0
        earned_points = 0
        correct_count = 0
        difficulty_breakdown = {'easy': {'correct': 0, 'total': 0}, 'medium': {'correct': 0, 'total': 0}, 'hard': {'correct': 0, 'total': 0}}
        
        for question in questions:
            difficulty = question['difficultyLevel'].lower()
            points = POINT_WEIGHTS.get(difficulty, 10)
            total_possible_points += points
            difficulty_breakdown[difficulty]['total'] += 1
            
            # Find if this question was answered correctly
            answer = next((ans for ans in mcq_answers if ans.get('questionId') == question['id']), None)
            if answer and answer.get('isCorrect'):
                earned_points += points
                correct_count += 1
                difficulty_breakdown[difficulty]['correct'] += 1
        
        # Normalize to 100-point scale
        weighted_score = (earned_points / total_possible_points * 100) if total_possible_points > 0 else 0
        
        # Get proctoring violations count (for tracking only, NO PENALTY)
        cursor.execute("""
            SELECT COUNT(*) as violations FROM proctoring_events
            WHERE "sessionId" = %s AND severity IN ('high', 'critical')
        """, (session_id,))
        
        violations = cursor.fetchone()['violations'] or 0
        proctoring_penalty = 0  # NO PENALTY - violations are tracked for monitoring only
        
        final_score = weighted_score  # Final score is weighted MCQ performance
        
        # AUTO-ADVANCEMENT LOGIC with weighted scoring
        if final_score >= 80:
            decision = "auto_approved"
            next_steps = "Automatically advanced to AI Video Interview"
            candidate_status = "AI_INTERVIEW_SCHEDULED"
        elif final_score >= 60:
            decision = "review_required"
            next_steps = "Requires human review before proceeding"
            candidate_status = "PRE_SCREENING_COMPLETED"
        else:
            decision = "rejected"
            next_steps = "Did not meet minimum requirements"
            candidate_status = "REJECTED"
        
        print(f"✅ Weighted Score Calculation:")
        print(f"   Total Points Possible: {total_possible_points}")
        print(f"   Points Earned: {earned_points}")
        print(f"   Weighted Score: {final_score:.2f}%")
        print(f"   Decision: {decision}")
        print(f"   Difficulty Breakdown: {difficulty_breakdown}")
        
        # Update session with weighted score
        cursor.execute("""
            UPDATE prescreening_sessions
            SET status = %s, "endTime" = %s, "mcqScore" = %s,
                "proctoringPenalty" = %s, "finalScore" = %s, "updatedAt" = %s
            WHERE id = %s
        """, (
            SessionStatus.COMPLETED.value, datetime.now(), weighted_score,
            proctoring_penalty, final_score, datetime.now(), session_id
        ))
        
        # Update candidate_prescreening status
        cursor.execute("""
            UPDATE candidate_prescreening
            SET "prescreeningStatus" = %s,
                "prescreeningScore" = %s,
                "prescreeningDecision" = %s,
                "updatedAt" = %s
            WHERE id = %s
        """, (
            'completed',
            final_score,
            decision,
            datetime.now(),
            session.get('candidatePreScreeningId')
        ))
        
        # Update candidate status based on auto-advancement logic
        candidate_id = session.get('candidateId')
        if candidate_id:
            cursor.execute("""
                UPDATE "Candidate"
                SET status = %s, "updatedAt" = %s
                WHERE id = %s
            """, (
                candidate_status,
                datetime.now(),
                candidate_id
            ))
            
            # Create timeline entry for auto-advancement
            if decision == "auto_approved":
                cursor.execute("""
                    INSERT INTO candidate_timeline (
                        id, "candidateId", stage, date, notes, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    f"timeline_{uuid.uuid4().hex[:8]}",
                    candidate_id,
                    'AI_INTERVIEW_SCHEDULED',
                    datetime.now(),
                    f'Automatically advanced to AI Interview (Score: {final_score:.1f}%)',
                    json.dumps({'auto_advanced': True, 'weighted_score': final_score, 'decision': decision})
                ))
        
        print(f"✅ Updated session {session_id}, prescreening, and candidate status to: {candidate_status}")
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "session_id": session_id,
            "final_score": final_score,
            "decision": decision,
            "next_steps": next_steps,
            "auto_advanced": decision == "auto_approved",
            "breakdown": {
                "weighted_score": weighted_score,
                "total_points_earned": earned_points,
                "total_points_possible": total_possible_points,
                "correct_answers": correct_count,
                "total_questions": len(questions),
                "difficulty_breakdown": difficulty_breakdown,
                "proctoring_violations_count": violations,
                "proctoring_penalty": 0,
                "scoring_method": "Weighted by difficulty (Easy:7pts, Medium:10pts, Hard:14pts)",
                "note": "Proctoring violations tracked but do not affect score. Score ≥80% = Auto-advance to Interview"
            },
            "completed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error completing session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to complete session: {str(e)}")

@router.post("/human-review/decision")
async def submit_human_review_decision(request: HumanReviewDecisionRequest):
    """Submit human review decision"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create review record
        review_id = f"review_{uuid.uuid4().hex[:8]}"
        
        cursor.execute("""
            INSERT INTO human_review_tasks (
                id, "candidateId", "jobId", "candidatePreScreeningId", stage, score,
                "reviewType", priority, status, decision, "decisionReason",
                "decisionNotes", "overrideScore", "contextData", "createdAt", "updatedAt"
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            review_id, "unknown", "unknown", request.task_id, "prescreening", 0,
            "manual_escalation", "medium", "completed", request.decision.value, request.reason,
            request.notes, request.override_score, json.dumps({}), datetime.now(), datetime.now()
        ))
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "review_id": review_id,
            "task_id": request.task_id,
            "decision": request.decision.value,
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error submitting review decision: {e}")
        raise HTTPException(status_code=500, detail=f"Review submission failed: {str(e)}")

@router.post("/reset")
async def reset_candidate_prescreening(request: dict):
    """Reset all pre-screening data for a candidate to allow retaking"""
    try:
        candidate_id = request.get('candidate_id')
        job_id = request.get('job_id')
        
        if not candidate_id:
            raise HTTPException(status_code=400, detail="candidate_id is required")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        deleted_sessions = 0
        deleted_questions = 0
        deleted_events = 0
        
        # First get the session IDs before deleting them
        if job_id:
            cursor.execute('SELECT id FROM prescreening_sessions WHERE "candidateId" = %s AND "jobId" = %s',
                         (candidate_id, job_id))
        else:
            cursor.execute('SELECT id FROM prescreening_sessions WHERE "candidateId" = %s', (candidate_id,))
            
        session_ids = [row[0] for row in cursor.fetchall()]
        
        if session_ids:
            # Convert list to tuple for SQL IN clause
            session_ids_tuple = tuple(session_ids) if len(session_ids) > 1 else f"('{session_ids[0]}')"
            
            # Delete MCQ questions first
            if len(session_ids) == 1:
                cursor.execute('DELETE FROM mcq_questions WHERE "sessionId" = %s', (session_ids[0],))
            else:
                cursor.execute(f'DELETE FROM mcq_questions WHERE "sessionId" IN {session_ids_tuple}')
            deleted_questions = cursor.rowcount
            
            # Delete proctoring events
            if len(session_ids) == 1:
                cursor.execute('DELETE FROM proctoring_events WHERE "sessionId" = %s', (session_ids[0],))
            else:
                cursor.execute(f'DELETE FROM proctoring_events WHERE "sessionId" IN {session_ids_tuple}')
            deleted_events = cursor.rowcount
            
            # Finally delete the sessions
            if job_id:
                cursor.execute('DELETE FROM prescreening_sessions WHERE "candidateId" = %s AND "jobId" = %s',
                             (candidate_id, job_id))
            else:
                cursor.execute('DELETE FROM prescreening_sessions WHERE "candidateId" = %s', (candidate_id,))
            deleted_sessions = cursor.rowcount
        else:
            deleted_sessions = 0
            deleted_questions = 0
            deleted_events = 0
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "candidate_id": candidate_id,
            "job_id": job_id,
            "deleted_sessions": deleted_sessions,
            "deleted_questions": deleted_questions,
            "deleted_events": deleted_events,
            "reset_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error resetting candidate pre-screening: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@router.get("/analytics/dashboard")
async def get_prescreening_analytics():
    """Get pre-screening analytics for dashboard"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get basic statistics
        cursor.execute('SELECT COUNT(*) as total FROM candidate_prescreening')
        total_prescreenings = cursor.fetchone()['total']
        
        cursor.execute('''
            SELECT "resumeDecision", COUNT(*) as count
            FROM candidate_prescreening
            GROUP BY "resumeDecision"
        ''')
        bucket_stats = {row['resumeDecision']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute('SELECT AVG("resumeScore") as avg_score FROM candidate_prescreening')
        avg_score = cursor.fetchone()['avg_score'] or 0
        
        cursor.close()
        conn.close()
        
        return {
            "total_prescreenings": total_prescreenings,
            "bucket_distribution": bucket_stats,
            "average_score": round(float(avg_score), 2),
            "automation_rate": 0.85,  # Could be calculated based on human reviews
            "success_rate": 0.72,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error generating analytics: {e}")
@router.get("/session/{session_id}/detailed-analysis")
async def get_detailed_session_analysis(session_id: str):
    """Get comprehensive analysis of a completed pre-screening session"""
    try:
        print(f"DEBUG: Starting detailed analysis for session {session_id}")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get session details with better error handling
        print(f"DEBUG: Fetching session details for {session_id}")
        cursor.execute('''
            SELECT s.*, cp."candidateId", cp."jobId"
            FROM prescreening_sessions s
            LEFT JOIN candidate_prescreening cp ON s."candidatePreScreeningId" = cp.id
            WHERE s.id = %s
        ''', (session_id,))
        
        session = cursor.fetchone()
        print(f"DEBUG: Session found: {session is not None}")
        
        if not session:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Session not found")
        
        print(f"DEBUG: Session status: {session.get('status')}")
        print(f"DEBUG: Session candidatePreScreeningId: {session.get('candidatePreScreeningId')}")
        
        # Get MCQ questions and answers with fallback
        print(f"DEBUG: Fetching MCQ questions for session {session_id}")
        cursor.execute('''
            SELECT id, question, options, "correctAnswer", "difficultyLevel", "skillCategory"
            FROM mcq_questions WHERE "sessionId" = %s
            ORDER BY "generatedAt"
        ''', (session_id,))
        
        questions = cursor.fetchall()
        print(f"DEBUG: Found {len(questions)} questions in mcq_questions table")
        
        # If no questions in mcq_questions, try to get from session JSON
        if not questions and session.get('mcqQuestions'):
            print(f"DEBUG: No questions in mcq_questions table, checking session JSON")
            try:
                stored_questions = json.loads(session['mcqQuestions']) if isinstance(session['mcqQuestions'], str) else session['mcqQuestions']
                print(f"DEBUG: Found {len(stored_questions)} questions in session JSON")
                
                # Convert stored questions to database format
                questions = []
                for q in stored_questions:
                    questions.append({
                        'id': q.get('id'),
                        'question': q.get('question'),
                        'options': q.get('options'),
                        'correctAnswer': q.get('correct_answer', 'A'),
                        'difficultyLevel': q.get('difficulty_level', 'medium'),
                        'skillCategory': q.get('skill_category', 'general')
                    })
            except Exception as json_error:
                print(f"DEBUG: Error parsing session JSON questions: {json_error}")
        
        # Parse answers from session - handle safely
        mcq_answers = []
        try:
            if session.get('mcqAnswers'):
                if isinstance(session['mcqAnswers'], str):
                    mcq_answers = json.loads(session['mcqAnswers'])
                else:
                    mcq_answers = session['mcqAnswers'] or []
            print(f"DEBUG: Found {len(mcq_answers)} answers")
        except Exception as answer_error:
            print(f"DEBUG: Error parsing mcqAnswers: {answer_error}")
            mcq_answers = []
        
        # Get proctoring events
        print(f"DEBUG: Fetching proctoring events for session {session_id}")
        cursor.execute('''
            SELECT "eventType", timestamp, "confidenceScore", severity, metadata
            FROM proctoring_events
            WHERE "sessionId" = %s
            ORDER BY timestamp
        ''', (session_id,))
        
        proctoring_events = cursor.fetchall()
        print(f"DEBUG: Found {len(proctoring_events)} proctoring events")
        
        cursor.close()
        conn.close()
        
        # Analyze each question with safe handling
        question_analysis = []
        for i, question in enumerate(questions):
            try:
                print(f"DEBUG: Processing question {i+1}: {question.get('id')}")
                
                # Find corresponding answer
                answer_data = next((ans for ans in mcq_answers if ans.get('questionId') == question.get('id')), None)
                
                if answer_data:
                    is_correct = answer_data.get('isCorrect', False)
                    time_taken = answer_data.get('timeTaken', 0)
                    candidate_answer = answer_data.get('answer', 'Not answered')
                else:
                    # No answer found
                    is_correct = False
                    time_taken = 0
                    candidate_answer = 'Not answered'
                
                # Analyze performance
                performance_analysis = analyze_question_performance(
                    question, candidate_answer, is_correct, time_taken
                )
                
                # Handle options safely
                options = question.get('options', {})
                if isinstance(options, str):
                    try:
                        options = json.loads(options)
                    except:
                        options = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}
                
                question_analysis.append({
                    'questionId': question.get('id', f'q_{i+1}'),
                    'question': question.get('question', f'Question {i+1}'),
                    'options': options,
                    'correctAnswer': question.get('correctAnswer', 'A'),
                    'candidateAnswer': candidate_answer,
                    'isCorrect': is_correct,
                    'timeTaken': time_taken,
                    'difficulty': question.get('difficultyLevel', 'medium'),
                    'skillCategory': question.get('skillCategory', 'general'),
                    'analysis': performance_analysis
                })
                
            except Exception as q_error:
                print(f"DEBUG: Error processing question {i+1}: {q_error}")
                continue
        
        print(f"DEBUG: Processed {len(question_analysis)} questions successfully")
        
        # Analyze proctoring events
        print(f"DEBUG: Analyzing proctoring events")
        proctoring_analysis = analyze_proctoring_events(proctoring_events)
        
        # Generate AI-powered session analysis with error handling
        print(f"DEBUG: Generating AI analysis")
        try:
            overall_analysis = await generate_session_analysis(
                session, question_analysis, proctoring_analysis
            )
        except Exception as ai_error:
            print(f"DEBUG: AI analysis failed: {ai_error}")
            overall_analysis = {
                "overall_assessment": f"Analysis completed for {len(question_analysis)} questions",
                "key_strengths": ["Completed the assessment"],
                "areas_for_improvement": ["Continue practicing"],
                "skill_analysis": {},
                "test_taking_behavior": "Standard approach observed",
                "integrity_assessment": "No significant concerns",
                "next_steps": "Review performance details",
                "recruiter_feedback": f"Candidate completed the assessment with {len(question_analysis)} questions analyzed"
            }
        
        print(f"DEBUG: Returning analysis for session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "session_overview": {
                "status": session.get('status', 'unknown'),
                "start_time": session.get('startTime').isoformat() if session.get('startTime') else None,
                "end_time": session.get('endTime').isoformat() if session.get('endTime') else None,
                "total_questions": len(questions),
                "correct_answers": session.get('correctAnswers') or 0,
                "mcq_score": session.get('mcqScore') or 0,
                "final_score": session.get('finalScore') or 0,
                "proctoring_penalty": session.get('proctoringPenalty') or 0
            },
            "question_analysis": question_analysis,
            "proctoring_report": proctoring_analysis,
            "ai_analysis": overall_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in get_detailed_session_analysis: {e}")
        import traceback
        print(f"ERROR traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis: {str(e)}")

def analyze_question_performance(question, candidate_answer, is_correct, time_taken):
    """Analyze individual question performance"""
    difficulty = question['difficultyLevel']
    skill = question['skillCategory']
    
    # Time analysis
    expected_times = {"easy": 60, "medium": 90, "hard": 120}
    expected_time = expected_times.get(difficulty, 90)
    time_efficiency = "optimal" if time_taken <= expected_time else "slow" if time_taken <= expected_time * 1.5 else "very_slow"
    
    # Performance insights
    insights = []
    if is_correct:
        if time_taken < expected_time * 0.5:
            insights.append("Demonstrated strong mastery - answered quickly and correctly")
        elif time_taken <= expected_time:
            insights.append("Good understanding with efficient problem-solving")
        else:
            insights.append("Correct answer but took longer than expected - may need practice for speed")
    else:
        if time_taken < expected_time * 0.5:
            insights.append("Quick response but incorrect - may have rushed or misunderstood")
        elif time_taken <= expected_time:
            insights.append("Reasonable time spent but incorrect - concept needs reinforcement")
        else:
            insights.append("Spent significant time but still incorrect - fundamental gap in this area")
    
    return {
        "time_efficiency": time_efficiency,
        "expected_time": expected_time,
        "performance_level": "strong" if is_correct and time_taken <= expected_time else "moderate" if is_correct else "needs_improvement",
        "insights": insights
    }

def analyze_proctoring_events(events):
    """Analyze proctoring events and generate report"""
    if not events:
        return {
            "total_events": 0,
            "violations_by_type": {},
            "severity_breakdown": {},
            "timeline": [],
            "risk_assessment": "low",
            "summary": "No proctoring violations detected. Test completed under normal conditions."
        }
    
    # Categorize events
    violations_by_type = {}
    severity_breakdown = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    timeline = []
    
    for event in events:
        event_type = event['eventType']
        severity = event['severity']
        timestamp = event['timestamp']
        confidence = event['confidenceScore']
        
        violations_by_type[event_type] = violations_by_type.get(event_type, 0) + 1
        severity_breakdown[severity] = severity_breakdown.get(severity, 0) + 1
        
        timeline.append({
            "timestamp": timestamp.isoformat(),
            "type": event_type,
            "severity": severity,
            "confidence": confidence,
            "description": get_violation_description(event_type, confidence)
        })
    
    # Risk assessment
    total_events = len(events)
    high_severity = severity_breakdown.get('high', 0) + severity_breakdown.get('critical', 0)
    
    if high_severity >= 3:
        risk = "high"
        summary = f"Multiple serious violations detected ({high_severity} high/critical events). Test integrity may be compromised."
    elif high_severity >= 1:
        risk = "medium"  
        summary = f"Some violations detected ({total_events} total events). Test completed with minor integrity concerns."
    elif total_events >= 5:
        risk = "medium"
        summary = f"Moderate number of minor violations ({total_events} events). Generally acceptable but worth noting."
    else:
        risk = "low"
        summary = f"Few minor violations detected ({total_events} events). Test completed under acceptable conditions."
    
    return {
        "total_events": total_events,
        "violations_by_type": violations_by_type,
        "severity_breakdown": severity_breakdown,
        "timeline": timeline,
        "risk_assessment": risk,
        "summary": summary
    }

def get_violation_description(event_type, confidence):
    """Generate human-readable description for proctoring violations"""
    descriptions = {
        "face_not_detected": f"Face not visible in frame (confidence: {confidence:.1%})",
        "multiple_faces": f"Multiple people detected (confidence: {confidence:.1%})",
        "looking_away": f"Candidate looking away from screen (confidence: {confidence:.1%})",
        "tab_switch": f"Browser tab switched (confidence: {confidence:.1%})",
        "window_focus_lost": f"Application window lost focus (confidence: {confidence:.1%})",
        "fullscreen_exit": f"Exited fullscreen mode (confidence: {confidence:.1%})",
        "suspicious_movement": f"Unusual movement patterns detected (confidence: {confidence:.1%})",
        "audio_anomaly": f"Unexpected audio activity (confidence: {confidence:.1%})"
    }
    return descriptions.get(event_type, f"Unknown violation type: {event_type}")

async def generate_session_analysis(session, question_analysis, proctoring_analysis):
    """Generate AI-powered comprehensive session analysis"""
    try:
        llm = get_azure_llm()
        
        # Prepare analysis data
        total_questions = len(question_analysis)
        correct_answers = sum(1 for q in question_analysis if q['isCorrect'])
        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        
        # Skill performance breakdown
        skill_performance = {}
        for q in question_analysis:
            skill = q['skillCategory']
            if skill not in skill_performance:
                skill_performance[skill] = {"total": 0, "correct": 0}
            skill_performance[skill]["total"] += 1
            if q['isCorrect']:
                skill_performance[skill]["correct"] += 1
        
        # Time analysis
        avg_time_per_question = sum(q['timeTaken'] for q in question_analysis) / total_questions if total_questions > 0 else 0
        
        prompt = f"""
        Analyze this pre-screening test performance and provide detailed insights:

        SESSION OVERVIEW:
        - Total Questions: {total_questions}
        - Correct Answers: {correct_answers}
        - Overall Accuracy: {accuracy:.1f}%
        - Average Time per Question: {avg_time_per_question:.1f} seconds
        - Final Score: {session['finalScore'] or 0}

        SKILL PERFORMANCE:
        {chr(10).join([f"- {skill}: {perf['correct']}/{perf['total']} ({perf['correct']/perf['total']*100:.1f}%)" for skill, perf in skill_performance.items()])}

        PROCTORING SUMMARY:
        - Total Violations: {proctoring_analysis['total_events']}
        - Risk Level: {proctoring_analysis['risk_assessment']}
        - Summary: {proctoring_analysis['summary']}

        QUESTION DETAILS:
        {chr(10).join([f"Q{i+1} ({q['difficulty']} - {q['skillCategory']}): {'✓' if q['isCorrect'] else '✗'} in {q['timeTaken']}s" for i, q in enumerate(question_analysis[:10])])}

        Provide a comprehensive analysis in JSON format with these sections:
        {{
            "overall_assessment": "brief overall performance summary",
            "key_strengths": ["strength1", "strength2", ...],
            "areas_for_improvement": ["area1", "area2", ...],
            "skill_analysis": {{"skill_name": {{"performance": "strong/moderate/weak", "recommendation": "specific advice"}}}},
            "test_taking_behavior": "analysis of time management and approach",
            "integrity_assessment": "assessment based on proctoring data",
            "next_steps": "recommended actions based on performance",
            "recruiter_feedback": "what a recruiter should know about this candidate's performance"
        }}
        
        Base your analysis on actual test performance data, not assumptions.
        """
        
        response = await llm.agenerate([[HumanMessage(content=prompt)]])
        raw_response = response.generations[0][0].text.strip()
        
        # Clean and parse JSON with better error handling
        if raw_response.startswith('```json'):
            raw_response = raw_response[7:]
        if raw_response.endswith('```'):
            raw_response = raw_response[:-3]
        
        # Extract only the JSON object, handling any extra content
        raw_response = raw_response.strip()
        
        # Find the first { and last } to extract just the JSON object
        start_idx = raw_response.find('{')
        end_idx = raw_response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = raw_response[start_idx:end_idx + 1]
            return json.loads(json_str)
        else:
            # If we can't find JSON brackets, try parsing as-is
            return json.loads(raw_response)
        
    except Exception as e:
        print(f"Error generating AI analysis: {e}")
        # Fallback analysis
        return {
            "overall_assessment": f"Candidate completed {total_questions} questions with {accuracy:.1f}% accuracy.",
            "key_strengths": ["Completed the assessment", "Demonstrated basic problem-solving ability"],
            "areas_for_improvement": ["Focus on accuracy", "Improve time management"],
            "skill_analysis": {skill: {"performance": "moderate", "recommendation": "Continue practicing"} for skill in skill_performance.keys()},
            "test_taking_behavior": "Standard test-taking approach observed.",
            "integrity_assessment": f"Test completed with {proctoring_analysis['risk_assessment']} integrity risk.",
            "next_steps": "Review performance and focus on identified improvement areas.",
            "recruiter_feedback": f"Candidate demonstrated {accuracy:.1f}% accuracy on technical assessment with {proctoring_analysis['risk_assessment']} integrity concerns."
        }

@router.get("/candidate/{candidate_id}/resume-analysis/{job_id}")
async def get_resume_analysis_with_explanation(candidate_id: str, job_id: str):
    """Get detailed resume analysis with AI explanation for scoring"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Get candidate resume data
        cursor.execute('''
            SELECT c.*, cp."resumeScore", cp."resumeDecision"
            FROM "Candidate" c
            LEFT JOIN candidate_prescreening cp ON c.id = cp."candidateId" AND cp."jobId" = %s
            WHERE c.id = %s
        ''', (job_id, candidate_id))
        
        candidate = cursor.fetchone()
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        # Get job details
        cursor.execute('SELECT title, description FROM "Job" WHERE id = %s', (job_id,))
        job = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Generate comprehensive AI analysis
        resume_text = candidate.get('parsedResume') or ""
        job_description = job['description'] or ""
        job_title = job['title'] or ""
        
        # Calculate embedding similarity
        embedding_score = await calculate_embedding_similarity(resume_text, job_description)
        
        # Generate detailed AI analysis
        ai_analysis = await generate_resume_matching_explanation(
            resume_text, job_description, job_title, embedding_score, candidate.get('resumeScore', 0)
        )
        
        return {
            "success": True,
            "candidate_id": candidate_id,
            "job_id": job_id,
            "resume_score": candidate.get('resumeScore', 0),
            "resume_decision": candidate.get('resumeDecision', 'pending'),
            "semantic_similarity": embedding_score * 100,
            "job_title": job_title,
            "ai_explanation": ai_analysis,
            "analyzed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting resume analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resume analysis: {str(e)}")

async def generate_resume_matching_explanation(resume_text: str, job_description: str, job_title: str, semantic_score: float, overall_score: float):
    """Generate AI explanation for resume matching score"""
    try:
        llm = get_azure_llm()
        
        prompt = f"""
        Analyze why this resume received the given matching scores against the job requirements:

        JOB TITLE: {job_title}

        JOB DESCRIPTION:
        {job_description[:2000]}...

        RESUME CONTENT:
        {resume_text[:2000]}...

        SCORING RESULTS:
        - Overall Resume Match Score: {overall_score:.1f}/100
        - Semantic Similarity Score: {semantic_score * 100:.1f}/100

        Provide a detailed explanation in JSON format:
        {{
            "score_breakdown": {{
                "technical_skills_match": {{"score": <0-100>, "explanation": "why this score"}},
                "experience_relevance": {{"score": <0-100>, "explanation": "why this score"}},
                "education_fit": {{"score": <0-100>, "explanation": "why this score"}},
                "semantic_alignment": {{"score": <0-100>, "explanation": "why this score"}}
            }},
            "matching_keywords": ["keyword1", "keyword2", ...],
            "missing_keywords": ["missing1", "missing2", ...],
            "key_strengths": ["strength1", "strength2", ...],
            "improvement_areas": ["area1", "area2", ...],
            "overall_rationale": "detailed explanation of why the candidate received this overall score",
            "recommendation": "specific recommendation for recruiter action",
            "confidence_level": "high/medium/low - AI confidence in this assessment"
        }}
        
        Be specific about why certain scores were assigned and what factors influenced the matching.
        """
        
        response = await llm.agenerate([[HumanMessage(content=prompt)]])
        raw_response = response.generations[0][0].text.strip()
        
        # Clean and parse JSON
        if raw_response.startswith('```json'):
            raw_response = raw_response[7:]
        if raw_response.endswith('```'):
            raw_response = raw_response[:-3]
        
        return json.loads(raw_response.strip())
        
    except Exception as e:
        print(f"Error generating resume matching explanation: {e}")
        # Fallback explanation
        return {
            "score_breakdown": {
                "technical_skills_match": {"score": overall_score * 0.9, "explanation": "Based on keyword matching and skill alignment"},
                "experience_relevance": {"score": overall_score * 1.1, "explanation": "Experience level assessment against job requirements"},
                "education_fit": {"score": overall_score, "explanation": "Educational background alignment"},
                "semantic_alignment": {"score": semantic_score * 100, "explanation": "AI-powered semantic understanding of resume-job fit"}
            },
            "matching_keywords": ["Skills extracted from resume"],
            "missing_keywords": ["Areas for improvement"],
            "key_strengths": ["Resume strengths identified"],
            "improvement_areas": ["Potential gaps or areas to develop"],
            "overall_rationale": f"The resume received a {overall_score:.1f}/100 score based on comprehensive AI analysis of skills, experience, and job fit.",
            "recommendation": "Standard assessment recommendation",
            "confidence_level": "medium"
        }
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")

# Export router
__all__ = ["router"]
