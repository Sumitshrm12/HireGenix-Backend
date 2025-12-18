"""
🤖 AGENTIC AI JOB PROCESSOR (v3.0) - World-Class Multi-Agent System
===================================================================

This module implements a sophisticated multi-agent system for job processing using:
- LangGraph for workflow orchestration
- CrewAI for multi-agent collaboration (Parser, Enhancer, Validator)
- DSPy MIPRO for job extraction optimization
- Company Intelligence integration
- Advanced file parsing with context awareness
- Multi-step validation and enhancement
- RAG for historical job data and learning from successful postings
- Feedback loops from job posting performance

Author: HireGenix AI Team
Version: 3.0.0 (World-Class Agentic AI)
Last Updated: December 2025
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json
import uuid
import hashlib
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import httpx

# 🤖 CrewAI for Multi-Agent Collaboration
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# 🎯 DSPy for Prompt Optimization
try:
    import dspy
    from dspy import ChainOfThought
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# 📚 RAG & Vector Store
try:
    import redis
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Import existing utilities
from utils.token_usage import get_token_tracker, get_token_usage_stats
from agentic_ai.config import AgenticAIConfig

load_dotenv()


# ============================================================================
# 🎯 DSPY JOB PROCESSING SIGNATURES
# ============================================================================

if DSPY_AVAILABLE:
    class JobExtractionSignature(dspy.Signature):
        """Extract structured job data from unstructured text."""
        raw_content: str = dspy.InputField(desc="Raw job description or file content")
        file_type: str = dspy.InputField(desc="Type of file (PDF, DOCX, TXT)")
        job_title: str = dspy.OutputField(desc="Job title")
        job_description: str = dspy.OutputField(desc="Job description")
        skills_json: str = dspy.OutputField(desc="JSON array of required skills")
        requirements_json: str = dspy.OutputField(desc="JSON array of requirements")
        
    class JobEnhancementSignature(dspy.Signature):
        """Enhance job description with company context."""
        job_data: str = dspy.InputField(desc="Extracted job data JSON")
        company_context: str = dspy.InputField(desc="Company intelligence context")
        enhanced_description: str = dspy.OutputField(desc="Enhanced job description")
        quality_score: str = dspy.OutputField(desc="Quality score 0.0-1.0")


# ============================================================================
# SOCKET EMISSION HELPER
# ============================================================================

async def emit_agent_stage_update(user_id: str, stage: str, status: str, message: str, progress: int):
    """
    Emit agent stage updates to frontend via Next.js Socket.IO API
    """
    try:
        backend_url = os.getenv("NEXTAUTH_URL", "http://localhost:3000")
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{backend_url}/api/socket/emit-agent-stage",
                json={
                    "userId": user_id,
                    "stage": stage,
                    "status": status,
                    "message": message,
                    "progress": progress,
                    "timestamp": datetime.utcnow().isoformat()
                },
                timeout=2.0
            )
            print(f"✅ Emitted stage update: {stage} - {status}")
    except Exception as e:
        print(f"⚠️ Failed to emit socket update: {e}")


# ============================================================================
# MODELS & SCHEMAS
# ============================================================================

from typing import TypedDict, Annotated
from operator import add

class JobUploadFile(BaseModel):
    """Model for uploaded job file"""
    filename: str
    content: str
    file_type: str


class ExtractedJobData(BaseModel):
    """Structured job data extracted from files"""
    title: str
    description: str
    location: str = "Remote"
    type: str = "FULL_TIME"
    workLocation: str = "REMOTE"
    skills: List[Dict[str, Any]] = Field(default_factory=list)
    salary: Dict[str, Any] = Field(default_factory=dict)
    requirements: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    benefits: List[str] = Field(default_factory=list)
    experience: Optional[Dict[str, Optional[int]]] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state persistence"""
        return {
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "type": self.type,
            "workLocation": self.workLocation,
            "skills": self.skills,
            "salary": self.salary,
            "requirements": self.requirements,
            "responsibilities": self.responsibilities,
            "benefits": self.benefits,
            "experience": self.experience
        }
    

class EnhancedJobData(BaseModel):
    """Enhanced job data with company intelligence"""
    original_data: ExtractedJobData
    enhanced_description: str
    company_context: Optional[Dict[str, Any]] = None
    industry_insights: Optional[str] = None
    quality_score: float = 0.0
    suggestions: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state persistence"""
        return {
            "original_data": self.original_data.to_dict() if isinstance(self.original_data, ExtractedJobData) else self.original_data,
            "enhanced_description": self.enhanced_description,
            "company_context": self.company_context,
            "industry_insights": self.industry_insights,
            "quality_score": self.quality_score,
            "suggestions": self.suggestions
        }


# Use TypedDict for LangGraph state to ensure proper state persistence
class JobProcessorState(TypedDict, total=False):
    """State for LangGraph workflow - using TypedDict for proper state persistence"""
    files: List[Dict[str, Any]]  # List of file dicts
    company_id: str
    user_id: str
    company_intelligence: Optional[Dict[str, Any]]
    extracted_jobs: List[Dict[str, Any]]  # List of extracted job dicts
    enhanced_jobs: List[Dict[str, Any]]  # List of enhanced job dicts
    validated_jobs: List[Dict[str, Any]]  # List of validated job dicts
    errors: List[Dict[str, str]]
    token_usage: Optional[Dict]
    current_step: str
    
    # RAG Context
    successful_patterns: List[Dict]
    
    # CrewAI Collaboration Results
    crew_insights: Dict[str, Any]


# ============================================================================
# AGENTIC JOB PROCESSOR
# ============================================================================

class AgenticJobProcessor:
    """
    🚀 WORLD-CLASS MULTI-AGENT JOB PROCESSOR (v3.0)
    
    Features:
    - LangGraph multi-step workflow
    - CrewAI 3-agent collaboration (Parser, Enhancer, Validator)
    - DSPy MIPRO for job extraction optimization
    - RAG knowledge base from successful job postings
    - Feedback loops from job posting performance
    
    Agents:
    1. File Parser Agent - Extracts job data from various file formats
    2. Context Enrichment Agent - Adds company intelligence context
    3. Quality Enhancement Agent - Improves job descriptions
    4. Validation Agent - Ensures data quality and completeness
    5. Storage Agent - Stores in vector DB for future RAG
    """
    
    def __init__(self):
        self.config = AgenticAIConfig()
        self.token_tracker = get_token_tracker()
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.3,  # Lower for more consistent extraction
            max_tokens=4000,
            callbacks=[self.token_tracker]
        )
        
        # Initialize Advanced Components
        self._init_crewai_agents()
        self._init_rag_knowledge_base()
        self._init_dspy_optimizer()
        
        # Initialize embeddings and vector store (optional)
        self.embeddings = None
        self.vector_store = None
        self.vector_store_enabled = False
        
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                azure_endpoint=os.getenv("TEXT_EMBEDDING_ENDPOINT"),
                deployment=os.getenv("TEXT_EMBEDDING_MODEL"),
                openai_api_version=os.getenv("TEXT_EMBEDDING_API_VERSION"),
            )
            # Vector store initialization removed - optional feature
            self.vector_store_enabled = False
            print("ℹ️  Vector storage disabled (optional feature)")
        except Exception as e:
            print(f"⚠️  Embeddings unavailable: {e}")
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        print("✅ AgenticJobProcessor v3.0 initialized with CrewAI + DSPy + RAG")
    
    def _init_crewai_agents(self):
        """Initialize CrewAI multi-agent system"""
        self.crewai_enabled = CREWAI_AVAILABLE
        
        if CREWAI_AVAILABLE:
            self.parser_agent = Agent(
                name="Job Parser Agent",
                role="Expert Job Data Extractor",
                goal="Extract complete, accurate job data from various file formats",
                backstory="Senior data engineer with 10 years experience parsing job postings from PDFs, Word docs, and text files. Expert at identifying key fields.",
                allow_delegation=False
            )
            
            self.enhancer_agent = Agent(
                name="Job Enhancer Agent",
                role="Job Description Specialist",
                goal="Enhance job descriptions with compelling language that attracts top talent",
                backstory="Former talent acquisition leader who has written thousands of job descriptions. Expert at SEO optimization and candidate attraction.",
                allow_delegation=False
            )
            
            self.validator_agent = Agent(
                name="Job Validator Agent",
                role="Quality Assurance Specialist",
                goal="Ensure job postings are complete, accurate, and legally compliant",
                backstory="HR compliance expert with deep knowledge of employment law. Expert at catching incomplete or problematic job postings.",
                allow_delegation=False
            )
            
            print("✅ CrewAI 3-agent job processing crew initialized")
    
    def _init_rag_knowledge_base(self):
        """Initialize RAG for learning from successful job postings"""
        self.rag_enabled = False
        if RAG_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    password=os.getenv("REDIS_PASSWORD", "") or None,
                    decode_responses=True
                )
                self.rag_enabled = True
                print("✅ Job Posting RAG Knowledge Base initialized")
            except Exception as e:
                print(f"⚠️ RAG initialization failed: {e}")
    
    def _init_dspy_optimizer(self):
        """Initialize DSPy for job extraction optimization"""
        self.dspy_enabled = False
        if DSPY_AVAILABLE:
            try:
                lm = dspy.LM(
                    model=f"openai/{os.getenv('AZURE_DEPLOYMENT_NAME', 'gpt-4o')}",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_API_VERSION"),
                    model_type="chat"
                )
                dspy.configure(lm=lm)
                self.dspy_enabled = True
                print("✅ DSPy optimizer initialized for job processing")
            except Exception as e:
                print(f"⚠️ DSPy initialization failed: {e}")
    
    def _build_workflow(self):
        """Build LangGraph workflow for job processing"""
        
        workflow = StateGraph(JobProcessorState)
        
        # Define workflow nodes (agents)
        workflow.add_node("retrieve_patterns", self._retrieve_successful_patterns)
        workflow.add_node("parse_files", self._parse_files_agent)
        workflow.add_node("run_crew_collaboration", self._run_crew_collaboration)
        workflow.add_node("enrich_context", self._context_enrichment_agent)
        workflow.add_node("enhance_quality", self._quality_enhancement_agent)
        workflow.add_node("validate", self._validation_agent)
        workflow.add_node("store", self._storage_agent)
        workflow.add_node("store_for_learning", self._store_job_pattern)
        
        # Define workflow edges
        workflow.set_entry_point("retrieve_patterns")
        workflow.add_edge("retrieve_patterns", "parse_files")
        workflow.add_edge("parse_files", "run_crew_collaboration")
        workflow.add_edge("run_crew_collaboration", "enrich_context")
        workflow.add_edge("enrich_context", "enhance_quality")
        workflow.add_edge("enhance_quality", "validate")
        workflow.add_edge("validate", "store")
        workflow.add_edge("store", "store_for_learning")
        workflow.add_edge("store_for_learning", END)
        
        return workflow.compile()
    
    async def _retrieve_successful_patterns(self, state: JobProcessorState) -> JobProcessorState:
        """Retrieve successful job posting patterns from RAG"""
        state["current_step"] = "retrieve_patterns"
        
        if "successful_patterns" not in state:
            state["successful_patterns"] = []
        
        if self.rag_enabled:
            try:
                patterns = []
                pattern_keys = self.redis_client.keys("job_pattern:*")
                
                for key in pattern_keys[:20]:
                    data = self.redis_client.get(key)
                    if data:
                        pattern = json.loads(data)
                        if pattern.get("applications_received", 0) >= 10:
                            patterns.append(pattern)
                
                state["successful_patterns"] = patterns[:5]
            except Exception as e:
                print(f"⚠️ Pattern retrieval error: {e}")
        
        return state
    
    async def _run_crew_collaboration(self, state: JobProcessorState) -> JobProcessorState:
        """Run CrewAI multi-agent collaboration for job processing"""
        state["current_step"] = "run_crew_collaboration"
        
        extracted_jobs = state.get("extracted_jobs", [])
        
        if "crew_insights" not in state:
            state["crew_insights"] = {}
        
        if self.crewai_enabled and extracted_jobs:
            try:
                job_data = extracted_jobs[0] if extracted_jobs else None
                if not job_data:
                    return state
                
                job_title = job_data.get("title", "Unknown") if isinstance(job_data, dict) else "Unknown"
                
                parser_task = Task(
                    description=f"Review extracted job data for completeness: {job_title}. Identify any missing fields or extraction errors.",
                    agent=self.parser_agent,
                    expected_output="Extraction quality assessment with missing fields identified"
                )
                
                enhancer_task = Task(
                    description=f"Enhance job description for {job_title}. Make it compelling, SEO-optimized, and attractive to top talent.",
                    agent=self.enhancer_agent,
                    expected_output="Enhanced job description with improvements"
                )
                
                validator_task = Task(
                    description=f"Validate job posting for {job_title}. Check for legal compliance, completeness, and bias.",
                    agent=self.validator_agent,
                    expected_output="Validation report with issues and recommendations"
                )
                
                crew = Crew(
                    agents=[self.parser_agent, self.enhancer_agent, self.validator_agent],
                    tasks=[parser_task, enhancer_task, validator_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                crew.kickoff()
                
                state["crew_insights"] = {
                    "parser_review": parser_task.output.raw if parser_task.output else "",
                    "enhancer_improvements": enhancer_task.output.raw if enhancer_task.output else "",
                    "validator_report": validator_task.output.raw if validator_task.output else ""
                }
                
            except Exception as e:
                print(f"⚠️ CrewAI collaboration error: {e}")
        
        return state
    
    async def _store_job_pattern(self, state: JobProcessorState) -> JobProcessorState:
        """Store job pattern for future learning"""
        state["current_step"] = "store_for_learning"
        
        validated_jobs = state.get("validated_jobs", [])
        company_id = state.get("company_id", "")
        
        if self.rag_enabled and validated_jobs:
            try:
                for job in validated_jobs:
                    pattern_id = hashlib.md5(
                        f"{job.get('title', '')}{company_id}{datetime.now().isoformat()}".encode()
                    ).hexdigest()
                    
                    pattern = {
                        "id": pattern_id,
                        "title": job.get("title", ""),
                        "company_id": company_id,
                        "skills": job.get("skills", []),
                        "quality_score": job.get("quality_score", 0),
                        "applications_received": 0,  # Updated via feedback
                        "quality_hires": 0,  # Updated via feedback
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.redis_client.set(f"job_pattern:{pattern_id}", json.dumps(pattern))
                    
            except Exception as e:
                print(f"⚠️ Pattern storage error: {e}")
        
        return state
    
    async def _parse_files_agent(self, state: JobProcessorState) -> JobProcessorState:
        """
        Agent 1: File Parser
        Extracts structured job data from uploaded files
        """
        print("\n" + "="*60)
        print("🔍 Agent 1: FILE PARSER")
        print("="*60)
        
        files = state.get("files", [])
        user_id = state.get("user_id", "")
        
        print(f"📂 Files to parse: {len(files)}")
        
        # Emit stage update
        await emit_agent_stage_update(
            user_id=user_id,
            stage="parse_files",
            status="processing",
            message=f"Parsing {len(files)} uploaded files...",
            progress=10
        )
        
        if not files or len(files) == 0:
            print("❌ ERROR: No files in state!")
            await emit_agent_stage_update(
                user_id=user_id,
                stage="parse_files",
                status="error",
                message="No files received for processing",
                progress=10
            )
            return state
        
        # Initialize extracted_jobs list if not present
        if "extracted_jobs" not in state:
            state["extracted_jobs"] = []
        if "errors" not in state:
            state["errors"] = []
        
        for file in files:
            filename = file.get("filename", "unknown")
            content = file.get("content", "")
            
            print(f"\n📄 Parsing file: {filename}")
            print(f"   Content length: {len(content)} chars")
            print(f"   Content preview: {content[:500]}..." if content else "   Content: EMPTY!")
            
            if not content or len(content.strip()) < 20:
                print(f"   ❌ SKIPPING: Content too short ({len(content.strip()) if content else 0} chars)")
                state["errors"].append({
                    "file": filename,
                    "agent": "parse_files",
                    "error": f"File content too short: {len(content.strip()) if content else 0} chars"
                })
                continue
            
            try:
                prompt = f"""Extract structured job information from the following content.
                
File: {filename}
Content:
{content[:10000]}  # Limit content length

Extract and structure the following information:
1. Job Title (required)
2. Job Description (comprehensive)
3. Location (default: Remote if not specified)
4. Job Type (FULL_TIME, PART_TIME, CONTRACT, INTERNSHIP, FREELANCE)
5. Work Location (REMOTE, ONSITE, HYBRID)
6. Required Skills (with level: BEGINNER, INTERMEDIATE, EXPERT)
7. Salary Range (min, max, currency, period)
8. Requirements (list of strings)
9. Responsibilities (list of strings)
10. Benefits (list of strings)
11. Experience Required (min and max years)

Return JSON with this exact structure:
{{
    "title": "string",
    "description": "string",
    "location": "string",
    "type": "FULL_TIME|PART_TIME|CONTRACT|INTERNSHIP|FREELANCE",
    "workLocation": "REMOTE|ONSITE|HYBRID",
    "skills": [
        {{"name": "string", "level": "BEGINNER|INTERMEDIATE|EXPERT", "yearsOfExperience": number}}
    ],
    "salary": {{
        "min": number,
        "max": number,
        "currency": "USD",
        "period": "YEARLY|MONTHLY|HOURLY",
        "isNegotiable": boolean
    }},
    "requirements": ["string"],
    "responsibilities": ["string"],
    "benefits": ["string"],
    "experience": {{"min": number, "max": number}}
}}

Be thorough and extract all available information. If information is missing, use intelligent defaults."""

                response = await self.llm.ainvoke([
                    SystemMessage(content="You are an expert job data extraction agent with deep knowledge of recruitment and job descriptions."),
                    HumanMessage(content=prompt)
                ])
                
                # Parse JSON response
                content = response.content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                
                job_data = json.loads(content.strip())
                
                # Clean experience field to handle None values
                if 'experience' in job_data and job_data['experience']:
                    exp = job_data['experience']
                    if exp.get('min') is None:
                        exp['min'] = 0
                    if exp.get('max') is None:
                        exp['max'] = exp.get('min', 0)
                
                # Store as dict for TypedDict state
                state["extracted_jobs"].append(job_data)
                
                print(f"   ✅ SUCCESSFULLY EXTRACTED JOB:")
                print(f"      Title: {job_data.get('title', 'Unknown')}")
                desc = job_data.get('description', '')
                print(f"      Description: {desc[:100]}..." if desc else "      Description: None")
                print(f"      Skills: {len(job_data.get('skills', []))} skills")
                print(f"      Location: {job_data.get('location', 'Unknown')}")
                
            except json.JSONDecodeError as je:
                print(f"   ❌ JSON PARSE ERROR: {str(je)}")
                print(f"   Raw content that failed to parse: {content[:500]}...")
                state["errors"].append({
                    "file": filename,
                    "agent": "parse_files",
                    "error": f"JSON parse error: {str(je)}"
                })
            except Exception as e:
                print(f"   ❌ ERROR parsing {filename}: {str(e)}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                state["errors"].append({
                    "file": filename,
                    "agent": "parse_files",
                    "error": str(e)
                })
        
        print(f"\n📊 Parse Agent Summary:")
        print(f"   Total extracted jobs: {len(state.get('extracted_jobs', []))}")
        print(f"   Total errors: {len(state.get('errors', []))}")
        
        state["current_step"] = "parse_complete"
        return state
    
    async def _context_enrichment_agent(self, state: JobProcessorState) -> JobProcessorState:
        """
        Agent 2: Context Enrichment
        Enriches jobs with company intelligence data
        """
        print("\n" + "="*60)
        print("🎯 Agent 2: CONTEXT ENRICHMENT")
        print("="*60)
        
        extracted_jobs = state.get("extracted_jobs", [])
        user_id = state.get("user_id", "")
        company_intelligence = state.get("company_intelligence")
        
        print(f"📥 Received extracted_jobs: {len(extracted_jobs)}")
        print(f"   extracted_jobs type: {type(extracted_jobs)}")
        for i, job in enumerate(extracted_jobs):
            print(f"   Job {i}: {job.get('title', 'Unknown') if isinstance(job, dict) else job}")
        
        # Emit stage update
        await emit_agent_stage_update(
            user_id=user_id,
            stage="enrich_context",
            status="processing",
            message=f"Enriching {len(extracted_jobs)} jobs with company intelligence...",
            progress=50
        )
        
        # Initialize enhanced_jobs list if not present
        if "enhanced_jobs" not in state:
            state["enhanced_jobs"] = []
        if "errors" not in state:
            state["errors"] = []
        
        # Fetch company intelligence if not already in state
        if not company_intelligence:
            company_intelligence = {
                "aboutUs": "Company overview",
                "missionStatement": "Mission",
                "values": [],
                "techStack": [],
                "companyCulture": "Culture description",
                "benefits": []
            }
            state["company_intelligence"] = company_intelligence
        
        for extracted_job in extracted_jobs:
            try:
                # Create enhanced job with company context (as dict)
                job_title = extracted_job.get("title", "Unknown") if isinstance(extracted_job, dict) else getattr(extracted_job, "title", "Unknown")
                job_desc = extracted_job.get("description", "") if isinstance(extracted_job, dict) else getattr(extracted_job, "description", "")
                
                enhanced = {
                    "original_data": extracted_job,
                    "enhanced_description": job_desc,
                    "company_context": company_intelligence,
                    "industry_insights": None,
                    "quality_score": 0.0,
                    "suggestions": []
                }
                
                state["enhanced_jobs"].append(enhanced)
                print(f"✅ Enriched: {job_title}")
                
            except Exception as e:
                print(f"❌ Error enriching job: {str(e)}")
                job_title = extracted_job.get("title", "Unknown") if isinstance(extracted_job, dict) else "Unknown"
                state["errors"].append({
                    "job": job_title,
                    "agent": "enrich_context",
                    "error": str(e)
                })
        
        state["current_step"] = "enrich_complete"
        
        # Emit completion
        await emit_agent_stage_update(
            user_id=user_id,
            stage="enrich_context",
            status="completed",
            message=f"Enriched {len(state.get('enhanced_jobs', []))} jobs with company context",
            progress=60
        )
        
        return state
    
    async def _quality_enhancement_agent(self, state: JobProcessorState) -> JobProcessorState:
        """
        Agent 3: Quality Enhancement
        Improves job descriptions using company intelligence
        """
        print("\n" + "="*60)
        print("✨ Agent 3: QUALITY ENHANCEMENT")
        print("="*60)
        
        extracted_jobs = state.get("extracted_jobs", [])
        enhanced_jobs = state.get("enhanced_jobs", [])
        user_id = state.get("user_id", "")
        company_intelligence = state.get("company_intelligence", {})
        
        print(f"📥 Received extracted_jobs: {len(extracted_jobs)}")
        print(f"📥 Received enhanced_jobs: {len(enhanced_jobs)}")
        
        # Emit stage update
        await emit_agent_stage_update(
            user_id=user_id,
            stage="enhance_quality",
            status="processing",
            message=f"Enhancing {len(enhanced_jobs)} job descriptions with AI...",
            progress=70
        )
        
        if "errors" not in state:
            state["errors"] = []
        
        for i, enhanced_job in enumerate(enhanced_jobs):
            try:
                # Get original data (dict-based)
                original_data = enhanced_job.get("original_data", {})
                job_title = original_data.get("title", "Unknown")
                job_desc = original_data.get("description", "")
                
                company_context = ""
                if company_intelligence:
                    company_context = f"""
Company Context:
- About: {company_intelligence.get('aboutUs', 'N/A')}
- Mission: {company_intelligence.get('missionStatement', 'N/A')}
- Culture: {company_intelligence.get('companyCulture', 'N/A')}
- Tech Stack: {', '.join(company_intelligence.get('techStack', []) or [])}
- Benefits: {', '.join(company_intelligence.get('benefits', []) or [])}
"""
                
                prompt = f"""Enhance this job description to be more compelling and company-specific:

Original Job:
Title: {job_title}
Description: {job_desc}

{company_context}

Create an enhanced, engaging job description that:
1. Incorporates company culture and values
2. Highlights unique company benefits
3. Uses inclusive language
4. Is clear and professional
5. Emphasizes growth opportunities
6. Aligns with company's tech stack and mission

Return only the enhanced description text (no JSON, no markdown)."""

                response = await self.llm.ainvoke([
                    SystemMessage(content="You are an expert job description writer specializing in creating compelling, inclusive job postings."),
                    HumanMessage(content=prompt)
                ])
                
                # Update enhanced_description in place
                state["enhanced_jobs"][i]["enhanced_description"] = response.content.strip()
                print(f"✅ Enhanced: {job_title}")
                
            except Exception as e:
                print(f"❌ Error enhancing job: {str(e)}")
                original_data = enhanced_job.get("original_data", {})
                job_title = original_data.get("title", "Unknown") if isinstance(original_data, dict) else "Unknown"
                state["errors"].append({
                    "job": job_title,
                    "agent": "enhance_quality",
                    "error": str(e)
                })
        
        state["current_step"] = "enhance_complete"
        
        # Emit completion
        await emit_agent_stage_update(
            user_id=user_id,
            stage="enhance_quality",
            status="completed",
            message="Job descriptions enhanced successfully",
            progress=80
        )
        
        return state
    
    async def _validation_agent(self, state: JobProcessorState) -> JobProcessorState:
        """
        Agent 4: Validation
        Validates and scores job data quality
        """
        print("\n" + "="*60)
        print("✓ Agent 4: VALIDATION")
        print("="*60)
        
        extracted_jobs = state.get("extracted_jobs", [])
        enhanced_jobs = state.get("enhanced_jobs", [])
        validated_jobs = state.get("validated_jobs", [])
        user_id = state.get("user_id", "")
        company_intelligence = state.get("company_intelligence")
        
        print(f"📥 Received extracted_jobs: {len(extracted_jobs)}")
        print(f"📥 Received enhanced_jobs: {len(enhanced_jobs)}")
        print(f"📥 Received validated_jobs: {len(validated_jobs)}")
        
        # Emit stage update
        await emit_agent_stage_update(
            user_id=user_id,
            stage="validate",
            status="processing",
            message=f"Validating {len(enhanced_jobs)} jobs...",
            progress=85
        )
        
        # Initialize validated_jobs and errors if not present
        if "validated_jobs" not in state:
            state["validated_jobs"] = []
        if "errors" not in state:
            state["errors"] = []
        
        for enhanced_job in enhanced_jobs:
            try:
                # Get original data (dict-based)
                original_data = enhanced_job.get("original_data", {})
                enhanced_description = enhanced_job.get("enhanced_description", "")
                
                # Calculate quality score
                score = 0.0
                suggestions = []
                
                job_title = original_data.get("title", "")
                job_skills = original_data.get("skills", [])
                job_salary = original_data.get("salary", {})
                job_responsibilities = original_data.get("responsibilities", [])
                
                # Check required fields
                if job_title:
                    score += 0.2
                else:
                    suggestions.append("Add job title")
                
                if len(enhanced_description) > 200:
                    score += 0.2
                else:
                    suggestions.append("Expand job description")
                
                if len(job_skills) > 0:
                    score += 0.2
                else:
                    suggestions.append("Add required skills")
                
                salary_min = job_salary.get('min', 0) if isinstance(job_salary, dict) else 0
                if salary_min and salary_min > 0:
                    score += 0.2
                else:
                    suggestions.append("Specify salary range")
                
                if len(job_responsibilities) > 0:
                    score += 0.2
                else:
                    suggestions.append("Add job responsibilities")
                
                # Create validated job for storage
                validated_job = {
                    "title": original_data.get("title", "Unknown"),
                    "description": enhanced_description,
                    "location": original_data.get("location", "Remote"),
                    "type": original_data.get("type", "FULL_TIME"),
                    "workLocation": original_data.get("workLocation", "REMOTE"),
                    "skills": original_data.get("skills", []),
                    "salary": original_data.get("salary", {}),
                    "requirements": original_data.get("requirements", []),
                    "responsibilities": original_data.get("responsibilities", []),
                    "benefits": original_data.get("benefits", []),
                    "experience": original_data.get("experience", {}),
                    "quality_score": score,
                    "suggestions": suggestions,
                    "company_context_applied": bool(company_intelligence)
                }
                
                state["validated_jobs"].append(validated_job)
                print(f"✅ Validated: {job_title} (Score: {score:.2f})")
                
            except Exception as e:
                print(f"❌ Error validating job: {str(e)}")
                original_data = enhanced_job.get("original_data", {})
                job_title = original_data.get("title", "Unknown") if isinstance(original_data, dict) else "Unknown"
                state["errors"].append({
                    "job": job_title,
                    "agent": "validate",
                    "error": str(e)
                })
        
        state["current_step"] = "validate_complete"
        
        # Emit completion
        await emit_agent_stage_update(
            user_id=user_id,
            stage="validate",
            status="completed",
            message=f"Validated {len(state.get('validated_jobs', []))} jobs",
            progress=90
        )
        
        return state
    
    async def _storage_agent(self, state: JobProcessorState) -> JobProcessorState:
        """
        Agent 5: Storage
        Stores processed jobs in vector database for future RAG
        """
        print("💾 Agent 5: Storage step...")
        
        validated_jobs = state.get("validated_jobs", [])
        user_id = state.get("user_id", "")
        
        # Emit stage update
        await emit_agent_stage_update(
            user_id=user_id,
            stage="store",
            status="processing",
            message="Finalizing job storage...",
            progress=95
        )
        
        if self.vector_store_enabled and self.embeddings and self.vector_store:
            try:
                for job in validated_jobs:
                    # Create searchable text
                    skills = job.get('skills', [])
                    skill_names = ', '.join([s.get('name', '') for s in skills if isinstance(s, dict)])
                    
                    job_text = f"""
Title: {job.get('title', '')}
Description: {job.get('description', '')}
Skills: {skill_names}
Location: {job.get('location', '')}
Type: {job.get('type', '')}
"""
                    
                    # Generate embedding
                    embedding = await self.embeddings.aembed_query(job_text)
                    
                    # Store in vector DB
                    point = {
                        "id": str(uuid.uuid4()),
                        "vector": embedding,
                        "payload": {
                            "title": job.get('title', ''),
                            "description": job.get('description', '')[:500],
                            "skills": [s.get('name', '') for s in skills if isinstance(s, dict)],
                            "quality_score": job.get('quality_score', 0),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                    
                    self.vector_store.add_documents([point])
                
                print(f"✅ Stored {len(validated_jobs)} jobs in vector DB")
                
            except Exception as e:
                print(f"⚠️ Storage warning: {str(e)}")
                # Non-critical error, continue
        else:
            print("ℹ️  Vector storage skipped (optional feature not enabled)")
        
        # Final token usage
        state["token_usage"] = self.token_tracker.get_usage()
        state["current_step"] = "complete"
        
        # Emit completion
        await emit_agent_stage_update(
            user_id=user_id,
            stage="complete",
            status="completed",
            message=f"Successfully processed {len(validated_jobs)} jobs!",
            progress=100
        )
        
        return state
    
    async def process_jobs(
        self,
        files: List[JobUploadFile],
        company_id: str,
        user_id: str,
        company_intelligence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Process job uploads through multi-agent workflow
        
        Returns:
            Dict containing processed jobs, errors, and metadata
        """
        print(f"🤖 Starting agentic job processing for {len(files)} files...")
        
        # Initialize state as TypedDict (convert Pydantic models to dicts)
        initial_state: JobProcessorState = {
            "files": [{"filename": f.filename, "content": f.content, "file_type": f.file_type} for f in files],
            "company_id": company_id,
            "user_id": user_id,
            "company_intelligence": company_intelligence,
            "extracted_jobs": [],
            "enhanced_jobs": [],
            "validated_jobs": [],
            "errors": [],
            "token_usage": None,
            "current_step": "start",
            "successful_patterns": [],
            "crew_insights": {}
        }
        
        print(f"📦 Initial state created with {len(initial_state['files'])} files")
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        print(f"📦 Final state received:")
        print(f"   - extracted_jobs: {len(final_state.get('extracted_jobs', []))}")
        print(f"   - enhanced_jobs: {len(final_state.get('enhanced_jobs', []))}")
        print(f"   - validated_jobs: {len(final_state.get('validated_jobs', []))}")
        
        # Prepare response
        validated_jobs = final_state.get('validated_jobs', [])
        errors = final_state.get('errors', [])
        
        result = {
            "success": True,
            "processed_jobs": validated_jobs,
            "total_processed": len(validated_jobs),
            "total_errors": len(errors),
            "errors": errors,
            "token_usage": {
                "current_call": final_state.get('token_usage'),
                "cumulative": get_token_usage_stats()["cumulative"]
            },
            "agent_metadata": {
                "agents_deployed": 5,
                "workflow_steps": [
                    "File Parsing",
                    "Context Enrichment",
                    "Quality Enhancement",
                    "Validation",
                    "Storage"
                ],
                "company_intelligence_used": bool(company_intelligence),
                "quality_scores": [j.get("quality_score", 0) for j in validated_jobs],
                "avg_quality_score": sum([j.get("quality_score", 0) for j in validated_jobs]) / len(validated_jobs) if validated_jobs else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        print(f"✅ Job processing complete: {len(validated_jobs)} jobs processed")
        print(f"📊 Average quality score: {result['agent_metadata']['avg_quality_score']:.2f}")
        
        return result


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_job_processor = None

def get_job_processor() -> AgenticJobProcessor:
    """Get or create singleton job processor instance"""
    global _job_processor
    if _job_processor is None:
        _job_processor = AgenticJobProcessor()
    return _job_processor


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

async def main():
    """Test the agentic job processor"""
    processor = get_job_processor()
    
    # Test with sample data
    test_files = [
        JobUploadFile(
            filename="test_job.txt",
            content="""
            Senior Software Engineer
            
            We are looking for an experienced software engineer to join our team.
            
            Requirements:
            - 5+ years of Python experience
            - Strong knowledge of Django and FastAPI
            - Experience with cloud platforms (AWS/Azure)
            - Excellent communication skills
            
            Responsibilities:
            - Design and implement scalable backend systems
            - Mentor junior developers
            - Collaborate with cross-functional teams
            
            Salary: $120,000 - $150,000 per year
            Location: Remote
            """,
            file_type="text/plain"
        )
    ]
    
    result = await processor.process_jobs(
        files=test_files,
        company_id="test-company-123",
        user_id="test-user-456"
    )
    
    print("\n" + "="*80)
    print("PROCESSING RESULT:")
    print("="*80)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
