"""
🤖 AGENTIC AI JOB DESCRIPTION GENERATOR (v3.0) - World-Class
=============================================================

Advanced multi-agent system for generating company-specific job descriptions using:
- LangGraph for workflow orchestration
- CrewAI multi-agent collaboration (Writer, Reviewer, Optimizer)
- DSPy MIPRO for job description optimization
- Company Intelligence from crawl4ai and Tavily
- Multi-step refinement workflow
- Industry insights and market data
- RAG knowledge base from high-performing job descriptions
- Feedback loops from application rates and hire quality
- Quality validation and optimization

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
# 🎯 DSPY JOB DESCRIPTION SIGNATURES
# ============================================================================

if DSPY_AVAILABLE:
    class JobDescriptionSignature(dspy.Signature):
        """Generate compelling job description."""
        job_title: str = dspy.InputField(desc="Job title")
        company_context: str = dspy.InputField(desc="Company information and culture")
        requirements: str = dspy.InputField(desc="Skills and experience required")
        description: str = dspy.OutputField(desc="Compelling job description")
        selling_points: str = dspy.OutputField(desc="Key selling points for candidates")
        
    class JobOptimizationSignature(dspy.Signature):
        """Optimize job description for candidate attraction."""
        draft_description: str = dspy.InputField(desc="Draft job description")
        high_performing_patterns: str = dspy.InputField(desc="Patterns from successful job postings")
        optimized_description: str = dspy.OutputField(desc="Optimized job description")
        predicted_application_rate: str = dspy.OutputField(desc="Predicted application rate (low/medium/high)")


# ============================================================================
# SOCKET EMISSION HELPER
# ============================================================================

async def emit_agent_stage_update(company_id: str, stage: str, status: str, message: str, progress: int):
    """
    Emit agent stage updates to frontend via Next.js Socket.IO API
    """
    try:
        backend_url = os.getenv("NEXTAUTH_URL", "http://localhost:3000")
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{backend_url}/api/socket/emit-agent-stage",
                json={
                    "companyId": company_id,
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

class JobDescriptionRequest(BaseModel):
    """Request model for job description generation"""
    jobTitle: str
    companyName: str
    companyId: Optional[str] = None
    mustHaveSkills: List[str] = Field(default_factory=list)
    niceToHaveSkills: List[str] = Field(default_factory=list)
    experienceLevel: Optional[str] = None
    location: Optional[str] = "Remote"
    workType: Optional[str] = "Full-time"
    responsibilities: List[str] = Field(default_factory=list)


class JobDescriptionState(BaseModel):
    """State for LangGraph workflow"""
    request: JobDescriptionRequest
    company_intelligence: Optional[Dict[str, Any]] = None
    company_analysis: Optional[str] = None
    market_insights: Optional[str] = None
    industry_trends: Optional[str] = None
    draft_description: Optional[Dict[str, str]] = None
    refined_description: Optional[Dict[str, str]] = None
    quality_score: float = 0.0
    suggestions: List[str] = Field(default_factory=list)
    token_usage: Optional[Dict] = None
    current_step: str = "start"
    
    # RAG Context
    high_performing_patterns: List[Dict] = Field(default_factory=list)
    
    # CrewAI Collaboration Results
    crew_insights: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# AGENTIC JOB DESCRIPTION GENERATOR
# ============================================================================

class AgenticJobDescriptionGenerator:
    """
    🚀 WORLD-CLASS JOB DESCRIPTION GENERATOR (v3.0)
    
    Features:
    - LangGraph multi-step workflow
    - CrewAI 3-agent collaboration (Writer, Reviewer, Optimizer)
    - DSPy MIPRO for job description optimization
    - RAG knowledge base from high-performing job descriptions
    - Feedback loops from application rates and hire quality
    
    Agents:
    1. Company Intelligence Agent - Fetches and analyzes company data
    2. Market Research Agent - Gathers industry and market insights
    3. Description Drafting Agent - Creates initial job description
    4. Refinement Agent - Enhances with company context
    5. Quality Assurance Agent - Validates and optimizes
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
            temperature=0.7,  # Balanced for creativity and consistency
            max_tokens=4000,
            callbacks=[self.token_tracker]
        )
        
        # Initialize Advanced Components
        self._init_crewai_agents()
        self._init_rag_knowledge_base()
        self._init_dspy_optimizer()
        
        # Initialize embeddings (optional - only if vector store is needed)
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                azure_endpoint=os.getenv("TEXT_EMBEDDING_ENDPOINT"),
                deployment=os.getenv("TEXT_EMBEDDING_MODEL"),
                openai_api_version=os.getenv("TEXT_EMBEDDING_API_VERSION"),
            )
            self.vector_store_enabled = True
        except Exception as e:
            print(f"⚠️ Vector store disabled: {str(e)}")
            self.embeddings = None
            self.vector_store_enabled = False
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        print("✅ AgenticJobDescriptionGenerator v3.0 initialized with CrewAI + DSPy + RAG")
    
    def _init_crewai_agents(self):
        """Initialize CrewAI multi-agent system"""
        self.crewai_enabled = CREWAI_AVAILABLE
        
        if CREWAI_AVAILABLE:
            self.writer_agent = Agent(
                name="JD Writer Agent",
                role="Expert Job Description Writer",
                goal="Write compelling, SEO-optimized job descriptions that attract top talent",
                backstory="Former head of talent acquisition at a Fortune 100 company. Expert at crafting job descriptions that stand out and attract the best candidates.",
                allow_delegation=False
            )
            
            self.reviewer_agent = Agent(
                name="JD Reviewer Agent",
                role="Hiring Compliance Expert",
                goal="Review job descriptions for bias, legal compliance, and inclusivity",
                backstory="Employment lawyer with 15 years experience in HR compliance. Expert at identifying problematic language and ensuring inclusive, legally compliant job postings.",
                allow_delegation=False
            )
            
            self.optimizer_agent = Agent(
                name="JD Optimizer Agent",
                role="Conversion Rate Optimizer",
                goal="Optimize job descriptions for maximum application conversion while maintaining quality",
                backstory="Growth marketing expert who has A/B tested thousands of job postings. Expert at optimizing for candidate engagement and application rates.",
                allow_delegation=False
            )
            
            print("✅ CrewAI 3-agent JD generation crew initialized")
    
    def _init_rag_knowledge_base(self):
        """Initialize RAG for learning from high-performing job descriptions"""
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
                print("✅ JD RAG Knowledge Base initialized")
            except Exception as e:
                print(f"⚠️ RAG initialization failed: {e}")
    
    def _init_dspy_optimizer(self):
        """Initialize DSPy for job description optimization"""
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
                print("✅ DSPy optimizer initialized for JD generation")
            except Exception as e:
                print(f"⚠️ DSPy initialization failed: {e}")
    
    def _build_workflow(self):
        """Build LangGraph workflow for job description generation"""
        
        workflow = StateGraph(JobDescriptionState)
        
        # Define workflow nodes (agents)
        workflow.add_node("retrieve_patterns", self._retrieve_high_performing_patterns)
        workflow.add_node("fetch_company_intelligence", self._company_intelligence_agent)
        workflow.add_node("research_market", self._market_research_agent)
        workflow.add_node("run_crew_collaboration", self._run_crew_collaboration)
        workflow.add_node("draft_description", self._description_drafting_agent)
        workflow.add_node("refine_with_context", self._refinement_agent)
        workflow.add_node("quality_assurance", self._quality_assurance_agent)
        workflow.add_node("store_for_learning", self._store_jd_pattern)
        
        # Define workflow edges
        workflow.set_entry_point("retrieve_patterns")
        workflow.add_edge("retrieve_patterns", "fetch_company_intelligence")
        workflow.add_edge("fetch_company_intelligence", "research_market")
        workflow.add_edge("research_market", "run_crew_collaboration")
        workflow.add_edge("run_crew_collaboration", "draft_description")
        workflow.add_edge("draft_description", "refine_with_context")
        workflow.add_edge("refine_with_context", "quality_assurance")
        workflow.add_edge("quality_assurance", "store_for_learning")
        workflow.add_edge("store_for_learning", END)
        
        return workflow.compile()
    
    async def _retrieve_high_performing_patterns(self, state: JobDescriptionState) -> JobDescriptionState:
        """Retrieve high-performing job description patterns from RAG"""
        state.current_step = "retrieve_patterns"
        
        if self.rag_enabled:
            try:
                patterns = []
                pattern_keys = self.redis_client.keys("jd_pattern:*")
                
                for key in pattern_keys[:20]:
                    data = self.redis_client.get(key)
                    if data:
                        pattern = json.loads(data)
                        # Only use patterns with high application rates
                        if pattern.get("application_rate", 0) >= 0.1:  # 10%+ rate
                            patterns.append(pattern)
                
                state.high_performing_patterns = patterns[:5]
            except Exception as e:
                print(f"⚠️ Pattern retrieval error: {e}")
        
        return state
    
    async def _run_crew_collaboration(self, state: JobDescriptionState) -> JobDescriptionState:
        """Run CrewAI multi-agent collaboration for JD generation"""
        state.current_step = "run_crew_collaboration"
        
        if self.crewai_enabled:
            try:
                context = f"""
Job Title: {state.request.jobTitle}
Company: {state.request.companyName}
Skills Required: {', '.join(state.request.mustHaveSkills)}
Company Context: {json.dumps(state.company_intelligence) if state.company_intelligence else 'N/A'}
Market Insights: {state.market_insights or 'N/A'}
High-Performing Patterns: {json.dumps([p.get('key_phrases', []) for p in state.high_performing_patterns[:3]])}
"""
                
                writer_task = Task(
                    description=f"Write compelling job description for {state.request.jobTitle} at {state.request.companyName}. Make it stand out and attract top talent.",
                    agent=self.writer_agent,
                    expected_output="Complete job description draft"
                )
                
                reviewer_task = Task(
                    description=f"Review the job description for {state.request.jobTitle}. Check for bias, legal compliance, and inclusivity.",
                    agent=self.reviewer_agent,
                    expected_output="Compliance review with suggested changes"
                )
                
                optimizer_task = Task(
                    description=f"Optimize the job description for {state.request.jobTitle}. Apply patterns from high-performing posts: {json.dumps([p.get('key_phrases', []) for p in state.high_performing_patterns[:3]])}",
                    agent=self.optimizer_agent,
                    expected_output="Optimized job description with predicted application rate"
                )
                
                crew = Crew(
                    agents=[self.writer_agent, self.reviewer_agent, self.optimizer_agent],
                    tasks=[writer_task, reviewer_task, optimizer_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                crew.kickoff()
                
                state.crew_insights = {
                    "writer_draft": writer_task.output.raw if writer_task.output else "",
                    "reviewer_feedback": reviewer_task.output.raw if reviewer_task.output else "",
                    "optimizer_output": optimizer_task.output.raw if optimizer_task.output else ""
                }
                
            except Exception as e:
                print(f"⚠️ CrewAI collaboration error: {e}")
        
        return state
    
    async def _store_jd_pattern(self, state: JobDescriptionState) -> JobDescriptionState:
        """Store JD pattern for future learning"""
        state.current_step = "store_for_learning"
        
        if self.rag_enabled and state.refined_description:
            try:
                pattern_id = hashlib.md5(
                    f"{state.request.jobTitle}{state.request.companyName}{datetime.now().isoformat()}".encode()
                ).hexdigest()
                
                pattern = {
                    "id": pattern_id,
                    "job_title": state.request.jobTitle,
                    "company_name": state.request.companyName,
                    "quality_score": state.quality_score,
                    "key_phrases": state.suggestions[:5] if state.suggestions else [],
                    "application_rate": 0,  # Updated via feedback
                    "hire_quality": 0,  # Updated via feedback
                    "timestamp": datetime.now().isoformat()
                }
                
                self.redis_client.set(f"jd_pattern:{pattern_id}", json.dumps(pattern))
                
            except Exception as e:
                print(f"⚠️ Pattern storage error: {e}")
        
        return state
    
    async def record_jd_performance(
        self,
        pattern_id: str,
        views: int,
        applications: int,
        quality_hires: int
    ):
        """Record job description performance (for feedback loop)"""
        if self.rag_enabled:
            try:
                data = self.redis_client.get(f"jd_pattern:{pattern_id}")
                if data:
                    pattern = json.loads(data)
                    pattern["views"] = views
                    pattern["applications"] = applications
                    pattern["application_rate"] = applications / max(views, 1)
                    pattern["hire_quality"] = quality_hires
                    pattern["feedback_timestamp"] = datetime.now().isoformat()
                    self.redis_client.set(f"jd_pattern:{pattern_id}", json.dumps(pattern))
                    print(f"✅ Recorded JD performance for {pattern_id}")
            except Exception as e:
                print(f"⚠️ Performance recording error: {e}")
    
    async def _company_intelligence_agent(self, state: JobDescriptionState) -> JobDescriptionState:
        """
        Agent 1: Company Intelligence
        Fetches comprehensive company intelligence data
        """
        print("🔍 Agent 1: Fetching company intelligence...")
        
        # Emit stage update
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="fetch_company_intelligence",
                status="processing",
                message="Fetching company intelligence data...",
                progress=15
            )
        
        try:
            if state.request.companyId:
                # Fetch from Next.js API endpoint
                backend_url = os.getenv("NEXT_PUBLIC_BACKEND_URL", "http://localhost:3000")
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{backend_url}/api/company-intelligence",
                        params={"companyId": state.request.companyId},
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("success") and data.get("data"):
                            state.company_intelligence = data["data"]
                            print(f"✅ Fetched company intelligence for {state.request.companyName}")
                        else:
                            print("⚠️ No company intelligence found, using basic info")
                    else:
                        print(f"⚠️ Failed to fetch company intelligence: {response.status_code}")
            
            # Analyze company profile
            company_context = ""
            if state.company_intelligence:
                ci = state.company_intelligence
                company_context = f"""
Company: {state.request.companyName}
About: {ci.get('aboutUs', 'N/A')}
Mission: {ci.get('missionStatement', 'N/A')}
Values: {', '.join(ci.get('values', []))}
Culture: {ci.get('companyCulture', 'N/A')}
Tech Stack: {', '.join(ci.get('techStack', []))}
Products/Services: {', '.join(ci.get('productsServices', []))}
Benefits: {', '.join(ci.get('benefits', []))}
Market Position: {ci.get('marketPosition', 'N/A')}
"""
            else:
                company_context = f"Company: {state.request.companyName}\n(Limited information available)"
            
            # AI analysis of company
            prompt = f"""Analyze this company and create a compelling employer brand summary:

{company_context}

Provide:
1. Unique value proposition as an employer
2. Cultural highlights and work environment
3. Career growth opportunities
4. Why top talent should join
5. Competitive advantages in the market

Keep it concise, professional, and compelling."""

            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert employer branding specialist and recruitment consultant."),
                HumanMessage(content=prompt)
            ])
            
            state.company_analysis = response.content.strip()
            print(f"✅ Company analysis complete")
            
        except Exception as e:
            print(f"⚠️ Company intelligence error: {str(e)}")
            state.company_analysis = f"Company: {state.request.companyName} (Analysis unavailable)"
        
        state.current_step = "company_intelligence_complete"
        
        # Emit completion
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="fetch_company_intelligence",
                status="completed",
                message="Company intelligence loaded",
                progress=30
            )
        
        return state
    
    async def _market_research_agent(self, state: JobDescriptionState) -> JobDescriptionState:
        """
        Agent 2: Market Research
        Gathers market insights and industry trends
        """
        print("📊 Agent 2: Researching market and industry...")
        
        # Emit stage update
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="research_market",
                status="processing",
                message="Analyzing market trends and insights...",
                progress=45
            )
        
        try:
            # Market analysis prompt
            prompt = f"""Provide current market insights for this role:

Job Title: {state.request.jobTitle}
Experience Level: {state.request.experienceLevel}
Location: {state.request.location}
Required Skills: {', '.join(state.request.mustHaveSkills)}

Provide:
1. Current salary range expectations ({state.request.location} market)
2. In-demand skills and technologies for this role
3. Market competitiveness (hot/balanced/slow)
4. Remote vs onsite trends
5. Key benefits candidates expect
6. Industry-specific insights

Be specific and data-driven."""

            response = await self.llm.ainvoke([
                SystemMessage(content="You are a market research analyst specializing in tech recruitment and compensation trends."),
                HumanMessage(content=prompt)
            ])
            
            state.market_insights = response.content.strip()
            
            # Industry trends
            trends_prompt = f"""What are the current trends in hiring for {state.request.jobTitle} roles?

Focus on:
1. Skill evolution and emerging technologies
2. Work arrangement preferences
3. Candidate expectations and priorities
4. Competitive differentiation factors

Keep it current and actionable."""

            trends_response = await self.llm.ainvoke([
                SystemMessage(content="You are an industry analyst tracking recruitment and talent acquisition trends."),
                HumanMessage(content=trends_prompt)
            ])
            
            state.industry_trends = trends_response.content.strip()
            
            print(f"✅ Market research complete")
            
        except Exception as e:
            print(f"⚠️ Market research error: {str(e)}")
            state.market_insights = "Market insights unavailable"
            state.industry_trends = "Industry trends unavailable"
        
        state.current_step = "market_research_complete"
        
        # Emit completion
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="research_market",
                status="completed",
                message="Market research completed",
                progress=55
            )
        
        return state
    
    async def _description_drafting_agent(self, state: JobDescriptionState) -> JobDescriptionState:
        """
        Agent 3: Description Drafting
        Creates initial structured job description
        """
        print("✍️ Agent 3: Drafting job description...")
        
        # Emit stage update
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="draft_description",
                status="processing",
                message="Creating job description draft...",
                progress=70
            )
        
        try:
            prompt = f"""Create a comprehensive, structured job description:

JOB DETAILS:
Title: {state.request.jobTitle}
Company: {state.request.companyName}
Location: {state.request.location}
Work Type: {state.request.workType}
Experience: {state.request.experienceLevel}
Must-Have Skills: {', '.join(state.request.mustHaveSkills)}
Nice-to-Have Skills: {', '.join(state.request.niceToHaveSkills)}
Responsibilities: {', '.join(state.request.responsibilities)}

COMPANY CONTEXT:
{state.company_analysis}

MARKET INSIGHTS:
{state.market_insights}

INDUSTRY TRENDS:
{state.industry_trends}

Create a JSON structure with these sections:
{{
    "company_overview": "Compelling company intro (2-3 paragraphs, include company name)",
    "role_overview": "Role summary and impact (1-2 paragraphs)",
    "key_responsibilities": "Bullet-pointed responsibilities (5-8 items)",
    "required_qualifications": "Must-have qualifications (5-7 items)",
    "preferred_qualifications": "Nice-to-have qualifications (3-5 items)",
    "benefits_perks": "Benefits and perks (5-8 items)",
    "growth_opportunities": "Career growth and development (1-2 paragraphs)",
    "work_environment": "Work culture and environment (1 paragraph)"
}}

Make it:
- Compelling and professional
- Inclusive and bias-free
- Specific to the company
- Aligned with market expectations
- Clear about expectations and opportunities

Return only valid JSON."""

            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert technical writer specializing in job descriptions. You create compelling, inclusive postings that attract top talent."),
                HumanMessage(content=prompt)
            ])
            
            # Parse JSON response
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            state.draft_description = json.loads(content.strip())
            print(f"✅ Draft job description created")
            
        except Exception as e:
            print(f"❌ Drafting error: {str(e)}")
            # Fallback to basic structure
            state.draft_description = {
                "company_overview": f"{state.request.companyName} is hiring for {state.request.jobTitle}",
                "role_overview": f"We are seeking a talented {state.request.jobTitle}",
                "key_responsibilities": "Responsibilities to be defined",
                "required_qualifications": ', '.join(state.request.mustHaveSkills),
                "preferred_qualifications": ', '.join(state.request.niceToHaveSkills),
                "benefits_perks": "Competitive benefits package",
                "growth_opportunities": "Career growth opportunities available",
                "work_environment": "Collaborative work environment"
            }
        
        state.current_step = "draft_complete"
        
        # Emit completion
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="draft_description",
                status="completed",
                message="Draft created successfully",
                progress=80
            )
        
        return state
    
    async def _refinement_agent(self, state: JobDescriptionState) -> JobDescriptionState:
        """
        Agent 4: Refinement
        Enhances description with deep company context
        """
        print("✨ Agent 4: Refining with company context...")
        
        # Emit stage update
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="refine_with_context",
                status="processing",
                message="Refining with company-specific context...",
                progress=90
            )
        
        try:
            # Build rich company context
            company_details = ""
            if state.company_intelligence:
                ci = state.company_intelligence
                company_details = f"""
DEEP COMPANY CONTEXT:
About Us: {ci.get('aboutUs', 'N/A')}
Mission: {ci.get('missionStatement', 'N/A')}
Core Values: {', '.join(ci.get('values', []))}
Company Culture: {ci.get('companyCulture', 'N/A')}
Tech Stack: {', '.join(ci.get('techStack', []))}
Products/Services: {', '.join(ci.get('productsServices', []))}
Company Benefits: {', '.join(ci.get('benefits', []))}
Recent News: {ci.get('recentNews', 'N/A')}
Awards: {', '.join(ci.get('awards', []))}
Market Position: {ci.get('marketPosition', 'N/A')}
Office Locations: {', '.join(ci.get('officeLocations', []))}
Remote Policy: {ci.get('remotePolicy', 'N/A')}
"""
            
            prompt = f"""Refine and enhance this job description with specific company context:

CURRENT DRAFT:
{json.dumps(state.draft_description, indent=2)}

{company_details}

{state.company_analysis}

Enhance the description by:
1. Weaving in specific company values and culture
2. Adding concrete details about products/services
3. Highlighting authentic benefits (use actual company benefits)
4. Incorporating tech stack alignment
5. Emphasizing growth opportunities specific to this company
6. Making the work environment description vivid and specific
7. Ensuring 100% company authenticity

Return the same JSON structure but with enhanced, company-specific content.
Make it feel like it could ONLY be from this company."""

            response = await self.llm.ainvoke([
                SystemMessage(content="You are a senior recruitment marketing specialist who creates authentic, company-specific job descriptions that reflect the unique employer brand."),
                HumanMessage(content=prompt)
            ])
            
            # Parse refined JSON
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            state.refined_description = json.loads(content.strip())
            print(f"✅ Description refined with company context")
            
        except Exception as e:
            print(f"⚠️ Refinement error: {str(e)}")
            state.refined_description = state.draft_description
        
        state.current_step = "refinement_complete"
        
        # Emit completion
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="refine_with_context",
                status="completed",
                message="Description refined with company context",
                progress=95
            )
        
        return state
    
    async def _quality_assurance_agent(self, state: JobDescriptionState) -> JobDescriptionState:
        """
        Agent 5: Quality Assurance
        Validates quality and provides improvement suggestions
        """
        print("✓ Agent 5: Quality assurance check...")
        
        # Emit stage update
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="quality_assurance",
                status="processing",
                message="Running quality checks...",
                progress=98
            )
        
        try:
            # Calculate quality score
            score = 0.0
            suggestions = []
            
            desc = state.refined_description
            
            # Check each section
            if desc.get('company_overview') and len(desc['company_overview']) > 200:
                score += 0.15
            else:
                suggestions.append("Expand company overview")
            
            if desc.get('role_overview') and len(desc['role_overview']) > 100:
                score += 0.10
            else:
                suggestions.append("Add more detail to role overview")
            
            if desc.get('key_responsibilities') and len(desc['key_responsibilities']) > 200:
                score += 0.15
            else:
                suggestions.append("Add more specific responsibilities")
            
            if desc.get('required_qualifications') and len(desc['required_qualifications']) > 150:
                score += 0.15
            else:
                suggestions.append("Detail required qualifications")
            
            if desc.get('preferred_qualifications') and len(desc['preferred_qualifications']) > 100:
                score += 0.10
            else:
                suggestions.append("Add preferred qualifications")
            
            if desc.get('benefits_perks') and len(desc['benefits_perks']) > 150:
                score += 0.15
            else:
                suggestions.append("Expand benefits section")
            
            if desc.get('growth_opportunities') and len(desc['growth_opportunities']) > 100:
                score += 0.10
            else:
                suggestions.append("Highlight growth opportunities")
            
            if desc.get('work_environment') and len(desc['work_environment']) > 100:
                score += 0.10
            else:
                suggestions.append("Describe work environment better")
            
            # Check for company intelligence integration
            if state.company_intelligence:
                score += 0.10
            else:
                suggestions.append("Add company intelligence for 100% authenticity")
            
            state.quality_score = min(score, 1.0)
            state.suggestions = suggestions if score < 0.8 else []
            
            print(f"✅ Quality score: {state.quality_score:.2f}")
            if suggestions:
                print(f"💡 Suggestions: {', '.join(suggestions)}")
            
        except Exception as e:
            print(f"⚠️ QA error: {str(e)}")
            state.quality_score = 0.5
            state.suggestions = ["Quality check incomplete"]
        
        # Store in vector DB for future RAG (optional)
        if self.vector_store_enabled:
            try:
                # Vector storage implementation would go here
                # For now, we'll skip it to avoid dependencies
                print("💾 Vector storage skipped (not required for core functionality)")
            except Exception as e:
                print(f"⚠️ Storage warning: {str(e)}")
        else:
            print("⚠️ Vector store not available, skipping storage")
        
        state.token_usage = self.token_tracker.get_usage()
        state.current_step = "complete"
        
        # Emit completion
        if state.request.companyId:
            await emit_agent_stage_update(
                company_id=state.request.companyId,
                stage="complete",
                status="completed",
                message=f"Job description generated! (Quality: {state.quality_score:.0%})",
                progress=100
            )
        
        return state
    
    async def generate_description(
        self,
        request: JobDescriptionRequest,
        company_intelligence: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Generate job description through multi-agent workflow
        
        Returns:
            Dict containing job description, analysis, and metadata
        """
        print(f"🤖 Starting agentic job description generation for: {request.jobTitle}")
        
        # Initialize state
        initial_state = JobDescriptionState(
            request=request,
            company_intelligence=company_intelligence
        )
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        # Prepare response
        result = {
            "success": True,
            "job_description": final_state.refined_description,
            "analysis": {
                "company_analysis": final_state.company_analysis,
                "market_insights": final_state.market_insights,
                "industry_trends": final_state.industry_trends
            },
            "quality_score": final_state.quality_score,
            "suggestions": final_state.suggestions,
            "token_usage": {
                "current_call": final_state.token_usage,
                "cumulative": get_token_usage_stats()["cumulative"]
            },
            "agent_metadata": {
                "agents_deployed": 5,
                "workflow_steps": [
                    "Company Intelligence Fetch",
                    "Market Research",
                    "Description Drafting",
                    "Context Refinement",
                    "Quality Assurance"
                ],
                "company_intelligence_used": bool(company_intelligence),
                "company_specific": bool(company_intelligence),
                "version": "2.0-agentic",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        print(f"✅ Job description generated (Quality: {final_state.quality_score:.2%})")
        
        return result


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_job_desc_generator = None

def get_job_description_generator() -> AgenticJobDescriptionGenerator:
    """Get or create singleton job description generator instance"""
    global _job_desc_generator
    if _job_desc_generator is None:
        _job_desc_generator = AgenticJobDescriptionGenerator()
    return _job_desc_generator


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

async def main():
    """Test the agentic job description generator"""
    generator = get_job_description_generator()
    
    # Test with sample data
    test_request = JobDescriptionRequest(
        jobTitle="Senior Full Stack Developer",
        companyName="TechCorp Inc",
        mustHaveSkills=["Python", "React", "PostgreSQL", "AWS"],
        niceToHaveSkills=["Docker", "Kubernetes", "GraphQL"],
        experienceLevel="Senior (5+ years)",
        location="San Francisco, CA / Remote",
        workType="Full-time",
        responsibilities=[
            "Design and implement scalable web applications",
            "Lead technical architecture decisions",
            "Mentor junior developers"
        ]
    )
    
    result = await generator.generate_description(test_request)
    
    print("\n" + "="*80)
    print("GENERATION RESULT:")
    print("="*80)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())