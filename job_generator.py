# job_generator.py - AGENTIC AI VERSION with LangGraph workflows
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langgraph.graph import StateGraph, END
import dspy
from dotenv import load_dotenv
from datetime import datetime
import json
import uuid
from utils.token_usage import get_token_tracker, get_token_usage_stats
from agentic_ai.config import AgenticAIConfig
from agentic_ai.dspy_integration import DSPySignature, DSPyModule, ChainOfThought
from agentic_ai.langrag_integration import VectorStoreManager, RAGChain

# Load environment variables
load_dotenv()

# Define the request model
class JobDescriptionParams(BaseModel):
    jobTitle: str
    companyName: Optional[str] = "Unnamed Company"
    mustHaveSkills: List[str] = Field(default_factory=list)
    niceToHaveSkills: List[str] = Field(default_factory=list)
    experienceLevel: Optional[str] = None
    location: Optional[str] = None
    workType: Optional[str] = None
    companyDescription: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)

# Define the structured output model
class JobDescriptionSections(BaseModel):
    companyName: str = Field(description="company name")
    company_overview: str = Field(description="Overview of the company")
    job_overview: str = Field(description="Overview of the job position")
    responsibilities: str = Field(description="Key responsibilities for the role")
    requirements: str = Field(description="Required skills and qualifications")
    nice_to_have: str = Field(description="Nice-to-have skills and qualifications")
    benefits: str = Field(description="Benefits and perks of the job")

class JobDescriptionOutput(BaseModel):
    description: str
    token_usage: Dict[str, Any]

# ============================================================================
# AGENTIC AI - JOB GENERATOR AGENT with LangGraph Workflow
# ============================================================================

class JobGeneratorState(BaseModel):
    """State for LangGraph workflow"""
    params: JobDescriptionParams
    company_analysis: Optional[str] = None
    job_analysis: Optional[str] = None
    industry_insights: Optional[str] = None
    market_data: Optional[str] = None
    structured_jd: Optional[Dict[str, str]] = None
    token_usage: Optional[Dict] = None
    current_step: str = "start"
    error: Optional[str] = None


class JobGeneratorAgent:
    """
    Production-grade Job Generator Agent using:
    - LangGraph for multi-step workflow orchestration
    - DSPy for structured outputs
    - RAG for industry insights
    - Qdrant for storing past job descriptions
    - Chain-of-Thought for quality improvement
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
            temperature=0.7,
            max_tokens=3000,
            callbacks=[self.token_tracker]
        )
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("TEXT_EMBEDDING_ENDPOINT"),
            deployment=os.getenv("TEXT_EMBEDDING_MODEL"),
            openai_api_version=os.getenv("TEXT_EMBEDDING_API_VERSION"),
        )
        
        # Initialize Qdrant for job descriptions
        self.vector_store = VectorStoreManager(
            store_type="qdrant",
            collection_name="job_descriptions",
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        
        # Build LangGraph workflow
        self.workflow = self._build_langgraph_workflow()
        
        print("✅ Job Generator Agent initialized with LangGraph workflow")
    
    def _build_langgraph_workflow(self):
        """Build multi-step LangGraph workflow for job generation"""
        
        workflow = StateGraph(JobGeneratorState)
        
        # Define workflow nodes (steps)
        workflow.add_node("analyze_company", self._analyze_company)
        workflow.add_node("analyze_job_requirements", self._analyze_job_requirements)
        workflow.add_node("gather_industry_insights", self._gather_industry_insights)
        workflow.add_node("gather_market_data", self._gather_market_data)
        workflow.add_node("generate_structured_jd", self._generate_structured_jd)
        workflow.add_node("refine_and_optimize", self._refine_and_optimize)
        workflow.add_node("store_in_vectordb", self._store_in_vectordb)
        
        # Define workflow edges (flow)
        workflow.set_entry_point("analyze_company")
        workflow.add_edge("analyze_company", "analyze_job_requirements")
        workflow.add_edge("analyze_job_requirements", "gather_industry_insights")
        workflow.add_edge("gather_industry_insights", "gather_market_data")
        workflow.add_edge("gather_market_data", "generate_structured_jd")
        workflow.add_edge("generate_structured_jd", "refine_and_optimize")
        workflow.add_edge("refine_and_optimize", "store_in_vectordb")
        workflow.add_edge("store_in_vectordb", END)
        
        return workflow.compile()
    
    async def _analyze_company(self, state: JobGeneratorState) -> JobGeneratorState:
        """Step 1: Analyze company profile"""
        print("🔍 Step 1: Analyzing company profile...")
        
        prompt = f"""Analyze this company and create a compelling profile:

Company Name: {state.params.companyName}
Description: {state.params.companyDescription or 'Not provided'}

Provide:
1. Company positioning and unique value proposition
2. Culture and work environment indicators
3. Why candidates would want to work here
4. Industry standing and reputation

Keep it professional and engaging."""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a company branding expert."),
            HumanMessage(content=prompt)
        ])
        
        state.company_analysis = response.content
        state.current_step = "analyze_company_complete"
        return state
    
    async def _analyze_job_requirements(self, state: JobGeneratorState) -> JobGeneratorState:
        """Step 2: Deep analysis of job requirements"""
        print("🔍 Step 2: Analyzing job requirements...")
        
        prompt = f"""Analyze these job requirements for a {state.params.jobTitle}:

Must-have skills: {', '.join(state.params.mustHaveSkills)}
Nice-to-have skills: {', '.join(state.params.niceToHaveSkills)}
Experience level: {state.params.experienceLevel}
Location: {state.params.location}
Work type: {state.params.workType}

Provide:
1. Skill prioritization and categorization
2. Experience level appropriateness
3. Market competitiveness assessment
4. Potential skill gaps or additions"""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a technical recruitment specialist."),
            HumanMessage(content=prompt)
        ])
        
        state.job_analysis = response.content
        state.current_step = "analyze_job_complete"
        return state
    
    async def _gather_industry_insights(self, state: JobGeneratorState) -> JobGeneratorState:
        """Step 3: Gather industry-specific insights using RAG"""
        print("🔍 Step 3: Gathering industry insights...")
        
        try:
            # Search similar past job descriptions
            query = f"{state.params.jobTitle} {' '.join(state.params.mustHaveSkills)}"
            query_embedding = await self.embeddings.aembed_query(query)
            
            similar_jds = self.vector_store.similarity_search(query_embedding, k=3)
            
            if similar_jds:
                insights = "Insights from similar job postings:\n"
                for jd in similar_jds:
                    insights += f"- {jd['payload'].get('summary', 'N/A')}\n"
                state.industry_insights = insights
            else:
                state.industry_insights = "No similar job descriptions found in database."
                
        except Exception as e:
            print(f"⚠️ RAG insights unavailable: {str(e)}")
            state.industry_insights = "Industry insights not available."
        
        state.current_step = "gather_insights_complete"
        return state
    
    async def _gather_market_data(self, state: JobGeneratorState) -> JobGeneratorState:
        """Step 4: Gather market data and trends"""
        print("🔍 Step 4: Gathering market data...")
        
        prompt = f"""Provide current market insights for {state.params.jobTitle}:

1. Salary range expectations for {state.params.experienceLevel} level
2. In-demand skills and technologies
3. Remote vs onsite trends
4. Competitive benefits and perks
5. Hiring market status (competitive/balanced/candidate-favored)

Base this on {state.params.location or 'global'} market."""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a market research analyst for tech hiring."),
            HumanMessage(content=prompt)
        ])
        
        state.market_data = response.content
        state.current_step = "gather_market_complete"
        return state
    
    async def _generate_structured_jd(self, state: JobGeneratorState) -> JobGeneratorState:
        """Step 5: Generate structured job description"""
        print("🔍 Step 5: Generating structured JD...")
        
        prompt = f"""Create a comprehensive job description using this analysis:

COMPANY PROFILE:
{state.company_analysis}

JOB ANALYSIS:
{state.job_analysis}

INDUSTRY INSIGHTS:
{state.industry_insights}

MARKET DATA:
{state.market_data}

JOB DETAILS:
- Title: {state.params.jobTitle}
- Must-have: {', '.join(state.params.mustHaveSkills)}
- Nice-to-have: {', '.join(state.params.niceToHaveSkills)}
- Responsibilities: {', '.join(state.params.responsibilities)}
- Experience: {state.params.experienceLevel}
- Location: {state.params.location}
- Type: {state.params.workType}

Return JSON with these sections:
{{
  "company_overview": "About the company (include company name: {state.params.companyName})",
  "job_overview": "Role summary and impact",
  "responsibilities": "Key responsibilities (bullet points)",
  "requirements": "Must-have qualifications",
  "nice_to_have": "Preferred qualifications",
  "benefits": "Benefits and perks"
}}

Make it compelling, clear, and professional."""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are an expert job description writer."),
            HumanMessage(content=prompt)
        ])
        
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        
        try:
            state.structured_jd = json.loads(content.strip())
        except json.JSONDecodeError:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                state.structured_jd = json.loads(content[json_start:json_end])
        
        state.current_step = "generate_jd_complete"
        return state
    
    async def _refine_and_optimize(self, state: JobGeneratorState) -> JobGeneratorState:
        """Step 6: Refine and optimize the JD"""
        print("🔍 Step 6: Refining and optimizing...")
        
        # Additional optimization step - check for inclusive language, clarity, etc.
        prompt = f"""Review and improve this job description for:
1. Inclusive language
2. Clear expectations
3. Compelling value proposition
4. SEO optimization
5. Removal of bias

Current JD:
{json.dumps(state.structured_jd, indent=2)}

Return improved version in same JSON format."""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a diversity and inclusion expert for job postings."),
            HumanMessage(content=prompt)
        ])
        
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        
        try:
            state.structured_jd = json.loads(content.strip())
        except:
            pass  # Keep original if refinement fails
        
        state.current_step = "refine_complete"
        return state
    
    async def _store_in_vectordb(self, state: JobGeneratorState) -> JobGeneratorState:
        """Step 7: Store in vector database for future RAG"""
        print("🔍 Step 7: Storing in vector database...")
        
        try:
            # Create searchable text
            jd_text = f"""
Job Title: {state.params.jobTitle}
Company: {state.params.companyName}
{state.structured_jd.get('job_overview', '')}
{state.structured_jd.get('requirements', '')}
Skills: {', '.join(state.params.mustHaveSkills)}
"""
            
            embedding = await self.embeddings.aembed_query(jd_text)
            
            point = {
                "id": str(uuid.uuid4()),
                "vector": embedding,
                "payload": {
                    "job_title": state.params.jobTitle,
                    "company": state.params.companyName,
                    "summary": jd_text[:500],
                    "structured_jd": state.structured_jd,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            self.vector_store.add_documents([point])
            print("💾 Stored job description in Qdrant")
            
        except Exception as e:
            print(f"⚠️ Failed to store in vector DB: {str(e)}")
        
        state.token_usage = self.token_tracker.get_usage()
        state.current_step = "complete"
        return state
    
    async def generate_job_description(self, params: JobDescriptionParams) -> Dict[str, Any]:
        """Main method: Execute LangGraph workflow"""
        print(f"🤖 Starting agentic job description generation workflow...")
        
        # Initialize state
        initial_state = JobGeneratorState(
            params=params,
            current_step="start"
        )
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        result = {
            "job_description": final_state.structured_jd,
            "analysis": {
                "company_profile": final_state.company_analysis,
                "job_analysis": final_state.job_analysis,
                "industry_insights": final_state.industry_insights,
                "market_data": final_state.market_data
            },
            "token_usage": {
                "current_call": final_state.token_usage,
                "cumulative": get_token_usage_stats()["cumulative"]
            },
            "agent_metadata": {
                "agent": "JobGeneratorAgent",
                "version": "2.0-agentic-langgraph",
                "workflow_steps": 7,
                "rag_enabled": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        print(f"✅ Job description generation complete with {len(final_state.structured_jd)} sections")
        return result


# Create global agent instance
_job_agent = None

def get_job_agent() -> JobGeneratorAgent:
    """Get or create singleton Job Generator Agent"""
    global _job_agent
    if _job_agent is None:
        _job_agent = JobGeneratorAgent()
    return _job_agent

async def generate_job_description(params: JobDescriptionParams) -> Dict[str, Any]:
    """
    AGENTIC AI VERSION: Multi-step LangGraph workflow
    
    Uses:
    - LangGraph for orchestrated multi-step generation
    - RAG for industry insights
    - Chain-of-Thought for quality optimization
    - Qdrant for persistent storage
    
    Returns enhanced dict with agent metadata and workflow details.
    """
    agent = get_job_agent()
    return await agent.generate_job_description(params)


# ============================================================================
# BACKWARD COMPATIBILITY - Legacy functions
# ============================================================================

def get_llm(token_tracker=None):
    """Helper function for legacy code - returns basic LLM"""
    callbacks = [token_tracker] if token_tracker else []
    return AzureChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_API_VERSION"),
        temperature=0.7,
        max_tokens=2500,
        callbacks=callbacks
    )


def generate_job_description_sync(params: JobDescriptionParams) -> JobDescriptionOutput:
    """LEGACY SYNC VERSION (deprecated)"""
    import asyncio
    
    print("⚠️ Warning: Using legacy sync generation. Consider upgrading to async version.")
    
    token_tracker = get_token_tracker()
    
    template = """
Create a professional job description for a {jobTitle} position.
Company Name: {companyName}

Company Description: {companyDescription}

Must-have skills: {mustHaveSkills}
Nice-to-have skills: {niceToHaveSkills}
Experience Level: {experienceLevel}
Location: {location}
Work Type: {workType}

Key Responsibilities: {responsibilities}

Please structure the job description with the following sections:
1. About the Company
2. Job Overview
3. Responsibilities
4. Requirements
5. Nice-to-Have Skills
6. Benefits (generic)

Make it compelling and professional.
    """

    prompt = PromptTemplate(
        input_variables=[
            "jobTitle", "companyName", "companyDescription", "mustHaveSkills",
            "niceToHaveSkills", "experienceLevel", "location", "workType",
            "responsibilities"
        ],
        template=template
    )

    chat = get_llm(token_tracker)
    chain = LLMChain(llm=chat, prompt=prompt)

    response = chain.run({
        "jobTitle": params.jobTitle,
        "companyName": params.companyName,
        "companyDescription": params.companyDescription or "A leading technology company",
        "mustHaveSkills": ", ".join(params.mustHaveSkills) if params.mustHaveSkills else "Not specified",
        "niceToHaveSkills": ", ".join(params.niceToHaveSkills) if params.niceToHaveSkills else "Not specified",
        "experienceLevel": params.experienceLevel or "Not specified",
        "location": params.location or "Not specified",
        "workType": params.workType or "Not specified",
        "responsibilities": ", ".join(params.responsibilities) if params.responsibilities else "Not specified"
    })

    return JobDescriptionOutput(
        description=response.strip(),
        token_usage={
            "current_call": token_tracker.get_usage(),
            "cumulative": get_token_usage_stats()["cumulative"]
        }
    )
def generate_structured_job_description(params: JobDescriptionParams) -> JobDescriptionSections:
    """Generate a job description with structured output using PydanticOutputParser"""
    
    # Initialize the parser with our output schema
    parser = PydanticOutputParser(pydantic_object=JobDescriptionSections)
    
    # Create a template that includes format instructions
    template = """
Create a professional job description for a {jobTitle} position.

Company Name: {companyName}
Company Description: {companyDescription}
Must-have skills: {mustHaveSkills}
Nice-to-have skills: {niceToHaveSkills}
Experience Level: {experienceLevel}
Location: {location}
Work Type: {workType}

Key Responsibilities: {responsibilities}

{format_instructions}
"""

    # Create the prompt with format instructions
    prompt = PromptTemplate(
        input_variables=[
            "jobTitle", "companyName", "companyDescription", "mustHaveSkills", 
            "niceToHaveSkills", "experienceLevel", "location", 
            "workType", "responsibilities"
        ],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Get the LLM
    chat = get_llm()
    
    # Create and run the chain
    chain = LLMChain(llm=chat, prompt=prompt)
    
    # Format the inputs
    formatted_input = {
        "jobTitle": params.jobTitle,
        "companyName": params.companyName,
        "companyDescription": params.companyDescription or "A leading technology company",
        "mustHaveSkills": ", ".join(params.mustHaveSkills) if params.mustHaveSkills else "Not specified",
        "niceToHaveSkills": ", ".join(params.niceToHaveSkills) if params.niceToHaveSkills else "Not specified",
        "experienceLevel": params.experienceLevel or "Not specified",
        "location": params.location or "Not specified",
        "workType": params.workType or "Not specified",
        "responsibilities": ", ".join(params.responsibilities) if params.responsibilities else "Not specified"
    }
    
    # Run the chain
    response = chain.run(formatted_input)
    
    # Parse the output into our structured format
    structured_output = parser.parse(response)
    
    return structured_output


def generate_advanced_job_description(params: JobDescriptionParams) -> dict:
    """Generate a job description using a multi-step reasoning chain with RAG context"""
    
    # Get the LLM
    llm = get_llm()
    
    # Step 1: Analyze the job requirements
    job_analysis_template = """
    Analyze the following job requirements for a {jobTitle} position:
    
    Must-have skills: {mustHaveSkills}
    Nice-to-have skills: {niceToHaveSkills}
    Experience Level: {experienceLevel}
    
    Provide a detailed analysis of:
    1. The core technical competencies needed
    2. The level of expertise required
    3. How these requirements compare to industry standards
    """
    
    job_analysis_prompt = PromptTemplate(
        input_variables=["jobTitle", "mustHaveSkills", "niceToHaveSkills", "experienceLevel"],
        template=job_analysis_template
    )
    
    job_analysis_chain = LLMChain(
        llm=llm,
        prompt=job_analysis_prompt,
        output_key="job_analysis"
    )
    
    # Step 2: Create a company profile
    company_profile_template = """
    Based on this company description and name:
    
    Company Name: {companyName}
    Company Description: {companyDescription}
    
    Create a compelling company profile that would attract candidates for a {jobTitle} position.
    Focus on the company's mission, culture, and why it's a great place to work.
    """
    
    company_profile_prompt = PromptTemplate(
        input_variables=["companyName", "companyDescription", "jobTitle"],
        template=company_profile_template
    )
    
    company_profile_chain = LLMChain(
        llm=llm,
        prompt=company_profile_prompt,
        output_key="company_profile"
    )
    
    # Step 3: Generate industry insights regardless of input
    industry_insights_template = """
    Based on the job title {jobTitle} and the skills {mustHaveSkills}, 
    provide industry insights that would be relevant for this role.
        """
    
    industry_insights_prompt = PromptTemplate(
        input_variables=["jobTitle", "mustHaveSkills"],
        template=industry_insights_template
    )
    
    industry_insights_chain = LLMChain(
        llm=llm,
        prompt=industry_insights_prompt,
        output_key="industry_insights"
    )
    
    # Step 4: Create the final job description with structured output
    response_schemas = [
        ResponseSchema(name="company_overview", description="Overview of the company including company name"),
        ResponseSchema(name="job_overview", description="Overview of the job position"),
        ResponseSchema(name="responsibilities", description="Key responsibilities for the role"),
        ResponseSchema(name="requirements", description="Required skills and qualifications"),
        ResponseSchema(name="nice_to_have", description="Nice-to-have skills and qualifications"),
        ResponseSchema(name="benefits", description="Benefits and perks of the job")
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    final_template = """
    Using the following detailed analysis and information, create a structured job description:
    
    Job Title: {jobTitle}
    Company Name: {companyName}
    Company Profile:
    {company_profile}
    
    Job Analysis:
    {job_analysis}
    
    Industry Insights:
    {industry_insights}
    
    Must-have skills: {mustHaveSkills}
    Nice-to-have skills: {niceToHaveSkills}
    Experience Level: {experienceLevel}
    Location: {location}
    Work Type: {workType}
    
    Key Responsibilities: {responsibilities}
    
    {format_instructions}
    
    Create a comprehensive, compelling, and professional job description.
    """
    
    final_prompt = PromptTemplate(
        input_variables=[
            "jobTitle", "companyName", "company_profile", "job_analysis", "industry_insights",
            "mustHaveSkills", "niceToHaveSkills", "experienceLevel", 
            "location", "workType", "responsibilities"
        ],
        template=final_template,
        partial_variables={"format_instructions": format_instructions}
    )
    
    final_chain = LLMChain(
        llm=llm,
        prompt=final_prompt,
        output_key="structured_output"
    )
    
    # Always use all chains including industry insights
    sequential_chain = SequentialChain(
        chains=[job_analysis_chain, company_profile_chain, industry_insights_chain, final_chain],
        input_variables=[
            "jobTitle", "companyName", "companyDescription",
            "mustHaveSkills", "niceToHaveSkills", "experienceLevel", 
            "location", "workType", "responsibilities"
        ],
        output_variables=["job_analysis", "company_profile", "industry_insights", "structured_output"],
        verbose=True
    )
        
    # Format inputs
    chain_input = {
        "jobTitle": params.jobTitle,
        "companyName": params.companyName,
        "companyDescription": params.companyDescription or "A leading technology company",
        "mustHaveSkills": ", ".join(params.mustHaveSkills) if params.mustHaveSkills else "Not specified",
        "niceToHaveSkills": ", ".join(params.niceToHaveSkills) if params.niceToHaveSkills else "Not specified",
        "experienceLevel": params.experienceLevel or "Not specified",
        "location": params.location or "Not specified",
        "workType": params.workType or "Not specified",
        "responsibilities": ", ".join(params.responsibilities) if params.responsibilities else "Not specified",
    }
    
    # Run the chain
    result = sequential_chain(chain_input)
    
    # Parse the structured output
    structured_sections = output_parser.parse(result["structured_output"])
    
    # Return both the intermediate analysis and the final structured output
    return {
        "analysis": {
            "job_analysis": result["job_analysis"],
            "company_profile": result["company_profile"],
            "industry_insights": result["industry_insights"]
        },
        "job_description": {
            "company_overview": structured_sections["company_overview"],
            "job_overview": structured_sections["job_overview"],
            "responsibilities": structured_sections["responsibilities"],
            "requirements": structured_sections["requirements"],
            "nice_to_have": structured_sections["nice_to_have"],
            "benefits": structured_sections["benefits"]
        }
    }

def format_job_description_markdown(sections: Dict[str, str]) -> str:
    """Format structured job description sections as markdown"""
    return f"""# About the Company
{sections.get('company_overview', 'N/A')}

# Job Overview
{sections.get('job_overview', 'N/A')}

# Responsibilities
{sections.get('responsibilities', 'N/A')}

# Requirements
{sections.get('requirements', 'N/A')}

# Nice-to-Have Skills
{sections.get('nice_to_have', 'N/A')}

# Benefits
{sections.get('benefits', 'N/A')}"""


# Legacy implementations kept for backward compatibility
def generate_structured_job_description(params: JobDescriptionParams) -> JobDescriptionSections:
    """LEGACY: Use generate_job_description() instead"""
    print("⚠️ Deprecated: Use async generate_job_description() for agentic AI features")
    import asyncio
    result = asyncio.run(generate_job_description(params))
    
    # Convert to legacy format
    jd = result['job_description']
    return JobDescriptionSections(
        companyName=params.companyName,
        company_overview=jd.get('company_overview', ''),
        job_overview=jd.get('job_overview', ''),
        responsibilities=jd.get('responsibilities', ''),
        requirements=jd.get('requirements', ''),
        nice_to_have=jd.get('nice_to_have', ''),
        benefits=jd.get('benefits', '')
    )


def generate_advanced_job_description(params: JobDescriptionParams) -> dict:
    """LEGACY: Use generate_job_description() instead"""
    print("⚠️ Deprecated: Use async generate_job_description() for enhanced agentic AI")
    import asyncio
    return asyncio.run(generate_job_description(params))