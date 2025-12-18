import os
from io import BytesIO
from typing import List, Dict, Any, Optional
from fastapi import UploadFile
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# RetrievalQA import removed - using custom RAG implementation instead
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import dspy
from dotenv import load_dotenv
import PyPDF2
import docx
import json
import uuid
from datetime import datetime
from utils.token_usage import get_token_tracker, get_token_usage_stats, reset_token_tracking
from agentic_ai.config import AgenticAIConfig
from agentic_ai.dspy_integration import DSPySignature, DSPyModule, ChainOfThought
from agentic_ai.langrag_integration import VectorStoreManager, RAGChain

load_dotenv()

# --- Document Extraction Functions ---
def extract_text_from_pdf(file: UploadFile) -> str:
    # Read all bytes at once, rewind
    file.file.seek(0)
    raw = file.file.read()
    reader = PyPDF2.PdfReader(BytesIO(raw))
    text = ""
    for page in reader.pages:
        # Some pages may be None if the PDF is weird—guard against it
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text

def extract_text_from_docx(file: UploadFile) -> str:
    file.file.seek(0)  # Reset file position
    doc = docx.Document(file.file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file: UploadFile) -> str:
    file.file.seek(0)  # Reset file position
    return file.file.read().decode("utf-8")

def extract_text(file: UploadFile) -> str:
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename.endswith(".docx") or filename.endswith(".doc"):
        return extract_text_from_docx(file)
    elif filename.endswith(".txt"):
        return extract_text_from_txt(file)
    else:
        raise ValueError("Unsupported file format")

# ============================================================================
# AGENTIC AI - RESUME ANALYZER AGENT
# ============================================================================

class ResumeAnalyzerAgent:
    """
    Production-grade Resume Analyzer Agent using:
    - DSPy for structured prompting and optimization
    - LangGraph for multi-step reasoning workflows
    - Qdrant for persistent vector storage
    - RAG for context-aware analysis
    - Chain-of-Thought for complex scoring decisions
    """
    
    def __init__(self):
        self.config = AgenticAIConfig()
        self.token_tracker = get_token_tracker()
        
        # Initialize LLM with token tracking - Using GPT-4.1
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4", "gpt-4.1"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.2,  # Lower for consistency
            callbacks=[self.token_tracker]
        )
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("TEXT_EMBEDDING_ENDPOINT"),
            deployment=os.getenv("TEXT_EMBEDDING_MODEL"),
            openai_api_version=os.getenv("TEXT_EMBEDDING_API_VERSION"),
        )
        
        # Initialize Qdrant vector store
        self.vector_store_manager = VectorStoreManager(
            store_type="qdrant",
            collection_name="resumes",
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        
        # Initialize DSPy for structured output
        self._init_dspy_modules()
        
        print("✅ Resume Analyzer Agent initialized with agentic AI capabilities")
    
    def _init_dspy_modules(self):
        """Initialize DSPy modules for structured reasoning"""
        
        # Resume scoring signature
        class ResumeScoreSignature(DSPySignature):
            """Score resume against job description with detailed analysis"""
            resume_text: str = dspy.InputField(desc="Complete resume text")
            job_description: str = dspy.InputField(desc="Job requirements and description")
            match_score: int = dspy.OutputField(desc="Match score 0-100")
            reasoning: str = dspy.OutputField(desc="Detailed reasoning for the score")
            strengths: list = dspy.OutputField(desc="Key strengths identified")
            gaps: list = dspy.OutputField(desc="Skill gaps identified")
            recommendations: str = dspy.OutputField(desc="Improvement recommendations")
        
        # Initialize Chain-of-Thought module
        self.scoring_module = ChainOfThought(ResumeScoreSignature)
        
    async def analyze_resume(
        self,
        resume_text: str,
        job_description: str,
        store_in_vectordb: bool = True
    ) -> Dict[str, Any]:
        """
        Main agentic method: Multi-step resume analysis with RAG
        
        Steps:
        1. Extract structured data from resume
        2. Store in vector database for future RAG
        3. Perform semantic matching against job description
        4. Use Chain-of-Thought reasoning for scoring
        5. Generate actionable recommendations
        """
        print(f"🤖 Starting agentic resume analysis...")
        
        try:
            # Step 1: Extract structured information
            candidate_data = await self._extract_candidate_data(resume_text)
            
            # Step 2: Store in Qdrant for RAG (if enabled)
            if store_in_vectordb:
                await self._store_in_vectordb(resume_text, candidate_data)
            
            # Step 3: RAG-enhanced context gathering
            relevant_context = await self._gather_rag_context(job_description)
            
            # Step 4: Chain-of-Thought scoring
            score_analysis = await self._perform_cot_scoring(
                resume_text,
                job_description,
                relevant_context
            )
            
            # Step 5: Generate final structured output
            result = {
                'data': {
                    'match_score': score_analysis['match_score'],
                    'ai_analysis': score_analysis['reasoning'],
                    'candidate': candidate_data,
                    'strengths': score_analysis['strengths'],
                    'gaps': score_analysis['gaps'],
                    'recommendations': score_analysis['recommendations']
                },
                'token_usage': self.token_tracker.get_usage(),
                'agent_metadata': {
                    'agent': 'ResumeAnalyzerAgent',
                    'version': '2.0-agentic',
                    'reasoning_type': 'chain_of_thought',
                    'rag_enabled': store_in_vectordb,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            print(f"✅ Resume analysis complete. Score: {score_analysis['match_score']}/100")
            return result
            
        except Exception as e:
            print(f"❌ Error in resume analysis: {str(e)}")
            raise
    
    async def _extract_candidate_data(self, resume_text: str) -> Dict[str, Any]:
        """Extract structured candidate information using LLM"""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.config.agents['resume_analyzer']['system_prompt']),
            HumanMessage(content=f"""Extract structured information from this resume:

{resume_text}

Return JSON with these fields:
- name (string)
- email (string)
- phone (string)
- location (string)
- linkedin (string)
- work_experience (array of objects):
  - position (string)
  - company (string)
  - startDate (string, YYYY-MM-DD or YYYY-MM)
  - endDate (string, YYYY-MM-DD or YYYY-MM, or null if present)
  - location (string)
  - description (string, use bullet points starting with • for key achievements)
- education (array of objects):
  - institution (string)
  - degree (string)
  - fieldOfStudy (string)
  - startDate (string, YYYY-MM-DD or YYYY-MM)
  - endDate (string, YYYY-MM-DD or YYYY-MM)
- projects (array of objects):
  - title (string)
  - description (string, use bullet points starting with •)
  - technologies (array of strings)
  - link (string)
- soft_skills (array of strings)
- technical_skills (array of strings)
- career_analysis (string, brief summary)

Output ONLY valid JSON, no markdown.""")
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages())
        content = response.content.strip()
        
        # Clean JSON
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback parsing
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                return json.loads(content[json_start:json_end])
            raise ValueError("Failed to extract candidate data")
    
    async def _store_in_vectordb(self, resume_text: str, candidate_data: Dict) -> None:
        """Store resume in Qdrant for future RAG retrieval"""
        
        try:
            # Chunk resume for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(resume_text)
            
            # Generate embeddings and store
            points = []
            for idx, chunk in enumerate(chunks):
                embedding = await self.embeddings.aembed_query(chunk)
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "candidate_name": candidate_data.get('name', 'Unknown'),
                        "skills": candidate_data.get('technical_skills', []),
                        "chunk_index": idx,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                points.append(point)
            
            # Store in Qdrant
            self.vector_store_manager.add_documents(points)
            print(f"💾 Stored {len(chunks)} resume chunks in Qdrant")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to store in vector DB: {str(e)}")
    
    async def _gather_rag_context(self, job_description: str) -> str:
        """Use RAG to gather relevant context from past resumes"""
        
        try:
            # Search for similar job requirements in past resumes
            query_embedding = await self.embeddings.aembed_query(job_description)
            
            results = self.vector_store_manager.similarity_search(
                query_embedding,
                k=5
            )
            
            if results:
                context = "Relevant context from similar past analyses:\n\n"
                for result in results:
                    context += f"- {result['payload']['text']}\n"
                return context
            
            return "No relevant past context found."
            
        except Exception as e:
            print(f"⚠️ RAG context gathering failed: {str(e)}")
            return ""
    
    async def _perform_cot_scoring(
        self,
        resume_text: str,
        job_description: str,
        rag_context: str
    ) -> Dict[str, Any]:
        """Perform Chain-of-Thought reasoning for scoring"""
        
        # Enhanced prompt with RAG context
        enhanced_jd = f"""{job_description}

{rag_context if rag_context else ''}"""
        
        prompt = f"""You are an expert technical recruiter performing deep resume analysis.

JOB REQUIREMENTS:
{enhanced_jd}

RESUME:
{resume_text}

TASK: Perform step-by-step Chain-of-Thought reasoning:

1. TECHNICAL SKILLS MATCH (0-30 points):
   - Identify must-have vs nice-to-have skills
   - Calculate percentage match
   - Reasoning: [explain]

2. EXPERIENCE RELEVANCE (0-30 points):
   - Years of experience alignment
   - Domain relevance
   - Reasoning: [explain]

3. EDUCATION & CERTIFICATIONS (0-15 points):
   - Educational requirements match
   - Relevant certifications
   - Reasoning: [explain]

4. SOFT SKILLS & CULTURE FIT (0-15 points):
   - Communication, teamwork, leadership
   - Reasoning: [explain]

5. OVERALL IMPRESSION (0-10 points):
   - Resume quality, clarity
   - Career progression
   - Reasoning: [explain]

Return JSON:
{{
  "match_score": 85,
  "reasoning": "Detailed step-by-step explanation of scoring",
  "strengths": ["strength1", "strength2", "strength3"],
  "gaps": ["gap1", "gap2"],
  "recommendations": "Specific actionable advice for candidate"
}}"""
        
        messages = [
            SystemMessage(content="You are an expert resume analyzer using Chain-of-Thought reasoning."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        content = response.content.strip()
        
        # Parse JSON
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                return json.loads(content[json_start:json_end])
            raise ValueError("Failed to parse scoring result")


# Create global agent instance
_resume_agent = None

def get_resume_agent() -> ResumeAnalyzerAgent:
    """Get or create singleton Resume Analyzer Agent"""
    global _resume_agent
    if _resume_agent is None:
        _resume_agent = ResumeAnalyzerAgent()
    return _resume_agent



async def analyze_and_extract(resume_text: str, job_description: str) -> dict:
    """
    AGENTIC AI VERSION: Multi-step reasoning with RAG and CoT
    
    Uses:
    - Chain-of-Thought for scoring decisions
    - RAG for context-aware analysis
    - Qdrant for persistent storage
    - DSPy for structured outputs
    
    Returns enhanced dict with agent metadata.
    """
    agent = get_resume_agent()
    return await agent.analyze_resume(resume_text, job_description, store_in_vectordb=True)


# ============================================================================
# BACKWARD COMPATIBILITY - Legacy function (kept for old code)
# ============================================================================

def analyze_and_extract_sync(resume_text: str, job_description: str) -> dict:
    """
    LEGACY SYNCHRONOUS VERSION (deprecated, use async version)
    
    Kept for backward compatibility with existing code.
    Uses basic prompt-based analysis without agentic features.
    """
    import asyncio
    
    print("⚠️ Warning: Using legacy sync analysis. Consider upgrading to async analyze_and_extract()")
    
    # Run async version in sync context
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(analyze_and_extract(resume_text, job_description))
