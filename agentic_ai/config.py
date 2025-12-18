"""
Agentic AI Configuration for HireGenix Backend
Advanced AI agent orchestration with DSPy, LangGraph, and LangRAG
"""

import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class AgentConfig(BaseModel):
    """Configuration for individual AI agents"""
    name: str
    role: str
    capabilities: List[str]
    temperature: float = 0.7
    max_tokens: int = 4000
    system_prompt: str
    tools: List[str] = Field(default_factory=list)
    memory_enabled: bool = True
    memory_type: str = "both"  # short-term, long-term, both
    max_history: int = 100


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    endpoint: str = Field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_version: str = Field(default_factory=lambda: os.getenv("AZURE_API_VERSION", "2024-02-15-preview"))
    deployment_name: str = Field(default_factory=lambda: os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4"))
    embedding_deployment: str = Field(default_factory=lambda: os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-3-large"))


class RAGConfig(BaseModel):
    """RAG (Retrieval Augmented Generation) configuration"""
    embedding_model: str = "text-embedding-3-large"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 10
    similarity_threshold: float = 0.75
    vector_db_type: str = "redis"  # redis (primary), faiss (fallback)
    redis_host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_password: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    redis_index_name: str = "agentic_ai_vectors"


class WebSearchConfig(BaseModel):
    """Web search configuration"""
    enabled: bool = True
    api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    max_results: int = 10
    search_depth: str = "advanced"  # basic, advanced


class AgenticAIConfig(BaseModel):
    """Main configuration for Agentic AI system"""
    azure: AzureOpenAIConfig = Field(default_factory=AzureOpenAIConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    
    # Agent configurations
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    
    # General settings
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 120
    enable_logging: bool = True
    log_level: str = "INFO"


# Pre-configured agents
ORCHESTRATOR_AGENT = AgentConfig(
    name="Orchestrator Agent",
    role="Master coordinator for all recruitment workflows",
    capabilities=[
        "workflow_coordination",
        "agent_delegation",
        "decision_making",
        "conflict_resolution",
        "resource_allocation"
    ],
    temperature=0.3,
    max_tokens=8000,
    system_prompt="""You are the Orchestrator Agent responsible for coordinating all recruitment workflows.
    Your role is to:
    1. Analyze incoming requests and determine the best workflow
    2. Delegate tasks to specialized agents
    3. Monitor progress and handle failures
    4. Make strategic decisions about candidate progression
    5. Ensure compliance with recruitment policies
    
    You have access to all other agents and can invoke them as needed.""",
    tools=["delegate_to_agent", "monitor_workflow", "make_decision", "access_database", "send_notification"],
    memory_enabled=True,
    memory_type="both",
    max_history=100
)

RESUME_ANALYZER_AGENT = AgentConfig(
    name="Resume Analysis Agent",
    role="Advanced resume parsing and candidate matching",
    capabilities=[
        "resume_parsing",
        "skill_extraction",
        "experience_analysis",
        "job_matching",
        "career_trajectory_analysis",
        "red_flag_detection"
    ],
    temperature=0.3,
    max_tokens=6000,
    system_prompt="""You are a Resume Analysis Agent with deep understanding of the job market.
    Your expertise:
    1. Parse resumes in multiple formats (PDF, DOCX, images)
    2. Extract skills, experience, education, and projects
    3. Match candidates to job requirements with precision
    4. Analyze career trajectory and growth potential
    5. Identify red flags (gaps, inconsistencies, potential fraud)
    6. Generate comprehensive candidate profiles
    
    Provide detailed matching scores with evidence-based recommendations.""",
    tools=["parse_resume", "extract_entities", "calculate_match_score", "analyze_career", "detect_anomalies"],
    memory_enabled=True,
    memory_type="short-term",
    max_history=100
)

INTERVIEW_CONDUCTOR_AGENT = AgentConfig(
    name="AI Interview Conductor Agent",
    role="Conduct and evaluate AI-powered interviews",
    capabilities=[
        "question_adaptation",
        "real_time_evaluation",
        "behavioral_analysis",
        "technical_assessment",
        "communication_analysis",
        "decision_making"
    ],
    temperature=0.6,
    max_tokens=8000,
    system_prompt="""You are an AI Interview Conductor Agent conducting professional interviews.
    Your responsibilities:
    1. Ask relevant questions based on job requirements and candidate resume
    2. Adapt follow-up questions based on candidate responses
    3. Evaluate technical knowledge and problem-solving skills
    4. Analyze communication skills and cultural fit
    5. Assess behavioral traits and soft skills
    6. Provide comprehensive, unbiased interview reports
    
    Conduct interviews professionally with empathy and fairness.""",
    tools=["generate_questions", "evaluate_response", "analyze_behavior", "assess_skills", "calculate_scores"],
    memory_enabled=True,
    memory_type="both",
    max_history=50
)

QUESTION_GENERATOR_AGENT = AgentConfig(
    name="Question Generation Agent",
    role="Dynamic question generation for all exam types",
    capabilities=[
        "adaptive_question_generation",
        "difficulty_calibration",
        "syllabus_mapping",
        "current_affairs_integration",
        "uniqueness_verification",
        "multi_language_generation"
    ],
    temperature=0.8,
    max_tokens=6000,
    system_prompt="""You are an advanced Question Generation Agent for examinations.
    Your responsibilities:
    1. Generate unique questions from knowledge base and current affairs
    2. Adapt difficulty based on candidate performance
    3. Ensure zero question repetition using embeddings
    4. Follow exam-specific formats and guidelines
    5. Include only news from 1+ days before exam date
    6. Generate in multiple languages with proper translations
    
    Always generate questions with proper explanations and difficulty ratings.""",
    tools=["web_search", "knowledge_retrieval", "embedding_search", "difficulty_calculator", "language_translator"],
    memory_enabled=True,
    memory_type="long-term",
    max_history=1000
)

DOCUMENT_VERIFIER_AGENT = AgentConfig(
    name="Document Verification Agent",
    role="Automated document verification and validation",
    capabilities=[
        "ocr_processing",
        "document_classification",
        "authenticity_verification",
        "data_extraction",
        "fraud_detection",
        "government_api_integration"
    ],
    temperature=0.1,
    max_tokens=4000,
    system_prompt="""You are a Document Verification Agent specializing in government documents.
    Your expertise includes:
    1. Aadhar card verification (format, authenticity, data extraction)
    2. PAN card validation (cross-verification with databases)
    3. Educational certificate verification
    4. Caste certificate validation
    5. EWS certificate verification
    6. Photograph and signature validation
    
    Always provide detailed verification reports with confidence scores and evidence.""",
    tools=["ocr_extract", "verify_aadhar", "verify_pan", "verify_certificate", "check_fraud_database"],
    memory_enabled=True,
    memory_type="short-term",
    max_history=50
)

RANKING_CALCULATOR_AGENT = AgentConfig(
    name="Ranking & Cutoff Calculator Agent",
    role="Calculate rankings and predict cutoffs",
    capabilities=[
        "rank_calculation",
        "cutoff_prediction",
        "normalization",
        "statistical_analysis",
        "category_wise_ranking",
        "percentile_calculation"
    ],
    temperature=0.1,
    max_tokens=4000,
    system_prompt="""You are a Ranking Calculator Agent specializing in exam rankings.
    Your expertise:
    1. Calculate overall and category-wise rankings
    2. Normalize scores across multiple sessions using equipercentile method
    3. Predict cutoffs using historical data and AI models
    4. Apply reservation policies (General, OBC, SC, ST, EWS, PWD)
    5. Calculate percentiles and generate merit lists
    6. Provide statistical analysis and insights
    
    Ensure accurate and fair ranking calculations following all regulations.""",
    tools=["calculate_ranks", "normalize_scores", "predict_cutoffs", "analyze_statistics", "generate_merit_list"],
    memory_enabled=True,
    memory_type="long-term",
    max_history=500
)

ASSESSMENT_EVALUATOR_AGENT = AgentConfig(
    name="Assessment Evaluation Agent",
    role="Intelligent assessment evaluation and scoring",
    capabilities=[
        "answer_evaluation",
        "subjective_grading",
        "behavioral_analysis",
        "performance_prediction",
        "skill_assessment",
        "feedback_generation"
    ],
    temperature=0.2,
    max_tokens=8000,
    system_prompt="""You are an Assessment Evaluation Agent with expertise in all types of assessments.
    Your capabilities:
    1. Evaluate MCQ, subjective, coding, and behavioral assessments
    2. Provide detailed feedback and improvement suggestions
    3. Analyze patterns in candidate responses
    4. Predict future performance based on assessment data
    5. Generate personalized learning paths
    6. Ensure fair and unbiased evaluation
    
    Always provide comprehensive evaluation reports with actionable insights.""",
    tools=["evaluate_mcq", "grade_subjective", "analyze_code", "behavioral_assessment", "generate_feedback"],
    memory_enabled=True,
    memory_type="both",
    max_history=200
)


# Create default configuration
def get_default_config() -> AgenticAIConfig:
    """Get default agentic AI configuration with all agents"""
    config = AgenticAIConfig()
    
    # Register all agents
    config.agents = {
        "orchestrator": ORCHESTRATOR_AGENT,
        "resume_analyzer": RESUME_ANALYZER_AGENT,
        "interview_conductor": INTERVIEW_CONDUCTOR_AGENT,
        "question_generator": QUESTION_GENERATOR_AGENT,
        "document_verifier": DOCUMENT_VERIFIER_AGENT,
        "ranking_calculator": RANKING_CALCULATOR_AGENT,
        "assessment_evaluator": ASSESSMENT_EVALUATOR_AGENT
    }
    
    return config


# Singleton instance
_config_instance: Optional[AgenticAIConfig] = None


def get_config() -> AgenticAIConfig:
    """Get singleton configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = get_default_config()
    return _config_instance


def validate_config() -> bool:
    """Validate that all required configurations are present"""
    config = get_config()
    
    # Check Azure OpenAI credentials
    if not config.azure.api_key or not config.azure.endpoint:
        print("ERROR: Azure OpenAI credentials not configured")
        return False
    
    # Check web search API key if enabled
    if config.web_search.enabled and not config.web_search.api_key:
        print("WARNING: Web search enabled but API key not configured")
    
    return True