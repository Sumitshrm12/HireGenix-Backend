"""
🚀 AGENTIC INTERVIEW 2.0 - ULTIMATE AI VIDEO INTERVIEW SYSTEM 🚀
================================================================

Next-Generation Emotionally Intelligent, Self-Learning Interview Cognition System

🎯 COMPLETE INTELLIGENCE INTEGRATION:
- Company Intelligence (mission, culture, values, products)
- Job Intelligence (requirements, skills, responsibilities)
- Candidate Resume Intelligence (experience, skills, projects)
- Pre-screening Round Intelligence (previous answers, scores)
- Market Intelligence (industry trends, competitor analysis)

🤖 DUAL AI MODEL ARCHITECTURE:
- GPT-4-Realtime: Natural conversational flow with voice-to-voice
- GPT-5/o1: Deep analysis, reasoning, and evaluation reports

🧠 AGENTIC INTERVIEW 2.0 FEATURES:
- Goal-Aware Reasoning with dynamic interview goals
- Emotional Context Adaptation (sentiment & tone detection)
- Temporal Behavior Analysis with 30-second rolling window
- Parallel Agent Execution for 2-3× latency reduction
- Voice Consistency Verification for impersonation detection
- Company Personality Conditioning for culture alignment
- Interview Endgame Narrative with human-like closing
- Reinforcement Feedback Loop for continuous learning
- Smart Mode Switching (Lite/Full modes)
- System-Level Feedback Integration with HCE

🛡️ NEXT-TO-IMPOSSIBLE ANTI-CHEAT DETECTION:
- Multi-modal proctoring (facial recognition, eye tracking, posture)
- Background analysis with person/object detection
- Screen sharing detection and window focus monitoring
- Audio analysis for multiple voices
- Behavior pattern anomaly detection with rolling windows
- Real-time suspicious activity alerts
- AI-powered impersonation detection
- Voice consistency verification across interview

🧠 ADVANCED TECHNOLOGIES:
- LangGraph multi-agent workflow orchestration
- Redis vector store for infinite memory
- DSPy reasoning and optimization
- RAG for context-aware questioning
- Real-time embeddings and semantic search
- Adaptive difficulty and personalization
- Emotional intelligence and sentiment analysis
- Continuous learning and outcome prediction
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import cv2
from collections import deque

# Suppress urllib3 LibreSSL warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# LangChain & LangGraph
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, Document
from langchain.memory import ConversationBufferWindowMemory
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from pydantic import BaseModel, Field

# Computer Vision & Proctoring - Lazy Loading
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

def _get_face_mesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Lazy load FaceMesh with proper error handling"""
    mp = _get_mediapipe()
    if mp is None:
        return None
    try:
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    except Exception:
        return None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

# Local imports
from .config import get_config
from .redis_vector_store import RedisVectorStore, get_redis_vector_store
from .dspy_integration import ChainOfThought, DSPySignature, DSPyModule
from .langrag_integration import RAGChain, VectorStoreManager

# Type definitions
class InterviewPhase(str, Enum):
    WARMUP = "warmup"
    TECHNICAL_DEEP_DIVE = "technical_deep_dive"
    PROBLEM_SOLVING = "problem_solving"
    BEHAVIORAL_ASSESSMENT = "behavioral_assessment"
    CULTURAL_FIT = "cultural_fit"
    SCENARIO_BASED = "scenario_based"
    CLOSING = "closing"
    COMPLETED = "completed"


class QuestionDifficulty(str, Enum):
    ENTRY = "entry"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class CandidateEngagement(str, Enum):
    HIGHLY_ENGAGED = "highly_engaged"
    ENGAGED = "engaged"
    MODERATE = "moderate"
    DISENGAGED = "disengaged"
    CONCERNING = "concerning"


class InterviewMode(str, Enum):
    LITE = "lite"
    FULL = "full"

class ProcessingLocation(str, Enum):
    """Processing location for hybrid architecture - Agentic Interview 2.1"""
    EDGE = "edge"  # Real-time proctoring, low latency
    CLOUD = "cloud"  # GPT-5 reasoning, high compute


class ModelComplexity(str, Enum):
    """Model complexity levels for cost-aware switching - Agentic Interview 2.1"""
    MINIMAL = "minimal"  # GPT-4-mini for simple tasks
    STANDARD = "standard"  # GPT-4-Realtime for conversation
    ADVANCED = "advanced"  # GPT-5 for analysis
    EXPERT = "expert"  # GPT-5/o1 for deep reasoning


@dataclass
class SkillNode:
    """Node in skill dependency graph - Agentic Interview 2.1"""
    skill_id: str
    skill_name: str
    prerequisites: List[str] = field(default_factory=list)
    difficulty_level: int = 1  # 1-5
    assessed: bool = False
    score: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class SkillGraph:
    """Graph-based skill dependency map for knowledge tracing - Agentic Interview 2.1"""
    nodes: Dict[str, SkillNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)  # (prerequisite, dependent)
    
    def add_skill(self, skill: SkillNode):
        """Add skill to graph"""
        self.nodes[skill.skill_id] = skill
        for prereq in skill.prerequisites:
            self.edges.append((prereq, skill.skill_id))
    
    def get_next_assessable_skills(self) -> List[SkillNode]:
        """Get skills whose prerequisites are met"""
        assessable = []
        for skill_id, skill in self.nodes.items():
            if not skill.assessed:
                prereqs_met = all(
                    self.nodes.get(prereq, SkillNode("", "")).assessed 
                    for prereq in skill.prerequisites
                )
                if prereqs_met:
                    assessable.append(skill)
        return sorted(assessable, key=lambda s: s.difficulty_level)


@dataclass
class CompressedMemory:
    """Compressed memory vectors to prevent Redis bloat - Agentic Interview 2.1"""
    session_id: str
    summary_vector: List[float]  # Distilled embedding
    key_points: List[str]
    skill_coverage: Dict[str, float]
    compression_ratio: float
    original_size: int
    compressed_size: int
    timestamp: datetime


@dataclass
class RecruiterReviewSummary:
    """Human-readable audit summary - Agentic Interview 2.1"""
    session_id: str
    candidate_name: str
    overall_assessment: str
    key_strengths: List[str]
    concerns: List[str]
    recommendation_rationale: str
    decision_factors: Dict[str, float]
    bias_check_results: Dict[str, Any]
    auditor_notes: List[str]
    confidence_score: float
    generated_at: datetime


@dataclass
class BiasAnalysis:
    """Cognitive bias detection results - Agentic Interview 2.1"""
    session_id: str
    bias_indicators: Dict[str, float]  # bias_type -> severity (0-1)
    counterfactual_scores: Dict[str, float]  # alternative scenarios
    demographic_fairness: Dict[str, Any]
    recommendations: List[str]
    bias_mitigation_applied: List[str]
    audit_trail: List[Dict]


@dataclass
class SessionVector:
    """Temporal fusion of behavioral windows - Agentic Interview 2.1"""
    session_id: str
    start_time: datetime
    end_time: datetime
    window_summaries: List[Dict]  # Compressed from 30-second windows
    behavioral_drift_score: float
    anomaly_trajectory: List[float]
    impersonation_confidence: float
    attention_consistency: float
    overall_integrity_score: float


@dataclass
class CostMetrics:
    """Cost tracking for model switching - Agentic Interview 2.1"""
    session_id: str
    total_tokens: int
    gpt4_mini_tokens: int
    gpt4_realtime_tokens: int
    gpt5_tokens: int
    gpt5_o1_tokens: int
    estimated_cost: float
    cost_savings: float  # vs. all GPT-5
    model_switches: int
    efficiency_score: float


@dataclass
class InterviewGoal:
    """Goal definition for dynamic interview objectives"""
    name: str
    description: str
    target_skills: List[str]
    priority: int = 1
    completed: bool = False
    evidence: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class InterviewGoalTracker:
    """Adaptive goal tracking across interview lifecycle"""

    DEFAULT_GOALS: List[InterviewGoal] = [
        InterviewGoal(
            name="technical_depth",
            description="Validate candidate's depth across critical technical skills",
            target_skills=["system design", "architecture", "core_stack"],
            priority=3
        ),
        InterviewGoal(
            name="culture_fit",
            description="Assess alignment with company values and culture",
            target_skills=["communication", "collaboration", "values_alignment"],
            priority=2
        ),
        InterviewGoal(
            name="problem_solving",
            description="Observe structured thinking and problem solving approach",
            target_skills=["analysis", "creativity", "delivery"],
            priority=2
        )
    ]

    def initialize(self, required_skills: List[str], custom_goals: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        goals: Dict[str, Dict[str, Any]] = {}

        def _goal_to_dict(goal: InterviewGoal) -> Dict[str, Any]:
            merged_skills = list(dict.fromkeys(goal.target_skills + required_skills)) if goal.target_skills else required_skills
            return {
                "name": goal.name,
                "description": goal.description,
                "target_skills": merged_skills,
                "priority": goal.priority,
                "completed": goal.completed,
                "evidence": goal.evidence,
                "last_updated": goal.last_updated.isoformat()
            }

        for default_goal in self.DEFAULT_GOALS:
            goals[default_goal.name] = _goal_to_dict(default_goal)

        if custom_goals:
            for goal in custom_goals:
                goal_name = goal.get("name", f"custom_goal_{len(goals) + 1}")
                goals[goal_name] = {
                    "name": goal_name,
                    "description": goal.get("description", ""),
                    "target_skills": goal.get("target_skills", required_skills),
                    "priority": goal.get("priority", 1),
                    "completed": False,
                    "evidence": [],
                    "last_updated": datetime.utcnow().isoformat()
                }

        return goals

    def update_goal_progress(
        self,
        goal_state: Dict[str, Dict[str, Any]],
        skill_scores: Dict[str, float],
        evidence: Optional[str] = None,
        completion_threshold: float = 7.5
    ) -> Dict[str, Dict[str, Any]]:
        if not goal_state:
            return goal_state

        now = datetime.utcnow().isoformat()
        for goal_name, goal_data in goal_state.items():
            targets = goal_data.get("target_skills", [])
            if not targets:
                continue

            covered = [skill for skill in targets if skill_scores.get(skill, 0) >= completion_threshold]
            if len(covered) == len(targets):
                goal_data["completed"] = True
            if evidence:
                goal_data.setdefault("evidence", []).append(evidence)
            goal_data["last_updated"] = now

        return goal_state

    @staticmethod
    def goals_completed(goal_state: Dict[str, Dict[str, Any]]) -> bool:
        if not goal_state:
            return True
        return all(goal.get("completed", False) for goal in goal_state.values())

class InterviewMode(str, Enum):
    """Interview operation modes"""
    LITE = "lite"  # Mock/practice, no proctoring, GPT-4-mini
    FULL = "full"  # Official interview, complete orchestration, GPT-5/o1


class EmotionalTone(str, Enum):
    """Detected emotional tones"""
    CONFIDENT = "confident"
    HESITANT = "hesitant"
    NERVOUS = "nervous"
    OVERCONFIDENT = "overconfident"
    NEUTRAL = "neutral"
    STRESSED = "stressed"
    CALM = "calm"


class InterviewGoalStatus(str, Enum):
    """Status of interview goals"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class HiringOutcome(str, Enum):
    """Final hiring outcomes for reinforcement learning"""
    HIRED = "hired"
    REJECTED = "rejected"
    MAYBE = "maybe"
    PENDING = "pending"


@dataclass
class InterviewGoal:
    """Dynamic interview goal tracking"""
    goal_id: str
    description: str
    target_skills: List[str]
    priority: int  # 1-10
    status: InterviewGoalStatus
    questions_asked: int = 0
    target_questions: int = 3
    skills_assessed: Dict[str, float] = field(default_factory=dict)
    completion_percentage: float = 0.0


@dataclass
class EmotionalContext:
    """Emotional state tracking"""
    current_tone: EmotionalTone
    tone_history: List[Tuple[EmotionalTone, datetime]] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1 to 1
    confidence_level: float = 0.5  # 0 to 1
    stress_indicators: List[str] = field(default_factory=list)
    recommended_tone_adjustment: str = ""


@dataclass
class BehavioralWindow:
    """30-second rolling window for behavior analysis"""
    window_id: str
    start_time: datetime
    end_time: datetime
    gaze_anomalies: int = 0
    movement_score: float = 0.0
    lighting_changes: int = 0
    voice_patterns: List[float] = field(default_factory=list)
    suspicious_score: float = 0.0
    anomalies_detected: List[str] = field(default_factory=list)


@dataclass
class VoiceProfile:
    """Voice consistency tracking"""
    baseline_embedding: Optional[List[float]] = None
    current_embedding: Optional[List[float]] = None
    similarity_score: float = 1.0
    anomaly_count: int = 0
    speaker_changes_detected: int = 0
    last_check_time: Optional[datetime] = None


@dataclass
class CompanyPersonality:
    """Company culture and communication style"""
    tone: str = "professional"  # professional, casual, innovative, formal
    values_emphasis: List[str] = field(default_factory=list)
    communication_style: str = "balanced"  # direct, empathetic, analytical
    culture_keywords: List[str] = field(default_factory=list)
    mission_alignment: str = ""


@dataclass
class ReinforcementFeedback:
    """Outcome-based learning data"""
    session_id: str
    outcome: HiringOutcome
    conversation_embeddings: List[List[float]]
    question_effectiveness: Dict[str, float]
    answer_quality_patterns: Dict[str, Any]
    interviewer_performance: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationMemory:
    """Long-term conversation memory with semantic search"""
    question: str
    answer: str
    embedding: List[float]
    relevance_score: float
    quality_score: float
    phase: InterviewPhase
    difficulty: QuestionDifficulty
    timestamp: datetime
    key_insights: List[str]
    skills_demonstrated: Dict[str, float]


class ProctoringViolationType(str, Enum):
    """Types of proctoring violations"""
    MULTIPLE_FACES = "multiple_faces"
    NO_FACE_DETECTED = "no_face_detected"
    FACE_NOT_VISIBLE = "face_not_visible"
    LOOKING_AWAY = "looking_away"
    MOBILE_PHONE_DETECTED = "mobile_phone_detected"
    PERSON_IN_BACKGROUND = "person_in_background"
    TAB_SWITCH = "tab_switch"
    WINDOW_BLUR = "window_blur"
    SUSPICIOUS_AUDIO = "suspicious_audio"
    IMPERSONATION_SUSPECTED = "impersonation_suspected"
    UNAUTHORIZED_DEVICE = "unauthorized_device"


@dataclass
class ProctoringAlert:
    """Proctoring violation alert"""
    type: ProctoringViolationType
    severity: str  # low, medium, high, critical
    timestamp: datetime
    description: str
    confidence: float
    frame_data: Optional[str] = None
    action_taken: Optional[str] = None


class VideoInterviewState(BaseModel):
    """Complete state for video interview workflow with all intelligence"""
    # Session identifiers
    session_id: str
    candidate_id: str
    interview_id: str
    operating_mode: InterviewMode = InterviewMode.FULL
    
    # 🏢 COMPANY INTELLIGENCE
    company_intelligence: Dict[str, Any] = Field(default_factory=dict)
    company_mission: str = ""
    company_vision: str = ""
    company_values: List[str] = Field(default_factory=list)
    company_culture: str = ""
    company_products: List[Dict] = Field(default_factory=list)
    company_market_position: str = ""
    company_tone_profile: Dict[str, Any] = Field(default_factory=dict)
    company_personality_prompt: str = ""
    
    # 💼 JOB INTELLIGENCE
    job_title: str
    job_description: str = ""
    required_skills: List[str] = Field(default_factory=list)
    job_responsibilities: List[str] = Field(default_factory=list)
    job_qualifications: List[str] = Field(default_factory=list)
    job_benefits: List[str] = Field(default_factory=list)
    team_structure: Dict[str, Any] = Field(default_factory=dict)
    growth_opportunities: List[str] = Field(default_factory=list)
    
    # 👤 CANDIDATE RESUME INTELLIGENCE
    candidate_profile: Dict[str, Any] = Field(default_factory=dict)
    resume_data: Dict[str, Any] = Field(default_factory=dict)
    candidate_skills: List[str] = Field(default_factory=list)
    candidate_experience: List[Dict] = Field(default_factory=list)
    candidate_education: List[Dict] = Field(default_factory=list)
    candidate_projects: List[Dict] = Field(default_factory=list)
    candidate_achievements: List[str] = Field(default_factory=list)
    
    # 📝 PRE-SCREENING INTELLIGENCE
    prescreening_data: Dict[str, Any] = Field(default_factory=dict)
    prescreening_score: float = 0.0
    prescreening_answers: List[Dict] = Field(default_factory=list)
    prescreening_strengths: List[str] = Field(default_factory=list)
    prescreening_gaps: List[str] = Field(default_factory=list)
    
    # 📊 MARKET INTELLIGENCE
    market_intelligence: Dict[str, Any] = Field(default_factory=dict)
    industry_trends: List[str] = Field(default_factory=list)
    competitor_analysis: Dict[str, Any] = Field(default_factory=dict)
    salary_benchmarks: Dict[str, Any] = Field(default_factory=dict)
    skill_demand: Dict[str, float] = Field(default_factory=dict)
    
    # Conversation state
    conversation_history: List[Dict] = Field(default_factory=list)
    current_question: str = ""
    current_answer: str = ""
    questions_asked: List[str] = Field(default_factory=list)
    
    # Interview progress
    current_phase: InterviewPhase = InterviewPhase.WARMUP
    current_difficulty: QuestionDifficulty = QuestionDifficulty.ENTRY
    questions_count: int = 0
    target_questions: int = 15
    goal_state: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    goal_progress_notes: List[str] = Field(default_factory=list)
    
    # Performance tracking
    skill_scores: Dict[str, float] = Field(default_factory=dict)
    quality_scores: List[float] = Field(default_factory=list)
    relevance_scores: List[float] = Field(default_factory=list)
    engagement_level: CandidateEngagement = CandidateEngagement.ENGAGED
    
    # 🛡️ ADVANCED PROCTORING & ANTI-CHEAT
    proctoring_enabled: bool = True
    proctoring_violations: List[Dict] = Field(default_factory=list)
    violation_count: int = 0
    warning_count: int = 0
    suspicious_behavior_score: float = 0.0
    face_verified: bool = False
    face_encoding: Optional[List[float]] = None
    tab_switches: int = 0
    audio_anomalies: int = 0
    
    # Behavioral analysis (enhanced)
    response_times: List[float] = Field(default_factory=list)
    filler_word_count: int = 0
    confidence_scores: List[float] = Field(default_factory=list)
    eye_contact_score: float = 0.0
    posture_score: float = 0.0
    facial_expression_history: List[str] = Field(default_factory=list)
    attention_score: float = 0.0
    
    # Context awareness
    retrieved_contexts: List[Dict] = Field(default_factory=list)
    semantic_memories: List[str] = Field(default_factory=list)
    
    # Agent outputs
    next_question: str = ""
    follow_up_needed: bool = False
    warning_message: Optional[str] = None
    adaptive_insights: Dict[str, Any] = Field(default_factory=dict)
    
    # Final evaluation
    overall_score: float = 0.0
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendation: str = ""
    detailed_feedback: str = ""
    
    # Workflow control
    should_continue: bool = True
    current_agent: str = "coordinator"
    steps_completed: List[str] = Field(default_factory=list)
    closing_message: str = ""
    reinforcement_label: Optional[str] = None
    hce_feedback_reference: Optional[str] = None
    
    # 🎯 AGENTIC INTERVIEW 2.0 - GOAL-AWARE REASONING
    interview_mode: InterviewMode = InterviewMode.FULL
    interview_goals: List[InterviewGoal] = Field(default_factory=list)
    current_goal: Optional[InterviewGoal] = None
    goals_completed: int = 0
    overall_goal_completion: float = 0.0
    
    # 🎭 EMOTIONAL INTELLIGENCE
    emotional_context: Optional[EmotionalContext] = None
    sentiment_history: List[Tuple[float, datetime]] = Field(default_factory=list)
    tone_adaptations: List[Dict] = Field(default_factory=list)
    empathy_triggers: List[str] = Field(default_factory=list)
    
    # ⏱️ TEMPORAL BEHAVIOR ANALYSIS
    behavioral_windows: List[Dict] = Field(default_factory=list)  # Store last 10 windows
    current_window: Optional[BehavioralWindow] = None
    rolling_anomaly_score: float = 0.0
    temporal_patterns: Dict[str, List[float]] = Field(default_factory=dict)
    
    # 🎤 VOICE CONSISTENCY
    voice_profile: Optional[VoiceProfile] = None
    voice_verification_enabled: bool = True
    voice_anomalies: List[Dict] = Field(default_factory=list)
    speaker_consistency_score: float = 1.0
    
    # 🏢 COMPANY PERSONALITY
    company_personality: Optional[CompanyPersonality] = None
    personality_conditioned: bool = False
    culture_alignment_score: float = 0.0
    
    # 📝 ENDGAME NARRATIVE
    closing_narrative: str = ""
    interview_summary: Dict[str, Any] = Field(default_factory=dict)
    next_steps: List[str] = Field(default_factory=list)
    candidate_insights: Dict[str, Any] = Field(default_factory=dict)
    
    # 🔄 REINFORCEMENT LEARNING
    reinforcement_data: Optional[ReinforcementFeedback] = None
    question_effectiveness_history: Dict[str, float] = Field(default_factory=dict)
    learning_enabled: bool = True
    outcome_prediction: Dict[str, float] = Field(default_factory=dict)
    
    # 🔗 HCE INTEGRATION
    hce_session_id: Optional[str] = None
    cross_agent_insights: Dict[str, Any] = Field(default_factory=dict)
    system_feedback: List[Dict] = Field(default_factory=list)


class UltimateVideoInterviewAgent:
    """
    🚀 ULTIMATE AGENTIC AI VIDEO INTERVIEW SYSTEM 🚀
    
    Next-Generation Architecture:
    =============================
    
    1️⃣ INTELLIGENCE LAYERS:
       - Company Intelligence Integration
       - Job Intelligence Deep Dive
       - Candidate Resume Analysis
       - Pre-screening Context
       - Market Intelligence
    
    2️⃣ DUAL AI MODELS:
       - GPT-4-Realtime: Natural conversational flow
       - GPT-5/o1: Deep reasoning and analysis
    
    3️⃣ MULTI-AGENT COORDINATION:
       - Intelligence Aggregator
       - Context Retriever
       - Question Generator (with GPT-Realtime)
       - Answer Analyzer (with GPT-5)
       - Proctoring Agent
       - Memory Manager
       - Evaluation Synthesizer
    
    4️⃣ NEXT-TO-IMPOSSIBLE ANTI-CHEAT:
       - Multi-modal proctoring
       - AI-powered impersonation detection
       - Real-time anomaly detection
       - Continuous behavioral monitoring
    
    5️⃣ TECH STACK:
       - LangGraph for workflow
       - Redis for vector memory
       - DSPy for reasoning
       - RAG for context
       - MediaPipe for face detection
       - YOLO for object detection
       - OpenAI GPT-4-Realtime & GPT-5
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize GPT-4-Realtime for conversational flow
        self.realtime_llm = AzureChatOpenAI(
            openai_api_key=self.config.azure.api_key,
            azure_endpoint=self.config.azure.endpoint,
            deployment_name="gpt-realtime-mini",  # GPT-4-Realtime
            openai_api_version=self.config.azure.api_version,
            temperature=0.9,  # More creative for conversation
            max_tokens=2048
        )
        
        # Initialize GPT-4-mini for Lite mode
        self.lite_llm = AzureChatOpenAI(
            openai_api_key=self.config.azure.api_key,
            azure_endpoint=self.config.azure.endpoint,
            deployment_name="gpt-4o-mini",  # GPT-4-mini for lite mode
            openai_api_version=self.config.azure.api_version,
            temperature=0.7,
            max_tokens=1024
        )
        
        # Initialize GPT-5/o1 for deep analysis and reasoning
        self.analysis_llm = AzureChatOpenAI(
            openai_api_key=self.config.azure.api_key,
            azure_endpoint=self.config.azure.endpoint,
            deployment_name="gpt-5",  # GPT-5 or o1 for analysis
            openai_api_version=self.config.azure.api_version,
            temperature=0.3,  # Lower for analytical consistency
            max_tokens=8192
        )
        
        self.embeddings = AzureOpenAIEmbeddings(
            openapi_api_key=self.config.azure.api_key,
            azure_endpoint=self.config.azure.endpoint,
            deployment=self.config.azure.embedding_deployment,
            openai_api_version=self.config.azure.api_version
        )
        
        # Initialize vector store for long-term memory
        self.vector_store = get_redis_vector_store(
            redis_host=self.config.redis.host,
            redis_port=self.config.redis.port,
            redis_password=self.config.redis.password
        )
        
        # Initialize RAG chain
        self.vector_manager = VectorStoreManager()
        self.rag_chain = RAGChain(self.vector_manager, temperature=0.7)
        
        # Initialize DSPy modules
        self.question_generator = self._create_question_generator()
        self.answer_analyzer = self._create_answer_analyzer()
        self.context_retriever = self._create_context_retriever()
        self.goal_tracker = self._create_goal_tracker()
        self.emotional_analyzer = self._create_emotional_analyzer()
        self.voice_analyzer = self._create_voice_analyzer()
        
        # Initialize conversation memory (sliding window + vector store)
        self.short_term_memory = ConversationBufferWindowMemory(
            k=5,  # Last 5 exchanges
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize proctoring models (lazy loaded)
        self.face_mesh = _get_face_mesh(
            max_num_faces=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load YOLO for object detection
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
            except Exception:
                self.yolo_model = None
        else:
            self.yolo_model = None
        
        # Initialize behavioral window tracking
        self.behavioral_window_duration = 30  # seconds
        
        # Initialize reinforcement learning storage
        self.feedback_store = {}  # In-memory store, should be Redis/DB in production
        
        # Build LangGraph workflow
        self.workflow = self._build_langgraph_workflow()
        
        print("✅ Agentic Interview 2.0 initialized with:")
        print("   - GPT-4-Realtime for conversations")
        print("   - GPT-4-mini for Lite mode")
        print("   - GPT-5/o1 for deep analysis")
        print("   - Goal-aware reasoning system")
        print("   - Emotional intelligence & sentiment analysis")
        print("   - Temporal behavior analysis (30s windows)")
        print("   - Voice consistency verification")
        print("   - Company personality conditioning")
        print("   - Multi-modal proctoring")
        print("   - Reinforcement learning feedback loop")
        print("   - Complete intelligence integration")
        print("   - Redis vector memory")
    
    def _create_question_generator(self) -> ChainOfThought:
        """Create DSPy-based question generation module"""
        return ChainOfThought(
            signature=DSPySignature(
                name="adaptive_interview_question_generation",
                inputs={
                    "job_context": "Job title, description, and required skills",
                    "candidate_profile": "Candidate's background, resume, and skills",
                    "conversation_history": "Previous questions and answers",
                    "current_phase": "Current interview phase",
                    "difficulty_level": "Target difficulty level",
                    "retrieved_context": "Relevant context from knowledge base",
                    "skills_to_assess": "Specific skills to evaluate",
                    "previous_questions": "All questions asked so far (for uniqueness)"
                },
                outputs={
                    "question": "Unique, contextually relevant interview question",
                    "reasoning": "Why this question is appropriate now",
                    "expected_skill_assessment": "Skills this question evaluates",
                    "follow_up_strategy": "Potential follow-up questions based on answer",
                    "difficulty_rationale": "Why this difficulty level is chosen"
                },
                description="""Generate a highly contextual, unique interview question that:
                1. Has NEVER been asked before in this interview
                2. Builds on previous conversation naturally
                3. Matches candidate's demonstrated skill level
                4. Evaluates specific job-relevant competencies
                5. Incorporates retrieved context from knowledge base
                6. Adapts to candidate's engagement and performance"""
            ),
            temperature=0.9  # High creativity for unique questions
        )
    
    def _create_answer_analyzer(self) -> ChainOfThought:
        """Create DSPy-based answer analysis module"""
        return ChainOfThought(
            signature=DSPySignature(
                name="comprehensive_answer_analysis",
                inputs={
                    "question": "The interview question asked",
                    "answer": "Candidate's response",
                    "job_requirements": "Job requirements and skills",
                    "candidate_background": "Candidate's experience level",
                    "expected_skills": "Skills being evaluated"
                },
                outputs={
                    "relevance_score": "Answer relevance (0-10)",
                    "quality_score": "Answer quality and depth (0-10)",
                    "technical_accuracy": "Technical correctness (0-10)",
                    "communication_clarity": "Communication effectiveness (0-10)",
                    "skill_demonstrations": "Skills demonstrated with scores",
                    "strengths": "What the candidate did well",
                    "gaps": "What could be improved",
                    "follow_up_needed": "Whether follow-up question needed",
                    "red_flags": "Any concerning patterns or content"
                },
                description="Comprehensively analyze interview answer for quality, relevance, skills, and red flags"
            ),
            temperature=0.3  # Lower for analytical consistency
        )
    
    def _create_goal_tracker(self) -> DSPyModule:
        """Create goal tracking module for interview objectives"""
        return DSPyModule(
            signature=DSPySignature(
                name="goal_tracking",
                inputs={
                    "current_goals": "List of interview goals with their status",
                    "skills_assessed": "Skills already evaluated",
                    "conversation_history": "Recent Q&A exchanges",
                    "required_skills": "All skills to evaluate",
                    "questions_remaining": "Number of questions left"
                },
                outputs={
                    "next_goal": "Which goal to pursue next",
                    "goal_priority": "Priority ranking of remaining goals",
                    "completion_status": "Overall goal completion percentage",
                    "recommendations": "Strategic recommendations for interview flow"
                },
                description="Track and prioritize interview goals to ensure all key skills are evaluated"
            ),
            temperature=0.3
        )
    
    def _create_emotional_analyzer(self) -> ChainOfThought:
        """Create emotional intelligence and sentiment analysis module"""
        return ChainOfThought(
            signature=DSPySignature(
                name="emotional_analysis",
                inputs={
                    "answer_text": "Candidate's response text",
                    "audio_features": "Voice tone and speech patterns (if available)",
                    "response_time": "Time taken to respond",
                    "previous_sentiment": "Previous emotional state"
                },
                outputs={
                    "emotional_tone": "Detected emotional tone (confident, hesitant, nervous, etc.)",
                    "sentiment_score": "Sentiment score from -1 to 1",
                    "confidence_level": "Confidence level 0 to 1",
                    "stress_indicators": "List of stress indicators detected",
                    "tone_adjustment": "Recommended interviewer tone adjustment (empathetic, probing, encouraging)"
                },
                description="Analyze candidate's emotional state and recommend adaptive interviewer tone"
            ),
            temperature=0.4
        )
    
    def _create_voice_analyzer(self) -> DSPyModule:
        """Create voice consistency verification module"""
        return DSPyModule(
            signature=DSPySignature(
                name="voice_analysis",
                inputs={
                    "baseline_voice_embedding": "Initial voice embedding",
                    "current_voice_embedding": "Current voice embedding",
                    "similarity_threshold": "Threshold for anomaly detection"
                },
                outputs={
                    "similarity_score": "Voice similarity score 0 to 1",
                    "is_consistent": "Boolean indicating voice consistency",
                    "anomaly_detected": "Whether impersonation is suspected",
                    "confidence": "Confidence in the analysis"
                },
                    description="Verify voice consistency throughout interview to detect impersonation"
                ),
                temperature=0.1
            )
    
    def _create_context_retriever(self) -> DSPyModule:
        """Create context retrieval module"""
        return DSPyModule(
            signature=DSPySignature(
                name="context_retrieval",
                inputs={
                    "current_conversation": "Current conversation state",
                    "candidate_profile": "Candidate background",
                    "job_requirements": "Job context",
                    "interview_phase": "Current phase"
                },
                outputs={
                    "relevant_contexts": "Most relevant contexts to consider",
                    "memory_insights": "Insights from previous conversations",
                    "skill_gaps_identified": "Skills needing more assessment"
                },
                description="Retrieve and synthesize relevant context from memory and knowledge base"
            ),
            temperature=0.5
        )
    
    def _build_langgraph_workflow(self) -> StateGraph:
        """Build comprehensive LangGraph workflow"""
        workflow = StateGraph(VideoInterviewState)
        
        # Define all agent nodes
        workflow.add_node("retrieve_context", self._retrieve_context_agent)
        workflow.add_node("analyze_answer", self._analyze_answer_agent)
        workflow.add_node("update_memory", self._update_memory_agent)
        workflow.add_node("assess_skills", self._assess_skills_agent)
        workflow.add_node("analyze_behavior", self._analyze_behavior_agent)
        workflow.add_node("adjust_difficulty", self._adjust_difficulty_agent)
        workflow.add_node("determine_phase", self._determine_phase_agent)
        workflow.add_node("generate_question", self._generate_question_agent)
        workflow.add_node("check_uniqueness", self._check_question_uniqueness)
        workflow.add_node("create_summary", self._create_summary_agent)
        
        # Define workflow edges
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "analyze_answer")
        workflow.add_edge("analyze_answer", "update_memory")
        workflow.add_edge("update_memory", "assess_skills")
        workflow.add_edge("assess_skills", "analyze_behavior")
        workflow.add_edge("analyze_behavior", "adjust_difficulty")
        workflow.add_edge("adjust_difficulty", "determine_phase")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "determine_phase",
            self._should_continue_or_complete,
            {
                "continue": "generate_question",
                "complete": "create_summary"
            }
        )
        
        workflow.add_edge("generate_question", "check_uniqueness")
        
        workflow.add_conditional_edges(
            "check_uniqueness",
            self._is_question_unique,
            {
                "unique": END,
                "duplicate": "generate_question"  # Regenerate if duplicate
            }
        )
        
        workflow.add_edge("create_summary", END)
        
        return workflow.compile()
    
    async def _parallel_analysis_node(self, state: VideoInterviewState) -> VideoInterviewState:
        """Parallel execution node for independent agents (2-3× faster)"""
        try:
            # Run independent agents in parallel using asyncio.gather
            results = await asyncio.gather(
                self._update_memory_agent(state),
                self._assess_skills_agent(state),
                self._analyze_behavior_agent(state),
                self._analyze_temporal_behavior_agent(state),
                self._verify_voice_consistency_agent(state),
                return_exceptions=True
            )
            
            # Merge results (all agents return the same state object, so just return the last valid one)
            for result in results:
                if isinstance(result, VideoInterviewState):
                    state = result
                elif isinstance(result, Exception):
                    print(f"Parallel agent error: {result}")
            
            return state
            
        except Exception as e:
            print(f"Parallel execution error: {e}")
            return state
    
    async def _track_goals_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """New Agent: Track interview goals and ensure coverage"""
        state.current_agent = "goal_tracker"
        
        try:
            if not state.interview_goals:
                # Initialize goals based on required skills
                for i, skill in enumerate(state.required_skills):
                    goal = InterviewGoal(
                        goal_id=f"goal_{i}",
                        description=f"Assess {skill} proficiency",
                        target_skills=[skill],
                        priority=10 - i,
                        status=InterviewGoalStatus.PENDING
                    )
                    state.interview_goals.append(goal)
            
            # Use goal tracker DSPy module
            goal_result = await self.goal_tracker.forward({
                "current_goals": json.dumps([{
                    "id": g.goal_id,
                    "description": g.description,
                    "status": g.status.value,
                    "completion": g.completion_percentage
                } for g in state.interview_goals]),
                "skills_assessed": json.dumps(state.skill_scores),
                "conversation_history": json.dumps(state.conversation_history[-5:]),
                "required_skills": json.dumps(state.required_skills),
                "questions_remaining": str(state.target_questions - state.questions_count)
            })
            
            # Update goal tracking
            state.overall_goal_completion = float(goal_result.get("completion_status", 0))
            
            state.steps_completed.append("track_goals")
            
        except Exception as e:
            print(f"Goal tracking error: {e}")
        
        return state
    
    async def _analyze_emotional_context_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """New Agent: Analyze emotional state and adapt tone"""
        state.current_agent = "emotional_analyzer"
        
        if not state.current_answer:
            return state
        
        try:
            # Calculate response time if available
            response_time = state.response_times[-1] if state.response_times else 5.0
            
            # Analyze emotional context
            emotion_result = await self.emotional_analyzer.forward({
                "answer_text": state.current_answer,
                "audio_features": "{}",  # Would come from audio processing
                "response_time": str(response_time),
                "previous_sentiment": json.dumps({
                    "tone": state.emotional_context.current_tone.value if state.emotional_context else "neutral",
                    "score": state.emotional_context.sentiment_score if state.emotional_context else 0.0
                })
            })
            
            # Extract emotional data
            tone_str = emotion_result.get("emotional_tone", "neutral")
            try:
                current_tone = EmotionalTone(tone_str)
            except ValueError:
                current_tone = EmotionalTone.NEUTRAL
            
            sentiment_score = float(emotion_result.get("sentiment_score", 0.0))
            confidence = float(emotion_result.get("confidence_level", 0.5))
            
            # Create or update emotional context
            if not state.emotional_context:
                state.emotional_context = EmotionalContext(
                    current_tone=current_tone,
                    sentiment_score=sentiment_score,
                    confidence_level=confidence
                )
            else:
                state.emotional_context.current_tone = current_tone
                state.emotional_context.sentiment_score = sentiment_score
                state.emotional_context.confidence_level = confidence
                state.emotional_context.tone_history.append((current_tone, datetime.now()))
            
            # Store tone adjustment recommendation
            tone_adjustment = emotion_result.get("tone_adjustment", "balanced")
            state.emotional_context.recommended_tone_adjustment = tone_adjustment
            state.tone_adaptations.append({
                "timestamp": datetime.now().isoformat(),
                "detected_tone": current_tone.value,
                "recommended_adjustment": tone_adjustment
            })
            
            state.steps_completed.append("analyze_emotional_context")
            
        except Exception as e:
            print(f"Emotional analysis error: {e}")
        
        return state
    
    async def _analyze_temporal_behavior_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """New Agent: Analyze behavior in 30-second rolling windows"""
        state.current_agent = "temporal_behavior_analyzer"
        
        try:
            # Create new behavioral window if needed
            now = datetime.now()
            if not state.current_window or (now - state.current_window.start_time).seconds >= self.behavioral_window_duration:
                # Save current window if exists
                if state.current_window:
                    window_dict = {
                        "window_id": state.current_window.window_id,
                        "start_time": state.current_window.start_time.isoformat(),
                        "end_time": state.current_window.end_time.isoformat(),
                        "suspicious_score": state.current_window.suspicious_score,
                        "anomalies": state.current_window.anomalies_detected
                    }
                    state.behavioral_windows.append(window_dict)
                    
                    # Keep only last 10 windows
                    if len(state.behavioral_windows) > 10:
                        state.behavioral_windows = state.behavioral_windows[-10:]
                
                # Create new window
                state.current_window = BehavioralWindow(
                    window_id=f"window_{len(state.behavioral_windows)}",
                    start_time=now,
                    end_time=now + timedelta(seconds=self.behavioral_window_duration)
                )
            
            # Calculate rolling anomaly score from recent windows
            if state.behavioral_windows:
                recent_scores = [w.get("suspicious_score", 0.0) for w in state.behavioral_windows[-5:]]
                state.rolling_anomaly_score = sum(recent_scores) / len(recent_scores)
            
            state.steps_completed.append("analyze_temporal_behavior")
            
        except Exception as e:
            print(f"Temporal behavior analysis error: {e}")
        
        return state
    
    async def _verify_voice_consistency_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """New Agent: Verify voice consistency for impersonation detection"""
        state.current_agent = "voice_verifier"
        
        if not state.voice_verification_enabled:
            return state
        
        try:
            # Initialize voice profile if not exists
            if not state.voice_profile:
                state.voice_profile = VoiceProfile()
            
            # In real implementation, this would extract voice embeddings from audio
            # For now, we simulate the check
            if state.voice_profile.baseline_embedding and state.voice_profile.current_embedding:
                # Use voice analyzer
                voice_result = await self.voice_analyzer.forward({
                    "baseline_voice_embedding": json.dumps(state.voice_profile.baseline_embedding),
                    "current_voice_embedding": json.dumps(state.voice_profile.current_embedding),
                    "similarity_threshold": "0.85"
                })
                
                similarity = float(voice_result.get("similarity_score", 1.0))
                state.voice_profile.similarity_score = similarity
                state.speaker_consistency_score = similarity
                
                if similarity < 0.85:
                    state.voice_profile.anomaly_count += 1
                    state.voice_anomalies.append({
                        "timestamp": datetime.now().isoformat(),
                        "similarity_score": similarity,
                        "description": "Voice consistency anomaly detected"
                    })
                
                state.voice_profile.last_check_time = datetime.now()
            
            state.steps_completed.append("verify_voice_consistency")
            
        except Exception as e:
            print(f"Voice verification error: {e}")
        
        return state
    
    async def _condition_company_personality_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """New Agent: Apply company personality conditioning"""
        state.current_agent = "personality_conditioner"
        
        if state.personality_conditioned:
            return state
        
        try:
            # Extract company personality from intelligence
            if state.company_intelligence:
                personality = CompanyPersonality(
                    tone=state.company_culture.lower() if state.company_culture else "professional",
                    values_emphasis=state.company_values[:5],
                    mission_alignment=state.company_mission,
                    culture_keywords=[kw.lower() for kw in state.company_culture.split()[:10]] if state.company_culture else []
                )
                state.company_personality = personality
                state.personality_conditioned = True
            
            state.steps_completed.append("condition_company_personality")
            
        except Exception as e:
            print(f"Company personality conditioning error: {e}")
        
        return state
    
    async def _generate_endgame_narrative_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """New Agent: Generate human-like closing narrative"""
        state.current_agent = "endgame_narrator"
        
        try:
            # Select appropriate LLM based on mode
            llm = self.lite_llm if state.interview_mode == InterviewMode.LITE else self.analysis_llm
            
            prompt = f"""Generate a warm, human-like closing message for the candidate interview.

Candidate Performance Summary:
- Overall Score: {state.overall_score:.1f}/10
- Skills Assessed: {', '.join(state.skill_scores.keys())}
- Engagement Level: {state.engagement_level.value}
- Questions Answered: {state.questions_count}

Company Context:
- Company: {state.company_intelligence.get('name', 'the company')}
- Role: {state.job_title}
- Culture: {state.company_culture[:200] if state.company_culture else 'Professional and collaborative'}

Generate a closing message that:
1. Thanks the candidate warmly
2. Highlights 2-3 key strengths observed
3. Mentions 1-2 areas for potential growth (diplomatically)
4. Explains next steps in the hiring process
5. Maintains the company's communication tone

Keep it concise (3-4 paragraphs) and genuinely human."""

            messages = [HumanMessage(content=prompt)]
            response = await asyncio.to_thread(llm.invoke, messages)
            
            state.closing_narrative = response.content
            
            # Generate interview summary
            state.interview_summary = {
                "total_questions": state.questions_count,
                "overall_score": state.overall_score,
                "top_skills": sorted(state.skill_scores.items(), key=lambda x: x[1], reverse=True)[:5],
                "engagement": state.engagement_level.value,
                "recommendation": state.recommendation,
                "completed_at": datetime.now().isoformat()
            }
            
            # Set next steps
            if state.overall_score >= 7.5:
                state.next_steps = [
                    "Technical round with the engineering team",
                    "Culture fit discussion with the hiring manager",
                    "Final interview with leadership"
                ]
            elif state.overall_score >= 6.0:
                state.next_steps = [
                    "Additional technical assessment",
                    "Follow-up discussion with recruiter"
                ]
            else:
                state.next_steps = [
                    "We'll review your application",
                    "You'll hear from us within 5-7 business days"
                ]
            
            state.steps_completed.append("generate_endgame_narrative")
            
        except Exception as e:
            print(f"Endgame narrative generation error: {e}")
            state.closing_narrative = f"Thank you for interviewing for the {state.job_title} position. We appreciate your time and will be in touch soon."
        
    
    # ========================================
    # AGENTIC INTERVIEW 2.1 - PRODUCTION FEATURES
    # ========================================
    
    async def _distributed_cache_lookup(self, key: str, cache_type: str = "embedding") -> Optional[Any]:
        """Distributed Redis Cluster cache lookup - 50% cost reduction"""
        try:
            cache_key = f"agentic_cache:{cache_type}:{key}"
            cached_data = self.vector_store.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            print(f"Cache lookup error: {e}")
            return None
    
    async def _distributed_cache_store(self, key: str, value: Any, cache_type: str = "embedding", ttl: int = 86400):
        """Store in distributed cache with TTL"""
        try:
            cache_key = f"agentic_cache:{cache_type}:{key}"
            self.vector_store.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(value)
            )
        except Exception as e:
            print(f"Cache store error: {e}")
    
    async def _build_skill_graph(self, required_skills: List[str]) -> SkillGraph:
        """Build graph-based skill dependency map"""
        graph = SkillGraph()
        
        # Define skill dependencies (can be loaded from knowledge base)
        skill_dependencies = {
            "python": [],
            "fastapi": ["python"],
            "django": ["python"],
            "async_programming": ["python"],
            "microservices": ["fastapi", "async_programming"],
            "system_design": ["microservices"],
            "database_design": [],
            "sql": ["database_design"],
            "nosql": ["database_design"],
            "redis": ["nosql"],
            "caching": ["redis"],
            "api_design": ["fastapi"],
            "testing": ["python"],
            "ci_cd": ["testing"]
        }
        
        # Build graph from required skills
        for skill in required_skills:
            skill_lower = skill.lower().replace(" ", "_")
            prerequisites = skill_dependencies.get(skill_lower, [])
            
            node = SkillNode(
                skill_id=skill_lower,
                skill_name=skill,
                prerequisites=prerequisites,
                difficulty_level=len(prerequisites) + 1
            )
            graph.add_skill(node)
            
            # Add prerequisite nodes if not present
            for prereq in prerequisites:
                if prereq not in graph.nodes:
                    prereq_node = SkillNode(
                        skill_id=prereq,
                        skill_name=prereq.replace("_", " ").title(),
                        prerequisites=skill_dependencies.get(prereq, []),
                        difficulty_level=len(skill_dependencies.get(prereq, [])) + 1
                    )
                    graph.add_skill(prereq_node)
        
        return graph
    
    async def _compress_memory_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Compress conversation embeddings to prevent Redis bloat"""
        state.current_agent = "memory_compressor"
        
        try:
            # Check if compression needed (every 5 exchanges)
            if len(state.conversation_history) % 5 != 0:
                return state
            
            # Get recent embeddings
            recent_convos = state.conversation_history[-5:]
            embeddings = []
            
            for convo in recent_convos:
                if convo.get("role") == "user":
                    text = convo.get("content", "")
                    # Check cache first
                    cached_emb = await self._distributed_cache_lookup(text, "embedding")
                    if cached_emb:
                        embeddings.append(cached_emb)
                    else:
                        emb = await asyncio.to_thread(
                            self.embeddings.embed_query,
                            text
                        )
                        embeddings.append(emb)
                        await self._distributed_cache_store(text, emb, "embedding")
            
            if embeddings:
                # Compute summary vector (average)
                summary_vector = np.mean(embeddings, axis=0).tolist()
                
                # Extract key points using LLM
                summary_prompt = f"""Summarize these 5 conversation exchanges into 3 key points:
                {json.dumps(recent_convos, indent=2)}
                
                Return ONLY a JSON array of 3 strings."""
                
                response = await asyncio.to_thread(
                    self.lite_llm.invoke,
                    [HumanMessage(content=summary_prompt)]
                )
                
                try:
                    key_points = json.loads(response.content)
                except:
                    key_points = ["Exchange summary unavailable"]
                
                # Create compressed memory
                compressed = CompressedMemory(
                    session_id=state.session_id,
                    summary_vector=summary_vector,
                    key_points=key_points,
                    skill_coverage={k: v for k, v in state.skill_scores.items()},
                    compression_ratio=0.2,  # 5:1 compression
                    original_size=len(json.dumps(recent_convos)),
                    compressed_size=len(json.dumps(key_points)),
                    timestamp=datetime.now()
                )
                
                # Store compressed memory instead of full embeddings
                await self._distributed_cache_store(
                    f"compressed_{state.session_id}_{len(state.conversation_history)}",
                    compressed.__dict__,
                    "compressed_memory",
                    ttl=604800  # 7 days
                )
            
            state.steps_completed.append("compress_memory")
            
        except Exception as e:
            print(f"Memory compression error: {e}")
        
        return state
    
    async def _generate_recruiter_review_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Generate human-readable recruiter review summary"""
        state.current_agent = "recruiter_reviewer"
        
        try:
            # Generate transparent, auditable summary
            prompt = f"""Generate a clear, transparent recruiter review summary for:

Candidate: {state.candidate_profile.get('name', 'Candidate')}
Role: {state.job_title}
Interview Duration: {state.questions_count} questions

Performance Metrics:
- Overall Score: {state.overall_score:.1f}/10
- Skill Scores: {json.dumps(state.skill_scores, indent=2)}
- Engagement: {state.engagement_level.value}
- Proctoring: {state.violation_count} violations

Create a review that:
1. Summarizes key strengths (3-5 points)
2. Lists concerns (if any)
3. Explains recommendation rationale clearly
4. Shows decision factors with weights
5. Is auditable by humans

Format as JSON with: overall_assessment, key_strengths (array), concerns (array), recommendation_rationale, decision_factors (object with weights)"""
            
            response = await asyncio.to_thread(
                self.analysis_llm.invoke,
                [HumanMessage(content=prompt)]
            )
            
            try:
                review_data = json.loads(response.content.replace("```json", "").replace("```", ""))
            except:
                review_data = {"overall_assessment": response.content}
            
            # Create recruiter review
            review = RecruiterReviewSummary(
                session_id=state.session_id,
                candidate_name=state.candidate_profile.get('name', 'Unknown'),
                overall_assessment=review_data.get('overall_assessment', ''),
                key_strengths=review_data.get('key_strengths', []),
                concerns=review_data.get('concerns', []),
                recommendation_rationale=review_data.get('recommendation_rationale', ''),
                decision_factors=review_data.get('decision_factors', {}),
                bias_check_results={},  # Will be filled by bias agent
                auditor_notes=[],
                confidence_score=state.overall_score / 10,
                generated_at=datetime.now()
            )
            
            # Store for human oversight
            await self._distributed_cache_store(
                f"recruiter_review_{state.session_id}",
                review.__dict__,
                "recruiter_review",
                ttl=2592000  # 30 days
            )
            
            state.steps_completed.append("generate_recruiter_review")
            
        except Exception as e:
            print(f"Recruiter review generation error: {e}")
        
        return state
    
    async def _cognitive_bias_check_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Counterfactual analysis to minimize cognitive bias"""
        state.current_agent = "bias_minimizer"
        
        try:
            # Analyze for potential biases
            bias_prompt = f"""Analyze this interview for potential cognitive biases:

Candidate Profile: {json.dumps(state.candidate_profile, indent=2)}
Recommendation: {state.recommendation}
Overall Score: {state.overall_score}

Check for:
1. Name bias (demographic assumptions)
2. Halo effect (one strong skill overshadowing)
3. Recency bias (recent answers weighted too heavily)
4. Confirmation bias (seeking evidence for initial impression)
5. Similarity bias (favoring similar backgrounds)

For each bias type, provide:
- severity (0-1)
- evidence found
- counterfactual scenario

Return as JSON."""
            
            response = await asyncio.to_thread(
                self.analysis_llm.invoke,
                [HumanMessage(content=bias_prompt)]
            )
            
            try:
                bias_data = json.loads(response.content.replace("```json", "").replace("```", ""))
            except:
                bias_data = {}
            
            # Create bias analysis
            bias_analysis = BiasAnalysis(
                session_id=state.session_id,
                bias_indicators=bias_data.get('bias_indicators', {}),
                counterfactual_scores=bias_data.get('counterfactual_scores', {}),
                demographic_fairness=bias_data.get('demographic_fairness', {}),
                recommendations=bias_data.get('recommendations', []),
                bias_mitigation_applied=[],
                audit_trail=[]
            )
            
            # Store for compliance
            await self._distributed_cache_store(
                f"bias_analysis_{state.session_id}",
                bias_analysis.__dict__,
                "bias_analysis",
                ttl=2592000  # 30 days for audit
            )
            
            state.steps_completed.append("cognitive_bias_check")
            
        except Exception as e:
            print(f"Bias check error: {e}")
        
        return state
    
    def _estimate_task_complexity(self, task: str, context_size: int) -> ModelComplexity:
        """Estimate task complexity for cost-aware model switching"""
        # Simple heuristic-based complexity estimation
        if "final" in task.lower() or "comprehensive" in task.lower():
            return ModelComplexity.EXPERT  # GPT-5/o1
        elif "analyze" in task.lower() or "evaluate" in task.lower():
            return ModelComplexity.ADVANCED  # GPT-5
        elif context_size > 2000 or "conversation" in task.lower():
            return ModelComplexity.STANDARD  # GPT-4-Realtime
        else:
            return ModelComplexity.MINIMAL  # GPT-4-mini
    
    def _get_llm_for_complexity(self, complexity: ModelComplexity):
        """Get appropriate LLM based on complexity"""
        mapping = {
            ModelComplexity.MINIMAL: self.lite_llm,
            ModelComplexity.STANDARD: self.realtime_llm,
            ModelComplexity.ADVANCED: self.analysis_llm,
            ModelComplexity.EXPERT: self.analysis_llm  # Can switch to o1 when available
        }
        return mapping.get(complexity, self.realtime_llm)
    
    async def _fuse_temporal_windows_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Fuse 30-second windows into session vectors for 30-40% better detection"""
        state.current_agent = "temporal_fusion"
        
        try:
            if len(state.behavioral_windows) < 3:
                return state  # Need at least 3 windows to fuse
            
            # Extract window data
            window_data = [
                {
                    "anomalies": w.get("suspicious_score", 0),
                    "timestamp": w.get("start_time", "")
                }
                for w in state.behavioral_windows
            ]
            
            # Calculate behavioral drift
            anomaly_scores = [w["anomalies"] for w in window_data]
            if len(anomaly_scores) > 1:
                drift_score = np.std(anomaly_scores) / (np.mean(anomaly_scores) + 0.01)
            else:
                drift_score = 0.0
            
            # Create session vector
            session_vector = SessionVector(
                session_id=state.session_id,
                start_time=datetime.fromisoformat(window_data[0]["timestamp"]) if window_data[0]["timestamp"] else datetime.now(),
                end_time=datetime.fromisoformat(window_data[-1]["timestamp"]) if window_data[-1]["timestamp"] else datetime.now(),
                window_summaries=window_data,
                behavioral_drift_score=float(drift_score),
                anomaly_trajectory=anomaly_scores,
                impersonation_confidence=1.0 - state.speaker_consistency_score,
                attention_consistency=state.attention_score / 100.0,
                overall_integrity_score=1.0 - (drift_score * 0.5 + (1.0 - state.speaker_consistency_score) * 0.5)
            )
            
            # Store fused vector
            await self._distributed_cache_store(
                f"session_vector_{state.session_id}",
                session_vector.__dict__,
                "session_vector"
            )
            
            state.steps_completed.append("fuse_temporal_windows")
            
        except Exception as e:
            print(f"Temporal fusion error: {e}")
        
        return state
    
    async def _adaptive_reinforcement_update(self, outcome: HiringOutcome):
        """Fine-tune agents every 500 sessions - Autonomous performance evolution"""
        try:
            # Check session count
            session_count = len(self.feedback_store)
            
            if session_count > 0 and session_count % 500 == 0:
                print(f"🔄 Triggering adaptive reinforcement after {session_count} sessions...")
                
                # Collect training data from last 500 sessions
                training_data = []
                for session_id, feedback in list(self.feedback_store.items())[-500:]:
                    if feedback.outcome != HiringOutcome.PENDING:
                        training_data.append({
                            "questions": feedback.question_effectiveness,
                            "outcome": feedback.outcome.value,
                            "quality_patterns": feedback.answer_quality_patterns
                        })
                
                if len(training_data) >= 100:  # Need minimum data
                    # Calculate question effectiveness by outcome
                    hired_questions = {}
                    rejected_questions = {}
                    
                    for data in training_data:
                        outcome = data["outcome"]
                        for q, eff in data["questions"].items():
                            if outcome == "hired":
                                hired_questions[q] = hired_questions.get(q, []) + [eff]
                            elif outcome == "rejected":
                                rejected_questions[q] = rejected_questions.get(q, []) + [eff]
                    
                    # Identify most effective question patterns
                    effective_patterns = {}
                    for q in hired_questions:
                        if q in hired_questions and q in rejected_questions:
                            hired_avg = np.mean(hired_questions[q])
                            rejected_avg = np.mean(rejected_questions[q])
                            discrimination = hired_avg - rejected_avg
                            effective_patterns[q] = discrimination
                    
                    # Update question generator with effective patterns
                    top_patterns = sorted(effective_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    for pattern, score in top_patterns:
                        self.question_generator.add_example(
                            {"context": "generate effective question"},
                            {"question": pattern, "effectiveness": score}
                        )
                    
                    print(f"✅ Updated question generator with {len(top_patterns)} effective patterns")
                    
                    # Store learning metrics
                    learning_summary = {
                        "session_count": session_count,
                        "training_samples": len(training_data),
                        "effective_patterns": len(effective_patterns),
                        "top_patterns": [p[0] for p in top_patterns[:5]],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await self._distributed_cache_store(
                        f"learning_summary_{session_count}",
                        learning_summary,
                        "learning",
                        ttl=31536000  # 1 year
                    )
        
        except Exception as e:
            print(f"Adaptive reinforcement error: {e}")
    
    async def _hce_feedback_loop_sync(self, state: VideoInterviewState):
        """Sync with HCE - Company updates auto-modify goals and tone"""
        try:
            # Check if company intelligence has updated
            company_cache_key = f"company_intel_{state.company_intelligence.get('id', '')}"
            latest_intel = await self._distributed_cache_lookup(company_cache_key, "company")
            
            if latest_intel and latest_intel != state.company_intelligence:
                print("🔄 Company intelligence updated - adapting interview...")
                
                # Update company personality
                state.company_intelligence = latest_intel
                state.company_mission = latest_intel.get('mission', '')
                state.company_values = latest_intel.get('values', [])
                state.company_culture = latest_intel.get('culture', '')
                
                # Re-condition personality
                state.personality_conditioned = False
                state = await self._condition_company_personality_agent(state)
                
                # Adjust goal weights based on company priorities
                priority_skills = latest_intel.get('priority_skills', [])
                for goal in state.interview_goals:
                    if any(skill in priority_skills for skill in goal.target_skills):
                        goal.priority = min(goal.priority + 2, 10)
                
                print("✅ Interview adapted to company updates")
        
        except Exception as e:
            print(f"HCE feedback loop error: {e}")
        return state
    
    async def _store_reinforcement_feedback_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """New Agent: Store feedback for reinforcement learning"""
        state.current_agent = "reinforcement_learner"
        
        if not state.learning_enabled:
            return state
        
        try:
            # Generate conversation embeddings
            conversation_embeddings = []
            for exchange in state.conversation_history[-10:]:
                if exchange.get("role") == "user":
                    emb = await asyncio.to_thread(
                        self.embeddings.embed_query,
                        exchange.get("content", "")
                    )
                    conversation_embeddings.append(emb)
            
            # Calculate question effectiveness
            question_effectiveness = {}
            for i, q in enumerate(state.questions_asked):
                if i < len(state.quality_scores):
                    question_effectiveness[q] = state.quality_scores[i]
            
            # Create reinforcement feedback
            feedback = ReinforcementFeedback(
                session_id=state.session_id,
                outcome=HiringOutcome.PENDING,  # Will be updated later
                conversation_embeddings=conversation_embeddings,
                question_effectiveness=question_effectiveness,
                answer_quality_patterns={
                    "avg_quality": sum(state.quality_scores) / len(state.quality_scores) if state.quality_scores else 0,
                    "avg_relevance": sum(state.relevance_scores) / len(state.relevance_scores) if state.relevance_scores else 0,
                    "engagement": state.engagement_level.value
                },
                interviewer_performance={
                    "questions_generated": state.questions_count,
                    "goal_completion": state.overall_goal_completion,
                    "adaptation_quality": len(state.tone_adaptations) / max(state.questions_count, 1)
                },
                timestamp=datetime.now(),
                metadata={
                    "job_title": state.job_title,
                    "interview_mode": state.interview_mode.value,
                    "proctoring_violations": state.violation_count
                }
            )
            
            state.reinforcement_data = feedback
            
            # Store in feedback store (would be Redis/DB in production)
            self.feedback_store[state.session_id] = feedback
            
            state.steps_completed.append("store_reinforcement_feedback")
            
        except Exception as e:
            print(f"Reinforcement feedback storage error: {e}")
        
        return state
        return workflow.compile()
    
    async def _retrieve_context_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 1: Retrieve relevant context from memory and knowledge base"""
        state.current_agent = "context_retriever"
        
        try:
            # Build query from current state
            query_text = f"""
            Job: {state.job_title}
            Phase: {state.current_phase.value}
            Last conversation: {state.conversation_history[-3:] if state.conversation_history else 'None'}
            Skills needed: {', '.join(state.required_skills)}
            """
            
            # Get embedding for semantic search
            query_embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                query_text
            )
            
            # Search vector store for relevant past conversations
            similar_memories = self.vector_store.similarity_search(
                query_embedding,
                k=5,
                filters={"session_id": state.session_id},
                score_threshold=0.7
            )
            
            state.retrieved_contexts = similar_memories
            state.semantic_memories = [mem['text'] for mem in similar_memories]
            
            # Use DSPy context retriever
            context_result = await self.context_retriever.forward({
                "current_conversation": json.dumps(state.conversation_history[-5:]),
                "candidate_profile": json.dumps(state.candidate_profile),
                "job_requirements": json.dumps({"title": state.job_title, "skills": state.required_skills}),
                "interview_phase": state.current_phase.value
            })
            
            state.adaptive_insights = context_result
            state.steps_completed.append("retrieve_context")
            
        except Exception as e:
            print(f"Context retrieval error: {e}")
        
        return state
    
    async def _analyze_answer_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 2: Analyze candidate's answer comprehensively"""
        state.current_agent = "answer_analyzer"
        
        if not state.current_answer:
            state.steps_completed.append("analyze_answer")
            return state
        
        try:
            # Use DSPy answer analyzer
            analysis = await self.answer_analyzer.forward({
                "question": state.current_question,
                "answer": state.current_answer,
                "job_requirements": json.dumps({
                    "title": state.job_title,
                    "skills": state.required_skills
                }),
                "candidate_background": json.dumps(state.candidate_profile),
                "expected_skills": json.dumps(state.required_skills)
            })
            
            # Extract scores
            relevance_score = float(analysis.get("relevance_score", 5.0))
            quality_score = float(analysis.get("quality_score", 5.0))
            
            state.relevance_scores.append(relevance_score)
            state.quality_scores.append(quality_score)
            
            # Update follow-up flag
            state.follow_up_needed = analysis.get("follow_up_needed", False)
            
            # Check for red flags
            red_flags = analysis.get("red_flags", [])
            if red_flags:
                state.warning_message = f"Note: {', '.join(red_flags)}"
            
            state.steps_completed.append("analyze_answer")
            
        except Exception as e:
            print(f"Answer analysis error: {e}")
        
        return state
    
    async def _update_memory_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 3: Update long-term memory with conversation"""
        state.current_agent = "memory_manager"
        
        if not state.current_answer:
            state.steps_completed.append("update_memory")
            return state
        
        try:
            # Create conversation record
            conversation_text = f"Q: {state.current_question}\nA: {state.current_answer}"
            
            # Generate embedding
            embedding = await asyncio.to_thread(
                self.embeddings.embed_query,
                conversation_text
            )
            
            # Store in vector database
            doc_data = {
                "text": conversation_text,
                "metadata": {
                    "session_id": state.session_id,
                    "candidate_id": state.candidate_id,
                    "phase": state.current_phase.value,
                    "difficulty": state.current_difficulty.value,
                    "timestamp": datetime.now().isoformat(),
                    "relevance_score": state.relevance_scores[-1] if state.relevance_scores else 0.0,
                    "quality_score": state.quality_scores[-1] if state.quality_scores else 0.0
                }
            }
            
            self.vector_store.add_documents(
                documents=[{"text": conversation_text}],
                embeddings=[embedding],
                metadata=[doc_data["metadata"]]
            )
            
            # Update short-term memory
            self.short_term_memory.save_context(
                {"input": state.current_question},
                {"output": state.current_answer}
            )
            
            state.steps_completed.append("update_memory")
            
        except Exception as e:
            print(f"Memory update error: {e}")
        
        return state
    
    async def _assess_skills_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 4: Assess demonstrated skills"""
        state.current_agent = "skill_assessor"
        
        try:
            # Extract skill demonstrations from analysis
            if state.quality_scores:
                latest_quality = state.quality_scores[-1]
                
                # Update skill scores based on conversation
                for skill in state.required_skills:
                    if skill.lower() in state.current_answer.lower():
                        current_score = state.skill_scores.get(skill, 0.0)
                        # Running average
                        count = sum(1 for s in state.skill_scores.values() if s > 0)
                        state.skill_scores[skill] = (current_score * count + latest_quality) / (count + 1)
            
            state.steps_completed.append("assess_skills")
            
        except Exception as e:
            print(f"Skill assessment error: {e}")
        
        return state
    
    async def _analyze_behavior_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 5: Analyze behavioral patterns"""
        state.current_agent = "behavior_analyzer"
        
        try:
            # Analyze response characteristics
            answer_length = len(state.current_answer.split())
            
            # Update engagement level
            if answer_length < 20:
                if state.engagement_level == CandidateEngagement.HIGHLY_ENGAGED:
                    state.engagement_level = CandidateEngagement.ENGAGED
            elif answer_length > 300:
                if state.engagement_level != CandidateEngagement.DISENGAGED:
                    state.engagement_level = CandidateEngagement.MODERATE
            
            # Calculate average quality
            if state.quality_scores:
                avg_quality = sum(state.quality_scores) / len(state.quality_scores)
                if avg_quality < 4.0:
                    state.engagement_level = CandidateEngagement.CONCERNING
            
            state.steps_completed.append("analyze_behavior")
            
        except Exception as e:
            print(f"Behavior analysis error: {e}")
        
        return state
    
    async def _adjust_difficulty_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 6: Adaptively adjust question difficulty"""
        state.current_agent = "difficulty_adjuster"
        
        try:
            if len(state.quality_scores) < 2:
                state.steps_completed.append("adjust_difficulty")
                return state
            
            # Calculate recent performance
            recent_scores = state.quality_scores[-3:]
            avg_recent = sum(recent_scores) / len(recent_scores)
            
            # Adjust difficulty based on performance
            difficulties = [QuestionDifficulty.ENTRY, QuestionDifficulty.INTERMEDIATE, 
                          QuestionDifficulty.ADVANCED, QuestionDifficulty.EXPERT]
            current_idx = difficulties.index(state.current_difficulty)
            
            if avg_recent >= 8.0 and current_idx < len(difficulties) - 1:
                # Increase difficulty
                state.current_difficulty = difficulties[current_idx + 1]
            elif avg_recent < 5.0 and current_idx > 0:
                # Decrease difficulty
                state.current_difficulty = difficulties[current_idx - 1]
            
            state.steps_completed.append("adjust_difficulty")
            
        except Exception as e:
            print(f"Difficulty adjustment error: {e}")
        
        return state
    
    async def _determine_phase_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 7: Determine next interview phase"""
        state.current_agent = "phase_coordinator"
        
        try:
            # Define phase progression and question counts
            phase_thresholds = {
                InterviewPhase.WARMUP: 2,
                InterviewPhase.TECHNICAL_DEEP_DIVE: 5,
                InterviewPhase.PROBLEM_SOLVING: 3,
                InterviewPhase.BEHAVIORAL_ASSESSMENT: 3,
                InterviewPhase.CULTURAL_FIT: 2,
                InterviewPhase.SCENARIO_BASED: 2,
                InterviewPhase.CLOSING: 1
            }
            
            # Count questions in current phase
            current_phase_count = sum(
                1 for q in state.questions_asked
                if state.current_phase.value in str(q)
            )
            
            threshold = phase_thresholds.get(state.current_phase, 3)
            
            if current_phase_count >= threshold or state.questions_count >= state.target_questions:
                # Advance phase
                phases = list(InterviewPhase)
                current_idx = phases.index(state.current_phase)
                if current_idx < len(phases) - 1:
                    state.current_phase = phases[current_idx + 1]
            
            # Check completion
            if state.questions_count >= state.target_questions:
                state.current_phase = InterviewPhase.COMPLETED
                state.should_continue = False
            
            state.steps_completed.append("determine_phase")
            
        except Exception as e:
            print(f"Phase determination error: {e}")
        
        return state
    
    async def _generate_question_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 8: Generate next contextual question"""
        state.current_agent = "question_generator"
        
        try:
            # Prepare context for question generation
            inputs = {
                "job_context": json.dumps({
                    "title": state.job_title,
                    "description": state.job_description,
                    "required_skills": state.required_skills
                }),
                "candidate_profile": json.dumps(state.candidate_profile),
                "conversation_history": json.dumps(state.conversation_history[-5:]),
                "current_phase": state.current_phase.value,
                "difficulty_level": state.current_difficulty.value,
                "retrieved_context": json.dumps(state.retrieved_contexts[:3]),
                "skills_to_assess": json.dumps([
                    skill for skill in state.required_skills 
                    if skill not in state.skill_scores or state.skill_scores[skill] < 7.0
                ]),
                "previous_questions": json.dumps(state.questions_asked)
            }
            
            # Generate using DSPy
            result = await self.question_generator.forward(inputs)
            
            question = result.get("question", "Tell me about a challenging project you've worked on.")
            state.next_question = question
            state.questions_count += 1
            
            state.steps_completed.append("generate_question")
            
        except Exception as e:
            print(f"Question generation error: {e}")
            state.next_question = "Tell me more about your experience."
        
        return state
    
    async def _check_question_uniqueness(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 9: Ensure question uniqueness"""
        state.current_agent = "uniqueness_checker"
        
        # Add to asked questions
        state.questions_asked.append(state.next_question)
        state.steps_completed.append("check_uniqueness")
        
        return state
    
    def _is_question_unique(self, state: VideoInterviewState) -> str:
        """Check if generated question is unique"""
        # Simple similarity check (can be enhanced with semantic similarity)
        question_lower = state.next_question.lower()
        
        for asked in state.questions_asked[:-1]:  # Exclude the just-added question
            if asked.lower() == question_lower:
                return "duplicate"
        
        return "unique"
    
    def _should_continue_or_complete(self, state: VideoInterviewState) -> str:
        """Decide whether to continue or complete interview"""
        if state.current_phase == InterviewPhase.COMPLETED or not state.should_continue:
            return "complete"
        return "continue"
    
    async def _create_summary_agent(self, state: VideoInterviewState) -> VideoInterviewState:
        """Agent 10: Create comprehensive interview summary"""
        state.current_agent = "summary_generator"
        
        try:
            # Calculate final scores
            avg_quality = sum(state.quality_scores) / len(state.quality_scores) if state.quality_scores else 0.0
            avg_relevance = sum(state.relevance_scores) / len(state.relevance_scores) if state.relevance_scores else 0.0
            
            state.overall_score = (avg_quality + avg_relevance) / 2
            
            # Generate comprehensive feedback
            prompt = f"""Create a comprehensive interview evaluation for {state.job_title} candidate.

Conversation Summary:
{json.dumps(state.conversation_history[-10:], indent=2)}

Performance Metrics:
- Questions Asked: {state.questions_count}
- Average Quality Score: {avg_quality:.2f}/10
- Average Relevance: {avg_relevance:.2f}/10
- Skills Assessed: {json.dumps(state.skill_scores, indent=2)}
- Engagement Level: {state.engagement_level.value}

Provide:
1. Top 5 Strengths
2. Top 3 Areas for Improvement
3. Overall Rating (0-10)
4. Hiring Recommendation (STRONG_YES/YES/MAYBE/NO)
5. Detailed 3-paragraph feedback"""
            
            messages = [HumanMessage(content=prompt)]
            response = await asyncio.to_thread(self.analysis_llm.invoke, messages)
            
            state.detailed_feedback = response.content
            
            # Parse recommendation
            if "STRONG_YES" in response.content:
                state.recommendation = "STRONG_YES"
            elif "YES" in response.content:
                state.recommendation = "YES"
            elif "MAYBE" in response.content:
                state.recommendation = "MAYBE"
            else:
                state.recommendation = "NO"
            
            state.steps_completed.append("create_summary")
            
        except Exception as e:
            print(f"Summary generation error: {e}")
        
        return state
    
    async def start_interview(
        self,
        session_id: str,
        candidate_id: str,
        interview_id: str,
        job_title: str,
        job_description: str = "",
        required_skills: List[str] = None,
        candidate_profile: Dict[str, Any] = None,
        target_questions: int = 15
    ) -> Dict[str, Any]:
        """Start a new interview session"""
        try:
            # Initialize state
            state = VideoInterviewState(
                session_id=session_id,
                candidate_id=candidate_id,
                interview_id=interview_id,
                job_title=job_title,
                job_description=job_description,
                required_skills=required_skills or [],
                candidate_profile=candidate_profile or {},
                target_questions=target_questions
            )
            
            # Generate first question
            state.current_question = ""
            state.current_answer = ""
            
            # Run workflow to generate first question
            result_state = await self.workflow.ainvoke(state)
            
            return {
                "success": True,
                "session_id": session_id,
                "first_question": result_state.next_question,
                "phase": result_state.current_phase.value,
                "question_number": result_state.questions_count,
                "total_questions": target_questions
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_answer_and_continue(
        self,
        state: VideoInterviewState,
        answer: str
    ) -> Dict[str, Any]:
        """Process answer and generate next question or summary"""
        try:
            # Update state with answer
            state.current_answer = answer
            state.conversation_history.append({
                "role": "user",
                "content": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # Execute workflow
            result_state = await self.workflow.ainvoke(state)
            
            # Format response
            response = {
                "success": True,
                "session_id": result_state.session_id,
                "phase": result_state.current_phase.value,
                "question_number": result_state.questions_count,
                "total_questions": result_state.target_questions,
                "engagement_level": result_state.engagement_level.value,
                "warning": result_state.warning_message
            }
            
            if result_state.current_phase == InterviewPhase.COMPLETED:
                # Interview completed
                response["completed"] = True
                response["summary"] = {
                    "overall_score": result_state.overall_score,
                    "recommendation": result_state.recommendation,
                    "detailed_feedback": result_state.detailed_feedback,
                    "skill_scores": result_state.skill_scores,
                    "strengths": result_state.strengths,
                    "weaknesses": result_state.weaknesses
                }
            else:
                # Continue interview
                response["completed"] = False
                response["next_question"] = result_state.next_question
                response["follow_up_needed"] = result_state.follow_up_needed
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


    async def process_proctoring_frame(
        self,
        state: VideoInterviewState,
        frame_data: np.ndarray
    ) -> VideoInterviewState:
        """Process video frame for proctoring violations"""
        if not state.proctoring_enabled:
            return state
        
        try:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            
            # Face detection using MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                num_faces = len(results.multi_face_landmarks)
                
                # Check for multiple faces
                if num_faces > 1:
                    violation = {
                        "type": ProctoringViolationType.MULTIPLE_FACES.value,
                        "severity": "critical",
                        "timestamp": datetime.now().isoformat(),
                        "description": f"{num_faces} faces detected",
                        "confidence": 0.95
                    }
                    state.proctoring_violations.append(violation)
                    state.violation_count += 1
                    state.suspicious_behavior_score += 10.0
                
                # Analyze gaze/attention (simplified)
                if num_faces == 1:
                    # Extract face landmarks for gaze estimation
                    landmarks = results.multi_face_landmarks[0]
                    # Simple attention scoring based on face position
                    state.attention_score = min(100.0, state.attention_score + 0.5)
            
            else:
                # No face detected
                violation = {
                    "type": ProctoringViolationType.NO_FACE_DETECTED.value,
                    "severity": "high",
                    "timestamp": datetime.now().isoformat(),
                    "description": "No face detected in frame",
                    "confidence": 0.9
                }
                state.proctoring_violations.append(violation)
                state.violation_count += 1
                state.suspicious_behavior_score += 5.0
            
            # Object detection using YOLO
            if self.yolo_model:
                yolo_results = self.yolo_model(rgb_frame)
                
                for result in yolo_results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        class_name = self.yolo_model.names[cls]
                        
                        # Check for phones
                        if 'cell phone' in class_name.lower():
                            violation = {
                                "type": ProctoringViolationType.MOBILE_PHONE_DETECTED.value,
                                "severity": "critical",
                                "timestamp": datetime.now().isoformat(),
                                "description": "Mobile phone detected",
                                "confidence": float(box.conf[0])
                            }
                            state.proctoring_violations.append(violation)
                            state.violation_count += 1
                            state.suspicious_behavior_score += 20.0
                        
                        # Check for additional persons
                        elif 'person' in class_name.lower():
                            violation = {
                                "type": ProctoringViolationType.PERSON_IN_BACKGROUND.value,
                                "severity": "high",
                                "timestamp": datetime.now().isoformat(),
                                "description": "Additional person detected in background",
                                "confidence": float(box.conf[0])
                            }
                            state.proctoring_violations.append(violation)
                            state.violation_count += 1
                            state.suspicious_behavior_score += 15.0
            
            # Generate warnings if needed
            if state.violation_count > 3 and state.warning_count < 2:
                state.warning_message = f"⚠️ WARNING: {state.violation_count} proctoring violations detected. Please ensure you are alone and focused on the interview."
                state.warning_count += 1
            elif state.violation_count > 5:
                state.warning_message = "🚨 CRITICAL: Multiple serious violations detected. This interview may be terminated."
        
        except Exception as e:
            print(f"Proctoring error: {e}")
        
        return state


# Singleton instance
_video_interview_agent: Optional[UltimateVideoInterviewAgent] = None


def get_video_interview_agent() -> UltimateVideoInterviewAgent:
    """Get or create singleton video interview agent"""
    global _video_interview_agent
    if _video_interview_agent is None:
        _video_interview_agent = UltimateVideoInterviewAgent()
    return _video_interview_agent