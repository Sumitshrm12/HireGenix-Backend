# prescreening/models.py - Pydantic models for pre-screening system
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ScoreBucket(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    POTENTIAL = "potential"
    NOT_ELIGIBLE = "not_eligible"

class PreScreeningStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class SessionStatus(str, Enum):
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_MORE_INFO = "request_more_info"
    OVERRIDE_AUTO_DECISION = "override_auto_decision"

# Request Models
class ResumeAnalysisRequest(BaseModel):
    candidate_id: str
    job_id: str
    resume_text: str

class PreScreeningSessionCreateRequest(BaseModel):
    candidate_id: str
    job_id: str
    expires_in_hours: int = 48

class MCQAnswerRequest(BaseModel):
    question_id: str
    answer: str  # A, B, C, or D
    time_taken: int  # seconds

class ProctoringEventRequest(BaseModel):
    session_id: str
    event_type: str
    timestamp: datetime
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = {}

class HumanReviewDecisionRequest(BaseModel):
    task_id: str
    decision: ReviewDecision
    reason: str
    notes: Optional[str] = None
    override_score: Optional[float] = Field(None, ge=0.0, le=100.0)

# Response Models
class ResumeMatchingResult(BaseModel):
    id: str
    candidate_id: str
    job_id: str
    overall_score: float
    embedding_score: float
    keyword_score: float
    experience_score: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    score_rationale: str
    bucket: ScoreBucket
    processing_time_ms: Optional[int] = None
    created_at: datetime

class MCQQuestion(BaseModel):
    id: str
    question: str
    options: Dict[str, str]  # {A: "option1", B: "option2", ...}
    difficulty_level: str
    skill_category: str
    time_limit: int = 60  # seconds

class MCQQuestionWithAnswer(MCQQuestion):
    correct_answer: str
    rationale: Optional[str] = None

class PreScreeningSession(BaseModel):
    id: str
    candidate_id: str
    job_id: str
    session_token: str
    status: SessionStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    expires_at: datetime
    
    intro_question: str
    intro_response_url: Optional[str] = None
    intro_transcript: Optional[str] = None
    
    mcq_questions: List[MCQQuestion] = []
    mcq_answers: List[Dict[str, Any]] = []
    correct_answers: int = 0
    total_questions: int = 10
    
    mcq_score: Optional[float] = None
    intro_score: Optional[float] = None
    proctoring_penalty: float = 0
    final_score: Optional[float] = None
    
    created_at: datetime
    updated_at: datetime

class ProctoringEvent(BaseModel):
    id: str
    session_id: str
    event_type: str
    timestamp: datetime
    confidence_score: float
    severity: str = "medium"
    metadata: Dict[str, Any] = {}
    reviewed: bool = False
    created_at: datetime

class HumanReviewTask(BaseModel):
    id: str
    candidate_id: str
    job_id: str
    stage: str
    score: float
    review_type: str
    priority: str = "medium"
    status: ReviewStatus
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    decision: Optional[ReviewDecision] = None
    decision_reason: Optional[str] = None
    decision_notes: Optional[str] = None
    override_score: Optional[float] = None
    context_data: Dict[str, Any] = {}
    ai_recommendation: Optional[str] = None
    recommendation_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class CandidatePreScreening(BaseModel):
    id: str
    candidate_id: str
    job_id: str
    resume_score: float
    resume_decision: str
    prescreening_status: PreScreeningStatus
    prescreening_score: Optional[float] = None
    prescreening_decision: Optional[str] = None
    human_review_required: bool = False
    created_at: datetime
    updated_at: datetime
    
    # Extended attributes for agentic AI workflow
    overall_score: Optional[float] = None
    embedding_score: Optional[float] = None
    keyword_score: Optional[float] = None
    experience_score: Optional[float] = None
    bucket: Optional[ScoreBucket] = None
    next_action: Optional[str] = None
    requires_human_review: bool = False
    resume_matching_result: Optional[ResumeMatchingResult] = None
    interview_recommendations: Optional[Dict[str, Any]] = None
    mcq_questions: Optional[List[MCQQuestionWithAnswer]] = None

# Detailed response models with nested data
class PreScreeningSessionDetail(PreScreeningSession):
    resume_matching_result: Optional[ResumeMatchingResult] = None
    proctoring_events: List[ProctoringEvent] = []
    human_review_tasks: List[HumanReviewTask] = []

class SessionStartResponse(BaseModel):
    session_id: str
    session_token: str
    intro_question: str
    proctoring_config: Dict[str, Any]
    time_limit_minutes: int = 30

class SessionCompleteResponse(BaseModel):
    final_score: float
    decision: str
    next_steps: str
    breakdown: Dict[str, Any]

class DashboardAnalytics(BaseModel):
    pending_reviews: int
    automation_rate: float
    avg_processing_time_minutes: float
    total_sessions_today: int
    completion_rate: float

# Configuration Models
class PreScreeningConfig(BaseModel):
    company_id: str
    job_id: Optional[str] = None
    excellent_threshold: float = 80.0
    good_threshold: float = 70.0
    potential_threshold: float = 60.0
    total_mcq_questions: int = 10
    easy_questions: int = 3
    medium_questions: int = 5
    hard_questions: int = 2
    session_time_limit_minutes: int = 30
    mcq_time_limit_seconds: int = 60
    intro_time_limit_seconds: int = 180
    enable_proctoring: bool = True
    strict_mode: bool = False
    max_tab_switches: int = 2
    max_face_lost_events: int = 3

# Notification Models
class NotificationType(str, Enum):
    SESSION_INVITE = "session_invite"
    SESSION_REMINDER = "session_reminder"
    SESSION_STARTED = "session_started"
    SESSION_COMPLETED = "session_completed"
    REVIEW_REQUIRED = "review_required"
    DECISION_MADE = "decision_made"

class PreScreeningNotification(BaseModel):
    id: str
    prescreening_id: str
    candidate_id: str
    job_id: str
    notification_type: NotificationType
    title: str
    message: str
    metadata: Dict[str, Any] = {}
    sent: bool = False
    created_at: datetime

class NotificationRequest(BaseModel):
    candidate_id: str
    job_id: str
    type: str
    channel: str
    template_variables: Dict[str, Any] = {}

class NotificationResponse(BaseModel):
    id: str
    status: str
    message: str
    sent_at: Optional[datetime] = None