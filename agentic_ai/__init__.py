"""
🤖 HIREGENIX AGENTIC AI - World-Class Interview Intelligence System

This package provides a comprehensive suite of agentic AI modules for
conducting human-like, context-aware, and deeply intelligent interviews.

Modules:
- config: Configuration management
- memory_layer: Persistent memory across sessions
- real_time_adaptation_engine: Dynamic interview adjustment
- human_behavior_simulator: Natural conversation patterns
- drill_down_engine: Multi-level deep probing
- cross_session_context: Multi-round interview context
- voice_native_processor: WebRTC + Whisper integration
- live_coding_observer: Real-time code analysis
- panel_interview_mode: Multi-persona interviews
- candidate_question_handler: Q&A phase handling
- enhanced_deep_sensing: Advanced behavioral analysis
- integration_layer: Unified orchestration hub

Tech Stack:
- LangGraph for multi-step workflows
- LangChain for LLM orchestration
- Azure OpenAI (GPT-4o/GPT-5)
- Redis for persistent storage
- SentenceTransformers for embeddings
"""

__version__ = "2.0.0"
__author__ = "HireGenix AI Team"

# Original config exports
from .config import (
    get_config,
    validate_config,
    AgenticAIConfig,
    AgentConfig,
    AzureOpenAIConfig,
    RAGConfig,
    WebSearchConfig
)

# Core integration layer
try:
    from .integration_layer import (
        AgenticAIIntegrationLayer,
        get_integration_layer,
        initialize_agentic_interview,
        process_interview_response,
        get_next_question,
        InterviewMode,
        ProcessingStage
    )
except ImportError:
    pass

# Memory and context modules
try:
    from .memory_layer import (
        PersistentMemoryLayer,
        get_memory_layer,
        InterviewMemoryState
    )
except ImportError:
    pass

try:
    from .cross_session_context import (
        CrossSessionContextManager,
        get_cross_session_manager,
        InterviewRound,
        HandoffPriority
    )
except ImportError:
    pass

# Real-time processing modules
try:
    from .real_time_adaptation_engine import (
        RealTimeAdaptationEngine,
        get_adaptation_engine,
        AdaptationMode
    )
except ImportError:
    pass

try:
    from .human_behavior_simulator import (
        HumanBehaviorSimulator,
        get_behavior_simulator,
        ConversationTone
    )
except ImportError:
    pass

try:
    from .enhanced_deep_sensing import (
        EnhancedDeepSensing,
        get_enhanced_deep_sensing,
        PauseType,
        MicroExpression,
        StressLevel
    )
except ImportError:
    pass

# Questioning engines
try:
    from .drill_down_engine import (
        DrillDownQuestionEngine,
        get_drill_down_engine,
        AnswerDepth,
        ProbeType
    )
except ImportError:
    pass

try:
    from .candidate_question_handler import (
        CandidateQuestionHandler,
        get_question_handler,
        QuestionQuality,
        QuestionCategory
    )
except ImportError:
    pass

# Specialized mode modules
try:
    from .voice_native_processor import (
        VoiceNativeProcessor,
        get_voice_processor,
        SpeechEmotion,
        SpeechClarity
    )
except ImportError:
    pass

try:
    from .live_coding_observer import (
        LiveCodingObserver,
        get_coding_observer,
        CodingPhase,
        CodeQuality,
        ProgrammingLanguage
    )
except ImportError:
    pass

try:
    from .panel_interview_mode import (
        PanelInterviewMode,
        get_panel_interview,
        InterviewerRole,
        InterviewerPersona
    )
except ImportError:
    pass

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Config (original)
    'get_config',
    'validate_config',
    'AgenticAIConfig',
    'AgentConfig',
    'AzureOpenAIConfig',
    'RAGConfig',
    'WebSearchConfig',
    
    # Integration Layer
    "AgenticAIIntegrationLayer",
    "get_integration_layer",
    "initialize_agentic_interview",
    "process_interview_response",
    "get_next_question",
    "InterviewMode",
    "ProcessingStage",
    
    # Memory Layer
    "PersistentMemoryLayer",
    "get_memory_layer",
    "InterviewMemoryState",
    
    # Cross-Session Context
    "CrossSessionContextManager",
    "get_cross_session_manager",
    "InterviewRound",
    "HandoffPriority",
    
    # Adaptation Engine
    "RealTimeAdaptationEngine",
    "get_adaptation_engine",
    "AdaptationMode",
    
    # Human Behavior
    "HumanBehaviorSimulator",
    "get_behavior_simulator",
    "ConversationTone",
    
    # Deep Sensing
    "EnhancedDeepSensing",
    "get_enhanced_deep_sensing",
    "PauseType",
    "MicroExpression",
    "StressLevel",
    
    # Drill Down
    "DrillDownQuestionEngine",
    "get_drill_down_engine",
    "AnswerDepth",
    "ProbeType",
    
    # Candidate Questions
    "CandidateQuestionHandler",
    "get_question_handler",
    "QuestionQuality",
    "QuestionCategory",
    
    # Voice Processing
    "VoiceNativeProcessor",
    "get_voice_processor",
    "SpeechEmotion",
    "SpeechClarity",
    
    # Coding Observer
    "LiveCodingObserver",
    "get_coding_observer",
    "CodingPhase",
    "CodeQuality",
    "ProgrammingLanguage",
    
    # Panel Interview
    "PanelInterviewMode",
    "get_panel_interview",
    "InterviewerRole",
    "InterviewerPersona"
]