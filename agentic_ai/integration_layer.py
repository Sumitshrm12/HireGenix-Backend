"""
🔗 AGENTIC AI INTEGRATION LAYER - Unified Orchestration Hub
Connects all agentic AI modules into a cohesive interview system.
Provides unified API for the main interview agent to leverage all capabilities.

Features:
- Unified module orchestration
- Automatic capability routing
- State synchronization
- Performance optimization
- Fallback handling
- Metrics collection
- Real-time coordination

Tech Stack:
- LangGraph for workflow management
- Async coordination
- Singleton pattern for modules
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from pydantic import BaseModel, Field

# Import all agentic modules
from .memory_layer import get_memory_layer, PersistentMemoryLayer
from .real_time_adaptation_engine import get_adaptation_engine, RealTimeAdaptationEngine
from .human_behavior_simulator import get_human_behavior_simulator as get_behavior_simulator, HumanBehaviorSimulator
from .drill_down_engine import get_drill_down_engine, DrillDownQuestionEngine
from .cross_session_context import get_cross_session_manager, CrossSessionContextManager
from .voice_native_processor import get_voice_processor, VoiceNativeProcessor
from .live_coding_observer import get_coding_observer, LiveCodingObserver
from .panel_interview_mode import get_panel_interview, PanelInterviewMode
from .candidate_question_handler import get_question_handler, CandidateQuestionHandler
from .enhanced_deep_sensing import get_enhanced_deep_sensing, EnhancedDeepSensing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewMode(str, Enum):
    """Available interview modes"""
    STANDARD = "standard"  # Single interviewer
    PANEL = "panel"  # Multiple AI personas
    TECHNICAL = "technical"  # Technical with coding
    BEHAVIORAL = "behavioral"  # Focus on behavioral questions
    EXECUTIVE = "executive"  # Senior-level interview


class ProcessingStage(str, Enum):
    """Interview processing stages"""
    INITIALIZATION = "initialization"
    GREETING = "greeting"
    MAIN_INTERVIEW = "main_interview"
    DEEP_DIVE = "deep_dive"
    CODING = "coding"
    QA_PHASE = "qa_phase"
    CLOSING = "closing"


class IntegrationState(BaseModel):
    """Unified state for integration layer"""
    # Session identifiers
    session_id: str = ""
    candidate_id: str = ""
    job_id: str = ""
    company_id: str = ""
    
    # Mode and stage
    interview_mode: str = InterviewMode.STANDARD.value
    current_stage: str = ProcessingStage.INITIALIZATION.value
    
    # Context
    candidate_profile: Dict[str, Any] = Field(default_factory=dict)
    job_requirements: Dict[str, Any] = Field(default_factory=dict)
    company_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Conversation state
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_question: str = ""
    current_answer: str = ""
    
    # Module outputs
    memory_context: Dict[str, Any] = Field(default_factory=dict)
    adaptation_signals: Dict[str, Any] = Field(default_factory=dict)
    behavioral_signals: Dict[str, Any] = Field(default_factory=dict)
    sensing_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Generated outputs
    enhanced_question: str = ""
    human_like_response: str = ""
    
    # Metrics
    module_timings: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class AgenticAIIntegrationLayer:
    """
    Central integration hub for all agentic AI modules.
    Orchestrates the modules and provides a unified interface.
    """
    
    def __init__(self):
        logger.info("🔗 Initializing Agentic AI Integration Layer...")
        
        # Initialize all module singletons
        self._init_modules()
        
        # Module registry
        self.module_registry = {}
        self._register_modules()
        
        # Active sessions
        self.sessions: Dict[str, IntegrationState] = {}
        
        logger.info("✅ Agentic AI Integration Layer initialized")
    
    def _init_modules(self):
        """Initialize all agentic AI modules"""
        try:
            self.memory_layer = get_memory_layer()
            logger.info("  ✓ Memory Layer initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Memory Layer unavailable: {e}")
            self.memory_layer = None
        
        try:
            self.adaptation_engine = get_adaptation_engine()
            logger.info("  ✓ Adaptation Engine initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Adaptation Engine unavailable: {e}")
            self.adaptation_engine = None
        
        try:
            self.behavior_simulator = get_behavior_simulator()
            logger.info("  ✓ Behavior Simulator initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Behavior Simulator unavailable: {e}")
            self.behavior_simulator = None
        
        try:
            self.drill_down_engine = get_drill_down_engine()
            logger.info("  ✓ Drill-Down Engine initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Drill-Down Engine unavailable: {e}")
            self.drill_down_engine = None
        
        try:
            self.cross_session_manager = get_cross_session_manager()
            logger.info("  ✓ Cross-Session Manager initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Cross-Session Manager unavailable: {e}")
            self.cross_session_manager = None
        
        try:
            self.voice_processor = get_voice_processor()
            logger.info("  ✓ Voice Processor initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Voice Processor unavailable: {e}")
            self.voice_processor = None
        
        try:
            self.coding_observer = get_coding_observer()
            logger.info("  ✓ Coding Observer initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Coding Observer unavailable: {e}")
            self.coding_observer = None
        
        try:
            self.panel_interview = get_panel_interview()
            logger.info("  ✓ Panel Interview initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Panel Interview unavailable: {e}")
            self.panel_interview = None
        
        try:
            self.question_handler = get_question_handler()
            logger.info("  ✓ Question Handler initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Question Handler unavailable: {e}")
            self.question_handler = None
        
        try:
            self.enhanced_sensing = get_enhanced_deep_sensing()
            logger.info("  ✓ Enhanced Sensing initialized")
        except Exception as e:
            logger.warning(f"  ⚠ Enhanced Sensing unavailable: {e}")
            self.enhanced_sensing = None
    
    def _register_modules(self):
        """Register modules with their capabilities"""
        self.module_registry = {
            "memory": {
                "module": self.memory_layer,
                "capabilities": ["context_retrieval", "pattern_analysis", "opening_generation"],
                "required_for": [InterviewMode.STANDARD, InterviewMode.PANEL, InterviewMode.TECHNICAL]
            },
            "adaptation": {
                "module": self.adaptation_engine,
                "capabilities": ["real_time_adjustment", "difficulty_scaling", "emotional_adaptation"],
                "required_for": [InterviewMode.STANDARD, InterviewMode.BEHAVIORAL]
            },
            "behavior": {
                "module": self.behavior_simulator,
                "capabilities": ["human_like_responses", "acknowledgments", "thinking_pauses"],
                "required_for": [InterviewMode.STANDARD, InterviewMode.PANEL, InterviewMode.BEHAVIORAL]
            },
            "drill_down": {
                "module": self.drill_down_engine,
                "capabilities": ["deep_probing", "contradiction_detection", "skill_verification"],
                "required_for": [InterviewMode.TECHNICAL, InterviewMode.EXECUTIVE]
            },
            "cross_session": {
                "module": self.cross_session_manager,
                "capabilities": ["round_handoff", "context_sharing", "journey_tracking"],
                "required_for": [InterviewMode.STANDARD, InterviewMode.PANEL]
            },
            "voice": {
                "module": self.voice_processor,
                "capabilities": ["transcription", "prosody_analysis", "voice_stress"],
                "required_for": []  # Optional for all
            },
            "coding": {
                "module": self.coding_observer,
                "capabilities": ["code_analysis", "bug_detection", "approach_assessment"],
                "required_for": [InterviewMode.TECHNICAL]
            },
            "panel": {
                "module": self.panel_interview,
                "capabilities": ["multi_persona", "coordinated_questioning", "consensus"],
                "required_for": [InterviewMode.PANEL, InterviewMode.EXECUTIVE]
            },
            "qa_handler": {
                "module": self.question_handler,
                "capabilities": ["candidate_questions", "engagement_assessment"],
                "required_for": [InterviewMode.STANDARD, InterviewMode.PANEL]
            },
            "sensing": {
                "module": self.enhanced_sensing,
                "capabilities": ["pause_analysis", "micro_expressions", "stress_tracking"],
                "required_for": [InterviewMode.BEHAVIORAL, InterviewMode.EXECUTIVE]
            }
        }
    
    def get_available_modules(self) -> Dict[str, bool]:
        """Get availability status of all modules"""
        return {
            name: info["module"] is not None
            for name, info in self.module_registry.items()
        }
    
    async def initialize_session(
        self,
        session_id: str,
        candidate_id: str,
        job_id: str,
        company_id: str,
        interview_mode: str = InterviewMode.STANDARD.value,
        candidate_profile: Optional[Dict[str, Any]] = None,
        job_requirements: Optional[Dict[str, Any]] = None,
        company_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Initialize a new interview session with all modules"""
        try:
            # Create session state
            state = IntegrationState(
                session_id=session_id,
                candidate_id=candidate_id,
                job_id=job_id,
                company_id=company_id,
                interview_mode=interview_mode,
                current_stage=ProcessingStage.INITIALIZATION.value,
                candidate_profile=candidate_profile or {},
                job_requirements=job_requirements or {},
                company_info=company_info or {}
            )
            
            # Initialize each relevant module
            initialization_results = {}
            
            # 1. Load memory context
            if self.memory_layer:
                start = datetime.now()
                try:
                    memory_result = await self.memory_layer.get_interview_context_summary(
                        candidate_id=candidate_id,
                        company_id=company_id
                    )
                    memory_result["success"] = True
                    memory_result["has_history"] = memory_result.get("previous_sessions", 0) > 0
                except Exception as mem_err:
                    logger.warning(f"Memory retrieval error: {mem_err}")
                    memory_result = {"success": False, "has_history": False}
                state.memory_context = memory_result
                state.module_timings["memory"] = (datetime.now() - start).total_seconds()
                initialization_results["memory"] = memory_result.get("success", False)
            
            # 2. Get cross-session context
            if self.cross_session_manager:
                start = datetime.now()
                try:
                    cross_context = await self.cross_session_manager.get_context_for_round(
                        candidate_id=candidate_id,
                        job_id=job_id,
                        round_type="initial",
                        candidate_profile=candidate_profile
                    )
                    state.module_timings["cross_session"] = (datetime.now() - start).total_seconds()
                    initialization_results["cross_session"] = cross_context.get("success", False) if isinstance(cross_context, dict) else False
                except Exception as cross_err:
                    logger.warning(f"Cross-session context error: {cross_err}")
                    initialization_results["cross_session"] = False
            
            # 3. Initialize panel if needed
            if interview_mode == InterviewMode.PANEL.value and self.panel_interview:
                panel_config = ["alex_tech_lead", "sarah_hiring_mgr", "mike_senior_eng"]
                panel_result = await self.panel_interview.create_panel(
                    session_id=session_id,
                    candidate_id=candidate_id,
                    job_id=job_id,
                    panel_config=panel_config,
                    candidate_profile=candidate_profile or {},
                    job_requirements=job_requirements or {}
                )
                initialization_results["panel"] = panel_result.get("success", False)
            
            # Store session
            self.sessions[session_id] = state
            
            return {
                "success": True,
                "session_id": session_id,
                "interview_mode": interview_mode,
                "modules_initialized": initialization_results,
                "available_modules": self.get_available_modules(),
                "has_previous_context": bool(state.memory_context.get("has_history"))
            }
            
        except Exception as e:
            logger.error(f"❌ Session initialization error: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_opening(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Generate personalized opening using all available modules"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            state.current_stage = ProcessingStage.GREETING.value
            
            opening_text = ""
            
            # Try memory-based personalized opening
            if self.memory_layer and state.memory_context.get("has_history"):
                memory_opening = await self.memory_layer.generate_personalized_opening(
                    candidate_id=state.candidate_id,
                    company_id=state.company_id,
                    job_id=state.job_id
                )
                if memory_opening.get("success"):
                    opening_text = memory_opening.get("opening", "")
            
            # Apply human behavior simulation
            if self.behavior_simulator and opening_text:
                humanized = await self.behavior_simulator.make_human_like(
                    text=opening_text,
                    context={
                        "candidate_name": state.candidate_profile.get("name", ""),
                        "is_opening": True
                    }
                )
                opening_text = humanized.get("result", opening_text)
            
            # Fallback to generic opening
            if not opening_text:
                name = state.candidate_profile.get("name", "")
                job_title = state.job_requirements.get("title", "the role")
                opening_text = f"Hi{' ' + name if name else ''}! Thank you for taking the time to interview with us today for {job_title}. I'm looking forward to learning more about your experience. How are you doing today?"
            
            self.sessions[session_id] = state
            
            return {
                "success": True,
                "opening": opening_text,
                "personalized": bool(state.memory_context.get("has_history")),
                "stage": state.current_stage
            }
            
        except Exception as e:
            logger.error(f"❌ Opening generation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_response(
        self,
        session_id: str,
        answer: str,
        audio_data: Optional[bytes] = None,
        video_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process candidate response through all relevant modules"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            state.current_answer = answer
            
            # Parallel processing of different signals
            tasks = []
            
            # 1. Voice processing (if audio available)
            if self.voice_processor and audio_data:
                tasks.append(("voice", self.voice_processor.process_audio(audio_data)))
            
            # 2. Enhanced deep sensing
            if self.enhanced_sensing:
                tasks.append(("sensing", self.enhanced_sensing.analyze(
                    session_id=session_id,
                    candidate_id=state.candidate_id,
                    transcript_segment=answer,
                    video_features=video_features
                )))
            
            # 3. Real-time adaptation
            if self.adaptation_engine:
                tasks.append(("adaptation", self.adaptation_engine.process_response(
                    session_id=session_id,
                    response=answer,
                    behavioral_signals=state.sensing_results
                )))
            
            # Execute in parallel
            results = {}
            if tasks:
                task_results = await asyncio.gather(
                    *[t[1] for t in tasks],
                    return_exceptions=True
                )
                for i, (name, _) in enumerate(tasks):
                    if not isinstance(task_results[i], Exception):
                        results[name] = task_results[i]
                    else:
                        logger.warning(f"⚠ {name} processing failed: {task_results[i]}")
            
            # Update state with results
            if "sensing" in results:
                state.sensing_results = results["sensing"]
            if "adaptation" in results:
                state.adaptation_signals = results["adaptation"]
            if "voice" in results:
                state.behavioral_signals.update(results.get("voice", {}))
            
            # Store interaction in memory
            if self.memory_layer:
                await self.memory_layer.store_interaction(
                    candidate_id=state.candidate_id,
                    company_id=state.company_id,
                    job_id=state.job_id,
                    interaction={
                        "question": state.current_question,
                        "answer": answer,
                        "timestamp": datetime.now().isoformat()
                    },
                    behavioral_signals=state.sensing_results
                )
            
            # Add to conversation history
            state.conversation_history.append({
                "question": state.current_question,
                "answer": answer,
                "sensing": state.sensing_results.get("metrics", {}),
                "timestamp": datetime.now().isoformat()
            })
            
            self.sessions[session_id] = state
            
            return {
                "success": True,
                "sensing_summary": state.sensing_results.get("summary", ""),
                "adaptation_mode": state.adaptation_signals.get("mode", ""),
                "engagement_score": state.sensing_results.get("metrics", {}).get("engagement_score", 0.5),
                "stress_level": state.sensing_results.get("metrics", {}).get("stress_level", "moderate"),
                "recommendations": state.adaptation_signals.get("recommendations", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Response processing error: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_next_question(
        self,
        session_id: str,
        topic: Optional[str] = None,
        enable_drill_down: bool = True
    ) -> Dict[str, Any]:
        """Generate the next question using all available intelligence"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            state.current_stage = ProcessingStage.MAIN_INTERVIEW.value
            
            question = ""
            question_type = "standard"
            
            # Check if we should drill down
            if enable_drill_down and self.drill_down_engine and state.current_answer:
                # Analyze if previous answer needs probing
                last_exchange = state.conversation_history[-1] if state.conversation_history else {}
                
                drill_result = await self.drill_down_engine.start_drill_down(
                    topic=topic or last_exchange.get("topic", "general"),
                    initial_question=last_exchange.get("question", ""),
                    initial_answer=state.current_answer,
                    job_context=state.job_requirements
                )
                
                if drill_result.get("success") and drill_result.get("should_continue"):
                    question = drill_result.get("next_question", "")
                    question_type = "drill_down"
            
            # Apply adaptation signals to question
            if self.adaptation_engine and question:
                adapted = await self.adaptation_engine.adapt_question(
                    session_id=session_id,
                    question=question,
                    current_mode=state.adaptation_signals.get("mode", "balanced")
                )
                question = adapted.get("adapted_question", question)
            
            # Apply human-like behavior
            if self.behavior_simulator and question:
                # Add acknowledgment of previous answer
                acknowledgment = await self.behavior_simulator.generate_acknowledgment(
                    answer=state.current_answer,
                    quality="good"
                )
                
                # Make question human-like
                humanized = await self.behavior_simulator.make_human_like(
                    text=question,
                    context={"previous_answer": state.current_answer}
                )
                
                ack_text = acknowledgment.get("acknowledgment", "")
                question_text = humanized.get("result", question)
                
                question = f"{ack_text} {question_text}".strip()
            
            # Store current question
            state.current_question = question
            self.sessions[session_id] = state
            
            return {
                "success": True,
                "question": question,
                "question_type": question_type,
                "adaptation_applied": bool(state.adaptation_signals),
                "current_stage": state.current_stage
            }
            
        except Exception as e:
            logger.error(f"❌ Question generation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def start_qa_phase(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Transition to Q&A phase where candidate asks questions"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            state.current_stage = ProcessingStage.QA_PHASE.value
            
            if not self.question_handler:
                return {
                    "success": True,
                    "opening": "Do you have any questions for me?",
                    "qa_enabled": False
                }
            
            # Initialize Q&A session
            qa_result = await self.question_handler.start_qa_session(
                session_id=session_id,
                candidate_id=state.candidate_id,
                job_id=state.job_id,
                company_info=state.company_info,
                job_details=state.job_requirements,
                team_info=state.company_info.get("team", {}),
                interview_history=state.conversation_history[-5:]
            )
            
            self.sessions[session_id] = state
            
            return {
                "success": True,
                "opening": qa_result.get("opening_prompt", "Do you have any questions?"),
                "qa_enabled": True,
                "current_stage": state.current_stage
            }
            
        except Exception as e:
            logger.error(f"❌ Q&A phase start error: {e}")
            return {"success": False, "error": str(e)}
    
    async def handle_candidate_question(
        self,
        session_id: str,
        question: str
    ) -> Dict[str, Any]:
        """Handle a question from the candidate"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            if not self.question_handler:
                return {
                    "success": True,
                    "answer": "That's a great question. I'd be happy to discuss that.",
                    "can_continue": True
                }
            
            result = await self.question_handler.handle_question(
                session_id=session_id,
                question=question
            )
            
            # Apply human behavior to response
            if self.behavior_simulator and result.get("answer"):
                humanized = await self.behavior_simulator.make_human_like(
                    text=result.get("answer", ""),
                    context={"is_qa_response": True}
                )
                result["answer"] = humanized.get("result", result.get("answer", ""))
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Candidate question handling error: {e}")
            return {"success": False, "error": str(e)}
    
    async def end_session(
        self,
        session_id: str,
        final_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """End session and generate comprehensive report"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            state.current_stage = ProcessingStage.CLOSING.value
            
            # Compile final report
            report = {
                "session_id": session_id,
                "candidate_id": state.candidate_id,
                "job_id": state.job_id,
                "interview_mode": state.interview_mode,
                "conversation_length": len(state.conversation_history),
                "module_timings": state.module_timings,
                "errors": state.errors
            }
            
            # Get Q&A assessment if available
            if self.question_handler:
                try:
                    qa_report = await self.question_handler.end_qa_session(session_id)
                    report["qa_assessment"] = qa_report
                except:
                    pass
            
            # Get sensing trajectory
            if self.enhanced_sensing:
                trajectory = self.enhanced_sensing.get_session_trajectory(session_id)
                report["behavioral_trajectory"] = trajectory
            
            # Record round completion for cross-session
            if self.cross_session_manager:
                await self.cross_session_manager.record_round_completion(
                    candidate_id=state.candidate_id,
                    job_id=state.job_id,
                    round_type="interview",
                    round_data={
                        "decision": "proceed",  # Would be determined by scoring
                        "skills_assessed": {},  # Would be collected during interview
                        "key_statements": [c.get("answer", "")[:100] for c in state.conversation_history[-3:]],
                        "topics_covered": [],
                        "notes": final_notes
                    }
                )
            
            # Clean up session
            del self.sessions[session_id]
            
            return {
                "success": True,
                "report": report
            }
            
        except Exception as e:
            logger.error(f"❌ Session end error: {e}")
            return {"success": False, "error": str(e)}
    
    # Specialized mode methods
    
    async def start_coding_session(
        self,
        session_id: str,
        problem_id: str,
        problem_statement: str,
        expected_approaches: List[str],
        difficulty: str = "medium",
        time_limit_minutes: int = 45,
        language: str = "python"
    ) -> Dict[str, Any]:
        """Start a live coding observation session"""
        if session_id not in self.sessions:
            return {"success": False, "error": "Session not found"}
        
        if not self.coding_observer:
            return {"success": False, "error": "Coding observer not available"}
        
        state = self.sessions[session_id]
        state.current_stage = ProcessingStage.CODING.value
        
        result = await self.coding_observer.start_session(
            session_id=session_id,
            candidate_id=state.candidate_id,
            problem_id=problem_id,
            problem_statement=problem_statement,
            expected_approaches=expected_approaches,
            difficulty=difficulty,
            time_limit_minutes=time_limit_minutes,
            language=language
        )
        
        self.sessions[session_id] = state
        return result
    
    async def update_code(
        self,
        session_id: str,
        code: str,
        time_elapsed_seconds: int
    ) -> Dict[str, Any]:
        """Update code in live coding session"""
        if not self.coding_observer:
            return {"success": False, "error": "Coding observer not available"}
        
        return await self.coding_observer.update_code(
            session_id=session_id,
            code=code,
            time_elapsed_seconds=time_elapsed_seconds
        )


# Singleton instance
_integration_layer = None

def get_integration_layer() -> AgenticAIIntegrationLayer:
    """Get singleton integration layer instance"""
    global _integration_layer
    if _integration_layer is None:
        _integration_layer = AgenticAIIntegrationLayer()
    return _integration_layer


# Convenience functions for common operations

async def initialize_agentic_interview(
    session_id: str,
    candidate_id: str,
    job_id: str,
    company_id: str,
    **kwargs
) -> Dict[str, Any]:
    """Initialize an agentic AI interview session"""
    layer = get_integration_layer()
    return await layer.initialize_session(
        session_id=session_id,
        candidate_id=candidate_id,
        job_id=job_id,
        company_id=company_id,
        **kwargs
    )


async def process_interview_response(
    session_id: str,
    answer: str,
    **kwargs
) -> Dict[str, Any]:
    """Process a candidate response"""
    layer = get_integration_layer()
    return await layer.process_response(
        session_id=session_id,
        answer=answer,
        **kwargs
    )


async def get_next_question(
    session_id: str,
    **kwargs
) -> Dict[str, Any]:
    """Get the next interview question"""
    layer = get_integration_layer()
    return await layer.generate_next_question(
        session_id=session_id,
        **kwargs
    )
