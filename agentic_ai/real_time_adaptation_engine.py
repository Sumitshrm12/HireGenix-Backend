"""
🎯 REAL-TIME ADAPTATION ENGINE - Dynamic Interview Adjustment System
Implements real-time feedback loop that adjusts questions based on emotional state,
cognitive load, and behavioral signals.

Features:
- Deep sensing integration for continuous monitoring
- Dynamic question difficulty adjustment
- Emotional state-aware response generation
- Cognitive load-based pacing control
- Stress detection with supportive interventions
- Confusion detection with clarification triggers
- Success recognition with challenge escalation

Tech Stack:
- LangGraph for reactive workflows
- Real-time signal processing
- Adaptive difficulty algorithms
- Emotional intelligence integration
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

# LangChain/LangGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionalState(str, Enum):
    """Detected emotional states"""
    CONFIDENT = "confident"
    NERVOUS = "nervous"
    CONFUSED = "confused"
    ENGAGED = "engaged"
    DISTRACTED = "distracted"
    STRUGGLING = "struggling"
    ENTHUSIASTIC = "enthusiastic"
    FRUSTRATED = "frustrated"
    CALM = "calm"
    ANXIOUS = "anxious"


class CognitiveLoad(str, Enum):
    """Cognitive load levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    OVERLOADED = "overloaded"


class AdaptationAction(str, Enum):
    """Actions to adapt the interview"""
    CONTINUE_NORMAL = "continue_normal"
    PROVIDE_ENCOURAGEMENT = "provide_encouragement"
    SIMPLIFY_QUESTION = "simplify_question"
    OFFER_CLARIFICATION = "offer_clarification"
    INCREASE_DIFFICULTY = "increase_difficulty"
    SLOW_DOWN_PACE = "slow_down_pace"
    TAKE_BRIEF_PAUSE = "take_brief_pause"
    SWITCH_TOPIC = "switch_topic"
    PROBE_DEEPER = "probe_deeper"
    PROVIDE_POSITIVE_FEEDBACK = "provide_positive_feedback"
    RE_ENGAGE_CANDIDATE = "re_engage_candidate"


@dataclass
class SignalSnapshot:
    """Snapshot of all current signals"""
    # Speech signals
    speech_rate_wpm: float = 130.0
    hesitation_count: int = 0
    pause_duration_avg: float = 0.0
    filler_word_frequency: float = 0.0
    voice_pitch_variance: float = 0.5
    
    # Visual signals
    gaze_on_screen_pct: float = 0.9
    expression: str = "neutral"
    posture_change_frequency: float = 0.0
    micro_expression_tension: float = 0.0
    
    # Response signals
    response_time_seconds: float = 0.0
    answer_length_words: int = 0
    technical_term_usage: float = 0.0
    answer_structure_score: float = 0.5
    
    # Semantic signals
    answer_confidence_markers: int = 0
    answer_uncertainty_markers: int = 0
    reformulation_count: int = 0
    
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "speech_rate_wpm": self.speech_rate_wpm,
            "hesitation_count": self.hesitation_count,
            "pause_duration_avg": self.pause_duration_avg,
            "filler_word_frequency": self.filler_word_frequency,
            "voice_pitch_variance": self.voice_pitch_variance,
            "gaze_on_screen_pct": self.gaze_on_screen_pct,
            "expression": self.expression,
            "posture_change_frequency": self.posture_change_frequency,
            "micro_expression_tension": self.micro_expression_tension,
            "response_time_seconds": self.response_time_seconds,
            "answer_length_words": self.answer_length_words,
            "technical_term_usage": self.technical_term_usage,
            "answer_structure_score": self.answer_structure_score,
            "answer_confidence_markers": self.answer_confidence_markers,
            "answer_uncertainty_markers": self.answer_uncertainty_markers,
            "reformulation_count": self.reformulation_count,
            "timestamp": self.timestamp or datetime.utcnow().isoformat()
        }


class AdaptationState(BaseModel):
    """State for adaptation workflow"""
    # Current signals
    current_signals: Dict[str, Any] = Field(default_factory=dict)
    signal_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Derived states
    emotional_state: EmotionalState = EmotionalState.CALM
    cognitive_load: CognitiveLoad = CognitiveLoad.MODERATE
    engagement_level: float = 0.7
    stress_level: float = 0.3
    confidence_level: float = 0.5
    
    # Trend analysis
    confidence_trend: str = "stable"  # rising, falling, stable
    engagement_trend: str = "stable"
    stress_trend: str = "stable"
    
    # Adaptation decisions
    recommended_actions: List[str] = Field(default_factory=list)
    primary_action: str = AdaptationAction.CONTINUE_NORMAL.value
    
    # Question modification
    difficulty_adjustment: float = 0.0  # -1 to +1
    tone_adjustment: str = "neutral"  # supportive, neutral, challenging
    pacing_adjustment: str = "normal"  # slower, normal, faster
    
    # Generated content
    encouragement_phrase: str = ""
    clarification_offer: str = ""
    transition_phrase: str = ""
    
    # Metadata
    analysis_confidence: float = 0.0
    last_update: str = ""


class RealTimeAdaptationEngine:
    """
    Real-time adaptation engine that continuously monitors candidate signals
    and adjusts interview parameters dynamically.
    """
    
    def __init__(self):
        logger.info("🎯 Initializing Real-Time Adaptation Engine...")
        
        # Azure OpenAI for intelligent adaptation
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.4
        )
        
        # Baseline thresholds
        self.SPEECH_RATE_BASELINE = 130  # words per minute
        self.HESITATION_THRESHOLD = 3
        self.PAUSE_THRESHOLD = 2.0  # seconds
        self.RESPONSE_TIME_THRESHOLD = 15.0  # seconds
        
        # Build adaptation workflow
        self.workflow = self._build_adaptation_workflow()
        
        # Encouragement phrases library
        self.encouragement_library = {
            "struggling": [
                "Take your time, there's no rush.",
                "That's a thoughtful approach. Feel free to think through it.",
                "I can see you're working through this carefully.",
                "It's okay to think out loud if that helps.",
            ],
            "nervous": [
                "You're doing great, keep going.",
                "I appreciate you sharing your perspective.",
                "That's a good point. Please continue.",
                "I can tell you have good experience with this.",
            ],
            "confused": [
                "Let me rephrase that question.",
                "Would you like me to break that down further?",
                "I may not have been clear. Here's another way to think about it.",
                "That's a complex topic. Let's approach it step by step.",
            ],
            "success": [
                "Excellent explanation!",
                "That shows deep understanding.",
                "Impressive approach to the problem.",
                "You clearly have strong experience here.",
            ]
        }
        
        logger.info("✅ Real-Time Adaptation Engine initialized")
    
    def _build_adaptation_workflow(self) -> StateGraph:
        """Build LangGraph workflow for real-time adaptation"""
        workflow = StateGraph(AdaptationState)
        
        # Define nodes
        workflow.add_node("analyze_signals", self._analyze_signals)
        workflow.add_node("detect_emotional_state", self._detect_emotional_state)
        workflow.add_node("assess_cognitive_load", self._assess_cognitive_load)
        workflow.add_node("analyze_trends", self._analyze_trends)
        workflow.add_node("determine_actions", self._determine_actions)
        workflow.add_node("generate_adaptations", self._generate_adaptations)
        
        # Define edges
        workflow.set_entry_point("analyze_signals")
        workflow.add_edge("analyze_signals", "detect_emotional_state")
        workflow.add_edge("detect_emotional_state", "assess_cognitive_load")
        workflow.add_edge("assess_cognitive_load", "analyze_trends")
        workflow.add_edge("analyze_trends", "determine_actions")
        workflow.add_edge("determine_actions", "generate_adaptations")
        workflow.add_edge("generate_adaptations", END)
        
        return workflow.compile()
    
    async def _analyze_signals(self, state: AdaptationState) -> AdaptationState:
        """Analyze raw signals and compute derived metrics"""
        signals = state.current_signals
        
        # Compute confidence level
        confidence_markers = signals.get("answer_confidence_markers", 0)
        uncertainty_markers = signals.get("answer_uncertainty_markers", 0)
        total_markers = confidence_markers + uncertainty_markers
        
        if total_markers > 0:
            state.confidence_level = confidence_markers / total_markers
        else:
            # Use other signals
            speech_rate = signals.get("speech_rate_wpm", self.SPEECH_RATE_BASELINE)
            hesitations = signals.get("hesitation_count", 0)
            
            rate_factor = 1.0 - abs(speech_rate - self.SPEECH_RATE_BASELINE) / self.SPEECH_RATE_BASELINE
            hesitation_factor = max(0, 1.0 - (hesitations / 5.0))
            
            state.confidence_level = (rate_factor + hesitation_factor) / 2
        
        # Compute engagement level
        gaze_on_screen = signals.get("gaze_on_screen_pct", 0.9)
        answer_length = signals.get("answer_length_words", 50)
        technical_usage = signals.get("technical_term_usage", 0.5)
        
        length_factor = min(answer_length / 100, 1.0)
        state.engagement_level = (gaze_on_screen * 0.4 + length_factor * 0.3 + technical_usage * 0.3)
        
        # Compute stress level
        expression = signals.get("expression", "neutral")
        pause_avg = signals.get("pause_duration_avg", 0.0)
        pitch_variance = signals.get("voice_pitch_variance", 0.5)
        
        expression_stress = {
            "neutral": 0.2,
            "happy": 0.1,
            "sad": 0.5,
            "fearful": 0.8,
            "angry": 0.6,
            "surprised": 0.4,
            "disgusted": 0.5
        }
        
        expr_factor = expression_stress.get(expression, 0.3)
        pause_factor = min(pause_avg / self.PAUSE_THRESHOLD, 1.0)
        pitch_factor = abs(pitch_variance - 0.5) * 2
        
        state.stress_level = (expr_factor * 0.4 + pause_factor * 0.3 + pitch_factor * 0.3)
        
        # Add to history
        state.signal_history.append({
            **signals,
            "confidence_level": state.confidence_level,
            "engagement_level": state.engagement_level,
            "stress_level": state.stress_level,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep only last 20 snapshots
        state.signal_history = state.signal_history[-20:]
        
        return state
    
    async def _detect_emotional_state(self, state: AdaptationState) -> AdaptationState:
        """Detect primary emotional state from analyzed signals"""
        confidence = state.confidence_level
        engagement = state.engagement_level
        stress = state.stress_level
        
        # Decision matrix for emotional state
        if stress > 0.7:
            if confidence < 0.3:
                state.emotional_state = EmotionalState.STRUGGLING
            else:
                state.emotional_state = EmotionalState.ANXIOUS
        elif stress > 0.5:
            if engagement < 0.4:
                state.emotional_state = EmotionalState.DISTRACTED
            else:
                state.emotional_state = EmotionalState.NERVOUS
        elif confidence > 0.7 and engagement > 0.7:
            state.emotional_state = EmotionalState.CONFIDENT
        elif confidence > 0.6 and engagement > 0.6:
            state.emotional_state = EmotionalState.ENTHUSIASTIC
        elif engagement > 0.6:
            state.emotional_state = EmotionalState.ENGAGED
        elif confidence < 0.4:
            state.emotional_state = EmotionalState.CONFUSED
        elif engagement < 0.4:
            state.emotional_state = EmotionalState.DISTRACTED
        else:
            state.emotional_state = EmotionalState.CALM
        
        return state
    
    async def _assess_cognitive_load(self, state: AdaptationState) -> AdaptationState:
        """Assess cognitive load from signals"""
        signals = state.current_signals
        
        # Cognitive load indicators
        response_time = signals.get("response_time_seconds", 5.0)
        reformulations = signals.get("reformulation_count", 0)
        hesitations = signals.get("hesitation_count", 0)
        pause_avg = signals.get("pause_duration_avg", 0.0)
        
        # Compute load score
        time_load = min(response_time / self.RESPONSE_TIME_THRESHOLD, 1.0)
        reformulation_load = min(reformulations / 3.0, 1.0)
        hesitation_load = min(hesitations / 5.0, 1.0)
        pause_load = min(pause_avg / self.PAUSE_THRESHOLD, 1.0)
        
        load_score = (time_load * 0.3 + reformulation_load * 0.25 + 
                      hesitation_load * 0.25 + pause_load * 0.2)
        
        if load_score > 0.8:
            state.cognitive_load = CognitiveLoad.OVERLOADED
        elif load_score > 0.6:
            state.cognitive_load = CognitiveLoad.HIGH
        elif load_score > 0.35:
            state.cognitive_load = CognitiveLoad.MODERATE
        else:
            state.cognitive_load = CognitiveLoad.LOW
        
        return state
    
    async def _analyze_trends(self, state: AdaptationState) -> AdaptationState:
        """Analyze trends across signal history"""
        history = state.signal_history
        
        if len(history) < 3:
            state.confidence_trend = "stable"
            state.engagement_trend = "stable"
            state.stress_trend = "stable"
            return state
        
        # Get recent values
        recent = history[-3:]
        
        # Analyze confidence trend
        conf_values = [h.get("confidence_level", 0.5) for h in recent]
        conf_change = conf_values[-1] - conf_values[0]
        if conf_change > 0.15:
            state.confidence_trend = "rising"
        elif conf_change < -0.15:
            state.confidence_trend = "falling"
        else:
            state.confidence_trend = "stable"
        
        # Analyze engagement trend
        eng_values = [h.get("engagement_level", 0.5) for h in recent]
        eng_change = eng_values[-1] - eng_values[0]
        if eng_change > 0.15:
            state.engagement_trend = "rising"
        elif eng_change < -0.15:
            state.engagement_trend = "falling"
        else:
            state.engagement_trend = "stable"
        
        # Analyze stress trend
        stress_values = [h.get("stress_level", 0.3) for h in recent]
        stress_change = stress_values[-1] - stress_values[0]
        if stress_change > 0.15:
            state.stress_trend = "rising"
        elif stress_change < -0.15:
            state.stress_trend = "falling"
        else:
            state.stress_trend = "stable"
        
        return state
    
    async def _determine_actions(self, state: AdaptationState) -> AdaptationState:
        """Determine adaptation actions based on analysis"""
        actions = []
        
        # Primary emotional state actions
        emotional_actions = {
            EmotionalState.STRUGGLING: [
                AdaptationAction.SIMPLIFY_QUESTION,
                AdaptationAction.PROVIDE_ENCOURAGEMENT,
                AdaptationAction.SLOW_DOWN_PACE
            ],
            EmotionalState.NERVOUS: [
                AdaptationAction.PROVIDE_ENCOURAGEMENT,
                AdaptationAction.SLOW_DOWN_PACE
            ],
            EmotionalState.CONFUSED: [
                AdaptationAction.OFFER_CLARIFICATION,
                AdaptationAction.SIMPLIFY_QUESTION
            ],
            EmotionalState.DISTRACTED: [
                AdaptationAction.RE_ENGAGE_CANDIDATE,
                AdaptationAction.SWITCH_TOPIC
            ],
            EmotionalState.CONFIDENT: [
                AdaptationAction.INCREASE_DIFFICULTY,
                AdaptationAction.PROBE_DEEPER
            ],
            EmotionalState.ENTHUSIASTIC: [
                AdaptationAction.PROBE_DEEPER,
                AdaptationAction.PROVIDE_POSITIVE_FEEDBACK
            ],
            EmotionalState.ANXIOUS: [
                AdaptationAction.TAKE_BRIEF_PAUSE,
                AdaptationAction.PROVIDE_ENCOURAGEMENT
            ],
            EmotionalState.FRUSTRATED: [
                AdaptationAction.SWITCH_TOPIC,
                AdaptationAction.TAKE_BRIEF_PAUSE
            ]
        }
        
        actions.extend(emotional_actions.get(state.emotional_state, [AdaptationAction.CONTINUE_NORMAL]))
        
        # Cognitive load actions
        if state.cognitive_load == CognitiveLoad.OVERLOADED:
            actions.append(AdaptationAction.TAKE_BRIEF_PAUSE)
            actions.append(AdaptationAction.SIMPLIFY_QUESTION)
        elif state.cognitive_load == CognitiveLoad.HIGH:
            actions.append(AdaptationAction.SLOW_DOWN_PACE)
        
        # Trend-based actions
        if state.confidence_trend == "falling":
            actions.append(AdaptationAction.PROVIDE_ENCOURAGEMENT)
        if state.engagement_trend == "falling":
            actions.append(AdaptationAction.RE_ENGAGE_CANDIDATE)
        if state.stress_trend == "rising":
            actions.append(AdaptationAction.SLOW_DOWN_PACE)
        
        # Deduplicate and prioritize
        unique_actions = list(dict.fromkeys(actions))
        state.recommended_actions = [a.value for a in unique_actions[:4]]
        state.primary_action = unique_actions[0].value if unique_actions else AdaptationAction.CONTINUE_NORMAL.value
        
        # Set adjustments
        if AdaptationAction.SIMPLIFY_QUESTION in unique_actions:
            state.difficulty_adjustment = -0.3
        elif AdaptationAction.INCREASE_DIFFICULTY in unique_actions:
            state.difficulty_adjustment = 0.3
        
        if AdaptationAction.PROVIDE_ENCOURAGEMENT in unique_actions:
            state.tone_adjustment = "supportive"
        elif AdaptationAction.PROBE_DEEPER in unique_actions:
            state.tone_adjustment = "challenging"
        
        if AdaptationAction.SLOW_DOWN_PACE in unique_actions:
            state.pacing_adjustment = "slower"
        
        return state
    
    async def _generate_adaptations(self, state: AdaptationState) -> AdaptationState:
        """Generate specific phrases and adaptations"""
        import random
        
        # Generate encouragement phrase if needed
        if AdaptationAction.PROVIDE_ENCOURAGEMENT.value in state.recommended_actions:
            if state.emotional_state == EmotionalState.STRUGGLING:
                state.encouragement_phrase = random.choice(self.encouragement_library["struggling"])
            elif state.emotional_state == EmotionalState.NERVOUS:
                state.encouragement_phrase = random.choice(self.encouragement_library["nervous"])
            else:
                state.encouragement_phrase = "You're doing well. Please continue."
        
        # Generate clarification offer if needed
        if AdaptationAction.OFFER_CLARIFICATION.value in state.recommended_actions:
            state.clarification_offer = random.choice(self.encouragement_library["confused"])
        
        # Generate transition phrase if switching topics
        if AdaptationAction.SWITCH_TOPIC.value in state.recommended_actions:
            state.transition_phrase = "Let's move on to something different."
        
        # Generate positive feedback if doing well
        if AdaptationAction.PROVIDE_POSITIVE_FEEDBACK.value in state.recommended_actions:
            state.encouragement_phrase = random.choice(self.encouragement_library["success"])
        
        # Compute analysis confidence
        history_length = len(state.signal_history)
        state.analysis_confidence = min(history_length / 10.0, 1.0)
        
        state.last_update = datetime.utcnow().isoformat()
        
        return state
    
    async def process_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Process a signal snapshot and return adaptation recommendations"""
        try:
            initial_state = AdaptationState(
                current_signals=signals
            )
            
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "success": True,
                "emotional_state": final_state.emotional_state.value,
                "cognitive_load": final_state.cognitive_load.value,
                "confidence_level": final_state.confidence_level,
                "engagement_level": final_state.engagement_level,
                "stress_level": final_state.stress_level,
                "trends": {
                    "confidence": final_state.confidence_trend,
                    "engagement": final_state.engagement_trend,
                    "stress": final_state.stress_trend
                },
                "actions": {
                    "primary": final_state.primary_action,
                    "recommended": final_state.recommended_actions
                },
                "adjustments": {
                    "difficulty": final_state.difficulty_adjustment,
                    "tone": final_state.tone_adjustment,
                    "pacing": final_state.pacing_adjustment
                },
                "phrases": {
                    "encouragement": final_state.encouragement_phrase,
                    "clarification": final_state.clarification_offer,
                    "transition": final_state.transition_phrase
                },
                "analysis_confidence": final_state.analysis_confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Signal processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "actions": {"primary": "continue_normal"},
                "adjustments": {"difficulty": 0, "tone": "neutral", "pacing": "normal"}
            }
    
    async def get_question_modifier(
        self,
        base_question: str,
        adaptation_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate modified question based on adaptation analysis"""
        try:
            adjustments = adaptation_result.get("adjustments", {})
            phrases = adaptation_result.get("phrases", {})
            emotional_state = adaptation_result.get("emotional_state", "calm")
            
            # Build prompt for question modification
            prompt = f"""
            You are an expert interviewer adjusting your approach based on candidate state.
            
            BASE QUESTION: {base_question}
            
            CANDIDATE STATE:
            - Emotional State: {emotional_state}
            - Cognitive Load: {adaptation_result.get("cognitive_load", "moderate")}
            - Confidence Level: {adaptation_result.get("confidence_level", 0.5):.2f}
            - Stress Level: {adaptation_result.get("stress_level", 0.3):.2f}
            
            ADJUSTMENTS NEEDED:
            - Difficulty: {adjustments.get("difficulty", 0)} (-1 = easier, +1 = harder)
            - Tone: {adjustments.get("tone", "neutral")}
            - Pacing: {adjustments.get("pacing", "normal")}
            
            PRE-PHRASE TO USE: {phrases.get("encouragement", "")} {phrases.get("clarification", "")}
            
            Generate a modified question that:
            1. Maintains the same intent but adjusts complexity if needed
            2. Incorporates any encouragement/clarification naturally
            3. Sounds human and conversational
            4. Matches the tone adjustment
            
            Return JSON:
            {{
                "modified_question": "The full modified question with any pre-phrases",
                "changes_made": ["list of changes"],
                "rationale": "why these changes help"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            return {
                "success": True,
                "original_question": base_question,
                "modified_question": result.get("modified_question", base_question),
                "changes_made": result.get("changes_made", []),
                "rationale": result.get("rationale", ""),
                "adaptation_applied": True
            }
            
        except Exception as e:
            logger.error(f"❌ Question modification error: {e}")
            # Return original question if modification fails
            pre_phrase = adaptation_result.get("phrases", {}).get("encouragement", "")
            return {
                "success": False,
                "original_question": base_question,
                "modified_question": f"{pre_phrase} {base_question}".strip() if pre_phrase else base_question,
                "error": str(e),
                "adaptation_applied": False
            }


# Singleton instance
_adaptation_engine = None

def get_adaptation_engine() -> RealTimeAdaptationEngine:
    """Get singleton adaptation engine instance"""
    global _adaptation_engine
    if _adaptation_engine is None:
        _adaptation_engine = RealTimeAdaptationEngine()
    return _adaptation_engine
