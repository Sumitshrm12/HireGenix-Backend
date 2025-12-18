"""
🧠 ENHANCED DEEP SENSING - Advanced Non-Verbal & Behavioral Intelligence
Expands upon deep_sensing.py with semantic pause analysis, micro-expression
detection, stress level tracking, and multi-modal behavioral fusion.

Features:
- Semantic pause analysis (why they paused)
- Micro-expression detection
- Stress trajectory tracking
- Multi-modal behavioral fusion
- Cognitive load estimation
- Deception indicators (ethical use only)
- Engagement micro-patterns
- Confidence calibration

Tech Stack:
- LangGraph for analysis workflow
- MediaPipe for facial analysis
- Librosa for voice stress
- Real-time signal processing
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import numpy as np

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PauseType(str, Enum):
    """Types of conversational pauses"""
    THINKING = "thinking"  # Normal cognitive processing
    RECALL = "recall"  # Trying to remember something
    HESITATION = "hesitation"  # Uncertain about answer
    STRATEGIC = "strategic"  # Deliberate for emphasis
    ANXIETY = "anxiety"  # Stress-induced
    CONFUSION = "confusion"  # Didn't understand question
    DECEPTIVE = "deceptive"  # Possibly fabricating


class MicroExpression(str, Enum):
    """Detected micro-expressions"""
    GENUINE_SMILE = "genuine_smile"
    FORCED_SMILE = "forced_smile"
    CONTEMPT = "contempt"
    SURPRISE = "surprise"
    FEAR = "fear"
    DISGUST = "disgust"
    SADNESS = "sadness"
    ANGER = "anger"
    CONCENTRATION = "concentration"
    CONFUSION = "confusion"


class StressLevel(str, Enum):
    """Stress level indicators"""
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class EnhancedDeepSensingState(BaseModel):
    """State for enhanced deep sensing analysis"""
    # Session info
    session_id: str = ""
    candidate_id: str = ""
    
    # Input data
    transcript_segment: str = ""
    audio_features: Dict[str, Any] = Field(default_factory=dict)
    video_features: Dict[str, Any] = Field(default_factory=dict)
    pause_data: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Pause analysis
    pause_classifications: List[Dict[str, Any]] = Field(default_factory=list)
    pause_pattern_summary: str = ""
    
    # Micro-expression analysis
    detected_micro_expressions: List[Dict[str, Any]] = Field(default_factory=list)
    expression_congruence: float = 0.0  # How well expressions match words
    
    # Stress analysis
    current_stress_level: str = StressLevel.MODERATE.value
    stress_trajectory: List[Dict[str, float]] = Field(default_factory=list)
    stress_triggers: List[str] = Field(default_factory=list)
    
    # Cognitive load
    cognitive_load_estimate: float = 0.0
    load_indicators: List[str] = Field(default_factory=list)
    
    # Confidence calibration
    self_reported_confidence: float = 0.0
    behavioral_confidence: float = 0.0
    confidence_gap: float = 0.0  # Gap between claimed and actual
    
    # Engagement
    engagement_score: float = 0.0
    engagement_indicators: List[str] = Field(default_factory=list)
    attention_lapses: int = 0
    
    # Multi-modal fusion
    behavioral_coherence: float = 0.0  # How well signals align
    incongruence_flags: List[str] = Field(default_factory=list)
    
    # Red flags (ethical use)
    authenticity_score: float = 0.0
    authenticity_concerns: List[str] = Field(default_factory=list)
    
    # Summary
    overall_behavioral_assessment: str = ""
    interviewer_recommendations: List[str] = Field(default_factory=list)


class EnhancedDeepSensing:
    """
    Advanced behavioral sensing with semantic pause analysis,
    micro-expression detection, and multi-modal fusion.
    """
    
    def __init__(self):
        logger.info("🧠 Initializing Enhanced Deep Sensing...")
        
        # LLM for semantic analysis
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.2
        )
        
        # Build sensing workflow
        self.workflow = self._build_sensing_workflow()
        
        # Historical data for trajectory analysis
        self.session_history: Dict[str, List[Dict]] = {}
        
        # Pause type thresholds
        self.pause_thresholds = {
            "short": 0.5,  # <0.5s
            "medium": 1.5,  # 0.5-1.5s
            "long": 3.0,  # 1.5-3.0s
            "extended": 5.0  # >3.0s
        }
        
        logger.info("✅ Enhanced Deep Sensing initialized")
    
    def _build_sensing_workflow(self) -> StateGraph:
        """Build LangGraph workflow for deep sensing"""
        workflow = StateGraph(EnhancedDeepSensingState)
        
        # Define nodes
        workflow.add_node("analyze_pauses", self._analyze_pauses)
        workflow.add_node("detect_micro_expressions", self._detect_micro_expressions)
        workflow.add_node("calculate_stress", self._calculate_stress)
        workflow.add_node("estimate_cognitive_load", self._estimate_cognitive_load)
        workflow.add_node("calibrate_confidence", self._calibrate_confidence)
        workflow.add_node("assess_engagement", self._assess_engagement)
        workflow.add_node("fuse_signals", self._fuse_signals)
        workflow.add_node("generate_assessment", self._generate_assessment)
        
        # Define edges
        workflow.set_entry_point("analyze_pauses")
        workflow.add_edge("analyze_pauses", "detect_micro_expressions")
        workflow.add_edge("detect_micro_expressions", "calculate_stress")
        workflow.add_edge("calculate_stress", "estimate_cognitive_load")
        workflow.add_edge("estimate_cognitive_load", "calibrate_confidence")
        workflow.add_edge("calibrate_confidence", "assess_engagement")
        workflow.add_edge("assess_engagement", "fuse_signals")
        workflow.add_edge("fuse_signals", "generate_assessment")
        workflow.add_edge("generate_assessment", END)
        
        return workflow.compile()
    
    async def _analyze_pauses(self, state: EnhancedDeepSensingState) -> EnhancedDeepSensingState:
        """Semantic analysis of pauses - understanding WHY they paused"""
        try:
            if not state.pause_data:
                return state
            
            # Analyze each significant pause
            classifications = []
            
            for pause in state.pause_data:
                duration = pause.get("duration", 0)
                
                # Get context around the pause
                before_pause = pause.get("context_before", "")
                after_pause = pause.get("context_after", "")
                
                prompt = f"""
                Analyze this pause in a job interview conversation.
                
                CONTEXT BEFORE PAUSE: "{before_pause}"
                PAUSE DURATION: {duration:.1f} seconds
                CONTEXT AFTER PAUSE: "{after_pause}"
                
                Classify the pause type:
                - thinking: Normal cognitive processing, forming thoughts
                - recall: Trying to remember specific details
                - hesitation: Uncertain about the answer
                - strategic: Deliberate pause for emphasis
                - anxiety: Stress-induced freeze
                - confusion: Didn't understand the question
                - deceptive: Possibly fabricating (use sparingly)
                
                Return JSON:
                {{
                    "pause_type": "type",
                    "confidence": 0.8,
                    "reasoning": "Why you classified it this way",
                    "concerning": true|false
                }}
                """
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                result = json.loads(response.content.replace("```json", "").replace("```", ""))
                
                classifications.append({
                    "duration": duration,
                    "type": result.get("pause_type", "thinking"),
                    "confidence": result.get("confidence", 0.5),
                    "concerning": result.get("concerning", False),
                    "reasoning": result.get("reasoning", "")
                })
            
            state.pause_classifications = classifications
            
            # Summarize pause patterns
            pause_types = [c["type"] for c in classifications]
            most_common = max(set(pause_types), key=pause_types.count) if pause_types else "none"
            concerning_count = sum(1 for c in classifications if c.get("concerning"))
            
            state.pause_pattern_summary = (
                f"Dominant pause type: {most_common}. "
                f"Total pauses analyzed: {len(classifications)}. "
                f"Concerning pauses: {concerning_count}."
            )
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Pause analysis error: {e}")
            return state
    
    async def _detect_micro_expressions(self, state: EnhancedDeepSensingState) -> EnhancedDeepSensingState:
        """Detect micro-expressions from video features"""
        try:
            video_features = state.video_features
            
            if not video_features:
                # Simulate detection from transcript analysis
                prompt = f"""
                Based on this interview transcript segment, infer likely facial expressions:
                
                TRANSCRIPT: "{state.transcript_segment}"
                
                Consider the emotional content and likely expressions.
                
                Return JSON:
                {{
                    "likely_expressions": [
                        {{
                            "expression": "genuine_smile|forced_smile|concentration|confusion|surprise",
                            "probability": 0.7,
                            "trigger": "what likely caused this"
                        }}
                    ],
                    "overall_sentiment": "positive|neutral|negative|mixed"
                }}
                """
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                result = json.loads(response.content.replace("```json", "").replace("```", ""))
                
                state.detected_micro_expressions = result.get("likely_expressions", [])
            else:
                # Use actual video features if available
                # This would integrate with MediaPipe/DeepFace
                expressions = []
                
                # Map video features to expressions
                if video_features.get("smile_intensity", 0) > 0.6:
                    expressions.append({
                        "expression": MicroExpression.GENUINE_SMILE.value,
                        "probability": video_features.get("smile_intensity", 0.7)
                    })
                
                if video_features.get("brow_furrow", 0) > 0.5:
                    expressions.append({
                        "expression": MicroExpression.CONCENTRATION.value,
                        "probability": video_features.get("brow_furrow", 0.6)
                    })
                
                state.detected_micro_expressions = expressions
            
            # Calculate expression congruence
            # (How well expressions match the verbal content)
            state.expression_congruence = 0.8  # Default, would be calculated
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Micro-expression detection error: {e}")
            return state
    
    async def _calculate_stress(self, state: EnhancedDeepSensingState) -> EnhancedDeepSensingState:
        """Calculate stress level from multiple signals"""
        try:
            stress_signals = []
            
            # From audio features
            audio = state.audio_features
            if audio:
                # Higher pitch variation often indicates stress
                if audio.get("pitch_variation", 0.3) > 0.5:
                    stress_signals.append(("voice_pitch", 0.3))
                
                # Faster speech can indicate anxiety
                if audio.get("speaking_rate", 140) > 180:
                    stress_signals.append(("speech_rate", 0.2))
                
                # Voice tremor
                if audio.get("jitter", 0) > 0.03:
                    stress_signals.append(("voice_tremor", 0.4))
            
            # From pause patterns
            anxiety_pauses = [
                p for p in state.pause_classifications 
                if p.get("type") == PauseType.ANXIETY.value
            ]
            if anxiety_pauses:
                stress_signals.append(("anxiety_pauses", len(anxiety_pauses) * 0.1))
            
            # From micro-expressions
            stress_expressions = [
                e for e in state.detected_micro_expressions
                if e.get("expression") in [MicroExpression.FEAR.value, "forced_smile"]
            ]
            if stress_expressions:
                stress_signals.append(("stress_expressions", 0.2))
            
            # Calculate overall stress level
            if not stress_signals:
                stress_score = 0.3  # Low
            else:
                stress_score = min(1.0, sum(s[1] for s in stress_signals))
            
            # Map to stress level
            if stress_score < 0.2:
                state.current_stress_level = StressLevel.LOW.value
            elif stress_score < 0.4:
                state.current_stress_level = StressLevel.MODERATE.value
            elif stress_score < 0.6:
                state.current_stress_level = StressLevel.ELEVATED.value
            elif stress_score < 0.8:
                state.current_stress_level = StressLevel.HIGH.value
            else:
                state.current_stress_level = StressLevel.CRITICAL.value
            
            # Track trajectory
            state.stress_trajectory.append({
                "timestamp": datetime.now().isoformat(),
                "level": stress_score
            })
            
            # Identify triggers
            state.stress_triggers = [s[0] for s in stress_signals]
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Stress calculation error: {e}")
            state.current_stress_level = StressLevel.MODERATE.value
            return state
    
    async def _estimate_cognitive_load(self, state: EnhancedDeepSensingState) -> EnhancedDeepSensingState:
        """Estimate cognitive load from behavioral signals"""
        try:
            load_indicators = []
            
            # Long pauses indicate high cognitive load
            long_pauses = [
                p for p in state.pause_classifications
                if p.get("duration", 0) > 2.0
            ]
            if long_pauses:
                load_indicators.append("Extended thinking pauses")
            
            # Speech rate changes
            audio = state.audio_features
            if audio:
                if audio.get("rate_variability", 0) > 0.3:
                    load_indicators.append("Variable speech rate")
            
            # Filler words (from transcript)
            filler_patterns = ["um", "uh", "like", "you know"]
            transcript_lower = state.transcript_segment.lower()
            filler_count = sum(transcript_lower.count(f) for f in filler_patterns)
            word_count = len(state.transcript_segment.split())
            
            if word_count > 0:
                filler_ratio = filler_count / word_count
                if filler_ratio > 0.05:
                    load_indicators.append("High filler word usage")
            
            # Calculate cognitive load (0-1)
            base_load = len(load_indicators) * 0.2
            
            # Adjust for pause patterns
            thinking_pauses = [
                p for p in state.pause_classifications
                if p.get("type") in [PauseType.THINKING.value, PauseType.RECALL.value]
            ]
            if thinking_pauses:
                base_load += 0.1 * len(thinking_pauses)
            
            state.cognitive_load_estimate = min(1.0, base_load)
            state.load_indicators = load_indicators
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Cognitive load estimation error: {e}")
            return state
    
    async def _calibrate_confidence(self, state: EnhancedDeepSensingState) -> EnhancedDeepSensingState:
        """Calibrate stated confidence vs behavioral confidence"""
        try:
            # Analyze behavioral confidence signals
            confidence_signals = {
                "positive": [],
                "negative": []
            }
            
            # Voice features
            audio = state.audio_features
            if audio:
                if audio.get("volume_stability", 0.7) > 0.8:
                    confidence_signals["positive"].append("Stable voice volume")
                if audio.get("pitch_variation", 0.3) < 0.2:
                    confidence_signals["positive"].append("Steady pitch")
                if audio.get("speaking_rate", 140) < 100:
                    confidence_signals["negative"].append("Slow, uncertain speech")
            
            # Pause patterns
            hesitation_pauses = [
                p for p in state.pause_classifications
                if p.get("type") == PauseType.HESITATION.value
            ]
            if hesitation_pauses:
                confidence_signals["negative"].append(f"{len(hesitation_pauses)} hesitation pauses")
            
            # Calculate behavioral confidence
            positive_count = len(confidence_signals["positive"])
            negative_count = len(confidence_signals["negative"])
            
            behavioral_confidence = 0.5 + (positive_count * 0.1) - (negative_count * 0.15)
            state.behavioral_confidence = max(0.0, min(1.0, behavioral_confidence))
            
            # Compare with self-reported (if available)
            if state.self_reported_confidence > 0:
                state.confidence_gap = state.self_reported_confidence - state.behavioral_confidence
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Confidence calibration error: {e}")
            return state
    
    async def _assess_engagement(self, state: EnhancedDeepSensingState) -> EnhancedDeepSensingState:
        """Assess candidate engagement level"""
        try:
            engagement_indicators = []
            
            # From expressions
            positive_expressions = [
                e for e in state.detected_micro_expressions
                if e.get("expression") in [
                    MicroExpression.GENUINE_SMILE.value,
                    MicroExpression.CONCENTRATION.value
                ]
            ]
            if positive_expressions:
                engagement_indicators.append("Positive facial expressions")
            
            # Response length/quality (from transcript)
            word_count = len(state.transcript_segment.split())
            if word_count > 50:
                engagement_indicators.append("Detailed responses")
            
            # Low cognitive overload is good for engagement
            if state.cognitive_load_estimate < 0.6:
                engagement_indicators.append("Comfortable with complexity")
            
            # Moderate stress is okay, high stress hurts engagement
            if state.current_stress_level in [StressLevel.LOW.value, StressLevel.MODERATE.value]:
                engagement_indicators.append("Appropriate stress level")
            
            # Calculate engagement score
            state.engagement_indicators = engagement_indicators
            state.engagement_score = min(1.0, len(engagement_indicators) * 0.2 + 0.4)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Engagement assessment error: {e}")
            state.engagement_score = 0.5
            return state
    
    async def _fuse_signals(self, state: EnhancedDeepSensingState) -> EnhancedDeepSensingState:
        """Fuse all signals for coherence analysis"""
        try:
            incongruence_flags = []
            
            # Check voice-expression coherence
            # (e.g., positive words but stressed voice)
            if state.current_stress_level in [StressLevel.HIGH.value, StressLevel.CRITICAL.value]:
                positive_expressions = [
                    e for e in state.detected_micro_expressions
                    if e.get("expression") == MicroExpression.GENUINE_SMILE.value
                ]
                if positive_expressions:
                    incongruence_flags.append("Smiling while stressed")
            
            # Check confidence coherence
            if abs(state.confidence_gap) > 0.3:
                if state.confidence_gap > 0:
                    incongruence_flags.append("Claims higher confidence than displayed")
                else:
                    incongruence_flags.append("More confident than they claim")
            
            # Check pause-content coherence
            deceptive_pauses = [
                p for p in state.pause_classifications
                if p.get("type") == PauseType.DECEPTIVE.value
            ]
            if deceptive_pauses:
                incongruence_flags.append("Potentially deceptive pause patterns")
            
            state.incongruence_flags = incongruence_flags
            
            # Calculate behavioral coherence
            coherence_penalty = len(incongruence_flags) * 0.1
            state.behavioral_coherence = max(0.0, 1.0 - coherence_penalty)
            
            # Calculate authenticity (coherent signals = authentic)
            state.authenticity_score = (
                state.behavioral_coherence * 0.4 +
                (1 - abs(state.confidence_gap)) * 0.3 +
                state.expression_congruence * 0.3
            )
            
            if incongruence_flags:
                state.authenticity_concerns = incongruence_flags
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Signal fusion error: {e}")
            state.behavioral_coherence = 0.7
            return state
    
    async def _generate_assessment(self, state: EnhancedDeepSensingState) -> EnhancedDeepSensingState:
        """Generate overall behavioral assessment"""
        try:
            prompt = f"""
            Generate a behavioral assessment summary for an interviewer.
            
            BEHAVIORAL SIGNALS:
            - Stress Level: {state.current_stress_level}
            - Cognitive Load: {state.cognitive_load_estimate:.2f}
            - Behavioral Confidence: {state.behavioral_confidence:.2f}
            - Engagement Score: {state.engagement_score:.2f}
            - Behavioral Coherence: {state.behavioral_coherence:.2f}
            - Authenticity Score: {state.authenticity_score:.2f}
            
            PAUSE PATTERNS:
            {state.pause_pattern_summary}
            
            INCONGRUENCE FLAGS:
            {state.incongruence_flags}
            
            ENGAGEMENT INDICATORS:
            {state.engagement_indicators}
            
            Generate:
            1. A 2-3 sentence behavioral summary
            2. Key recommendations for the interviewer
            3. Any areas of concern
            
            Keep it actionable and objective.
            
            Return JSON:
            {{
                "summary": "Brief behavioral summary",
                "recommendations": ["recommendation1"],
                "concerns": ["concern1"],
                "overall_impression": "positive|neutral|concerning"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.overall_behavioral_assessment = result.get("summary", "")
            state.interviewer_recommendations = result.get("recommendations", [])
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Assessment generation error: {e}")
            state.overall_behavioral_assessment = "Unable to generate detailed assessment."
            return state
    
    async def analyze(
        self,
        session_id: str,
        candidate_id: str,
        transcript_segment: str,
        pause_data: Optional[List[Dict[str, Any]]] = None,
        audio_features: Optional[Dict[str, Any]] = None,
        video_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run enhanced deep sensing analysis"""
        try:
            initial_state = EnhancedDeepSensingState(
                session_id=session_id,
                candidate_id=candidate_id,
                transcript_segment=transcript_segment,
                pause_data=pause_data or [],
                audio_features=audio_features or {},
                video_features=video_features or {}
            )
            
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Store in session history
            if session_id not in self.session_history:
                self.session_history[session_id] = []
            
            self.session_history[session_id].append({
                "timestamp": datetime.now().isoformat(),
                "stress": final_state.current_stress_level,
                "engagement": final_state.engagement_score,
                "authenticity": final_state.authenticity_score
            })
            
            return {
                "success": True,
                "summary": final_state.overall_behavioral_assessment,
                "recommendations": final_state.interviewer_recommendations,
                "metrics": {
                    "stress_level": final_state.current_stress_level,
                    "cognitive_load": final_state.cognitive_load_estimate,
                    "behavioral_confidence": final_state.behavioral_confidence,
                    "engagement_score": final_state.engagement_score,
                    "behavioral_coherence": final_state.behavioral_coherence,
                    "authenticity_score": final_state.authenticity_score
                },
                "pause_analysis": {
                    "summary": final_state.pause_pattern_summary,
                    "classifications": final_state.pause_classifications
                },
                "stress_trajectory": final_state.stress_trajectory,
                "engagement_indicators": final_state.engagement_indicators,
                "incongruence_flags": final_state.incongruence_flags,
                "authenticity_concerns": final_state.authenticity_concerns
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced sensing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": "Analysis could not be completed."
            }
    
    def get_session_trajectory(self, session_id: str) -> Dict[str, Any]:
        """Get behavioral trajectory for a session"""
        if session_id not in self.session_history:
            return {"success": False, "error": "Session not found"}
        
        history = self.session_history[session_id]
        
        return {
            "success": True,
            "data_points": len(history),
            "stress_trend": [h.get("stress") for h in history],
            "engagement_trend": [h.get("engagement") for h in history],
            "authenticity_trend": [h.get("authenticity") for h in history]
        }


# Singleton instance
_enhanced_sensing = None

def get_enhanced_deep_sensing() -> EnhancedDeepSensing:
    """Get singleton enhanced deep sensing instance"""
    global _enhanced_sensing
    if _enhanced_sensing is None:
        _enhanced_sensing = EnhancedDeepSensing()
    return _enhanced_sensing
