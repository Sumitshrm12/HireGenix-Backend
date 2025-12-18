"""
Deep Sensing Service
Analyzes non-verbal cues (audio/visual metadata) and behavioral patterns to assess 
candidate confidence, engagement, and stress levels.

Features:
- Speech Pattern Analysis (Hesitation, Rate, Clarity)
- Visual Engagement Tracking (Gaze, Expression metadata processing)
- Behavioral State Inference (Confident, Nervous, Distracted)
- Real-time Feedback Generation for the Interview Agent
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehavioralState(str, Enum):
    CONFIDENT = "confident"
    NERVOUS = "nervous"
    DISTRACTED = "distracted"
    ENGAGED = "engaged"
    HESITANT = "hesitant"
    NEUTRAL = "neutral"

@dataclass
class NonVerbalSignals:
    """Raw signals received from frontend/media processor"""
    speech_rate_wpm: float  # Words per minute
    hesitation_count: int   # Number of filler words (um, uh, like)
    pause_duration_avg: float # Average pause duration in seconds
    gaze_off_screen_pct: float # Percentage of time looking away
    dominant_expression: str # happy, neutral, fearful, sad, angry
    voice_pitch_variance: float # Proxy for monotone vs expressive

@dataclass
class BehavioralInsight:
    """Derived insight from signals"""
    state: BehavioralState
    confidence_score: float # 0.0 to 1.0
    engagement_score: float # 0.0 to 1.0
    stress_level: float # 0.0 to 1.0
    observation: str # Human-readable observation

class DeepSensingService:
    """
    Interprets multi-modal signals to provide behavioral intelligence.
    """
    
    def __init__(self):
        # Baselines (could be calibrated per candidate in a real scenario)
        self.BASELINE_WPM = 130.0
        self.BASELINE_HESITATION_RATE = 2.0 # per minute
        self.MAX_GAZE_OFF_PCT = 15.0
        
    def analyze_signals(self, signals: Dict[str, Any]) -> BehavioralInsight:
        """
        Process raw metadata signals into behavioral insights.
        
        Args:
            signals: Dictionary containing signal data
                - speech_rate_wpm (float)
                - hesitation_count (int)
                - gaze_off_screen_pct (float)
                - dominant_expression (str)
                
        Returns:
            BehavioralInsight object
        """
        try:
            # Extract signals with defaults
            wpm = signals.get('speech_rate_wpm', self.BASELINE_WPM)
            hesitations = signals.get('hesitation_count', 0)
            gaze_off = signals.get('gaze_off_screen_pct', 0.0)
            expression = signals.get('dominant_expression', 'neutral')
            
            # 1. Calculate Confidence Score
            # Factors: Speech rate (too slow/fast is bad), hesitations (fewer is better)
            confidence_score = 1.0
            
            # Penalize for hesitations
            if hesitations > 3:
                confidence_score -= 0.1 * (hesitations - 3)
            
            # Penalize for extreme speech rates
            if wpm < 100: # Too slow
                confidence_score -= 0.2
            elif wpm > 180: # Too fast (nervousness)
                confidence_score -= 0.1
                
            confidence_score = max(0.1, min(1.0, confidence_score))
            
            # 2. Calculate Engagement Score
            # Factors: Gaze (looking at screen), Expression (positive/neutral vs bored)
            engagement_score = 1.0
            
            if gaze_off > self.MAX_GAZE_OFF_PCT:
                engagement_score -= (gaze_off - self.MAX_GAZE_OFF_PCT) / 100.0
            
            if expression in ['bored', 'sleepy']:
                engagement_score -= 0.3
            elif expression in ['happy', 'surprised']:
                engagement_score += 0.1
                
            engagement_score = max(0.1, min(1.0, engagement_score))
            
            # 3. Calculate Stress Level
            # Factors: Nervous expressions, high speech rate, high hesitations
            stress_level = 0.2 # Base stress
            
            if expression in ['fearful', 'nervous']:
                stress_level += 0.4
            
            if wpm > 160:
                stress_level += 0.2
                
            if hesitations > 5:
                stress_level += 0.2
                
            stress_level = max(0.0, min(1.0, stress_level))
            
            # 4. Determine Dominant State
            state = BehavioralState.NEUTRAL
            observation = "Candidate appears composed."
            
            if stress_level > 0.7:
                state = BehavioralState.NERVOUS
                observation = "Candidate shows signs of high stress or anxiety."
            elif engagement_score < 0.5:
                state = BehavioralState.DISTRACTED
                observation = "Candidate seems distracted or disengaged."
            elif confidence_score < 0.5:
                state = BehavioralState.HESITANT
                observation = "Candidate appears unsure or hesitant."
            elif confidence_score > 0.8 and engagement_score > 0.8:
                state = BehavioralState.CONFIDENT
                observation = "Candidate demonstrates strong confidence and engagement."
            elif engagement_score > 0.7:
                state = BehavioralState.ENGAGED
                observation = "Candidate is actively engaged in the conversation."
                
            return BehavioralInsight(
                state=state,
                confidence_score=confidence_score,
                engagement_score=engagement_score,
                stress_level=stress_level,
                observation=observation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing signals: {e}")
            return BehavioralInsight(
                state=BehavioralState.NEUTRAL,
                confidence_score=0.5,
                engagement_score=0.5,
                stress_level=0.5,
                observation="Unable to analyze behavioral signals."
            )

    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze text for sentiment and confidence markers (fallback if no audio/video).
        """
        # Simple heuristic-based analysis for text-only scenarios
        hesitation_markers = ['um', 'uh', 'like', 'maybe', 'sort of', 'i think', 'probably']
        confidence_markers = ['definitely', 'certainly', 'i know', 'experience with', 'successfully']
        
        words = text.lower().split()
        word_count = len(words)
        if word_count == 0:
            return {"confidence": 0.5, "sentiment": 0.5}
            
        hesitation_count = sum(1 for w in words if w in hesitation_markers)
        confidence_count = sum(1 for w in words if w in confidence_markers)
        
        confidence_ratio = (confidence_count - hesitation_count) / word_count
        # Normalize roughly to 0-1
        confidence_score = 0.5 + (confidence_ratio * 2) 
        confidence_score = max(0.1, min(1.0, confidence_score))
        
        return {
            "confidence": confidence_score,
            "hesitation_count": hesitation_count
        }

# Singleton
_deep_sensing_service = None

def get_deep_sensing_service() -> DeepSensingService:
    global _deep_sensing_service
    if _deep_sensing_service is None:
        _deep_sensing_service = DeepSensingService()
    return _deep_sensing_service