# prescreening/proctoring_system.py - Client-Side Detection System
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import base64

# For computer vision (would need opencv-python)
import cv2
import numpy as np

import statistics
import logging
from enum import Enum
from typing import Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database imports
import psycopg2
import psycopg2.extras
from decouple import config

# AI imports
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Define enums and data classes that were missing
class ViolationType(Enum):
    FACE_NOT_DETECTED = "FACE_NOT_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    LOOKING_AWAY = "LOOKING_AWAY"
    MOBILE_DEVICE = "MOBILE_DEVICE"
    SUSPICIOUS_OBJECT = "SUSPICIOUS_OBJECT"
    TAB_SWITCHING = "TAB_SWITCHING"
    WINDOW_BLUR = "WINDOW_BLUR"
    FULLSCREEN_EXIT = "FULLSCREEN_EXIT"
    AUDIO_DETECTION = "AUDIO_DETECTION"
    LIGHTING_CHANGE = "LIGHTING_CHANGE"
    SCREEN_SHARING = "SCREEN_SHARING"
    COPY_PASTE = "COPY_PASTE"
    UNAUTHORIZED_SOFTWARE = "UNAUTHORIZED_SOFTWARE"

class SeverityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class ProctoringEvent:
    event_id: str
    session_id: str
    violation_type: ViolationType
    timestamp: datetime
    confidence_score: float
    severity: SeverityLevel
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0

@dataclass
class FaceMetrics:
    face_count: int = 0
    confidence: float = 0.0
    face_area: float = 0.0
    position: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h

@dataclass
class GazeMetrics:
    looking_at_screen: bool = False
    attention_score: float = 0.0
    gaze_stability: float = 0.0
    eye_aspect_ratio: float = 0.0

@dataclass
class BehavioralMetrics:
    head_movement_frequency: float = 0.0
    typing_pattern_anomaly: bool = False
    activity_level: float = 0.0
    posture_consistency: float = 0.0

@dataclass
class EnvironmentalMetrics:
    lighting_consistency: float = 0.0
    background_changes: int = 0
    audio_level: float = 0.0
    technical_interruptions: int = 0

@dataclass
class ProctoringSession:
    session_id: str
    candidate_id: str
    start_time: datetime = field(default_factory=datetime.now)
    current_time: datetime = field(default_factory=datetime.now)
    events: List[ProctoringEvent] = field(default_factory=list)
    violation_counts: Dict[ViolationType, int] = field(default_factory=lambda: defaultdict(int))
    overall_integrity_score: float = 1.0
    risk_assessment: Optional[Dict[str, Any]] = None
    
    # Metrics history for analysis
    face_metrics_history: Deque[FaceMetrics] = field(default_factory=lambda: deque(maxlen=100))
    gaze_metrics_history: Deque[GazeMetrics] = field(default_factory=lambda: deque(maxlen=100))
    behavioral_metrics_history: Deque[BehavioralMetrics] = field(default_factory=lambda: deque(maxlen=100))
    environmental_metrics_history: Deque[EnvironmentalMetrics] = field(default_factory=lambda: deque(maxlen=100))

@dataclass
class DetectionResult:
    """Result of a proctoring detection"""
    detection_type: str
    confidence: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    evidence_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ProctoringConfiguration:
    """Configuration for proctoring system"""
    enable_face_detection: bool = True
    enable_multiple_person_detection: bool = True
    enable_audio_analysis: bool = True
    enable_tab_monitoring: bool = True
    enable_screen_recording: bool = True
    face_detection_interval: int = 2  # seconds
    audio_analysis_interval: int = 5  # seconds
    violation_threshold_score: int = 50
    auto_terminate_on_critical: bool = False

class FaceDetectionAnalyzer:
    """Analyze video frames for face detection violations"""
    
    def __init__(self):
        # Initialize face detector (OpenCV Haar Cascades)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        except:
            print("Warning: OpenCV face cascades not found. Using mock detection.")
            self.face_cascade = None
            self.eye_cascade = None
    
    def analyze_frame(self, frame_base64: str) -> List[DetectionResult]:
        """Analyze a single video frame for violations"""
        detections = []
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(frame_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return [DetectionResult(
                    detection_type="TECHNICAL_ERROR",
                    confidence=1.0,
                    severity="MEDIUM",
                    description="Unable to decode video frame"
                )]
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            face_detections = self._detect_faces(gray, frame)
            detections.extend(face_detections)
            
            # Multiple person detection
            person_detections = self._detect_multiple_persons(gray, frame)
            detections.extend(person_detections)
            
            # Attention analysis
            attention_detections = self._analyze_attention(gray, frame)
            detections.extend(attention_detections)
            
        except Exception as e:
            detections.append(DetectionResult(
                detection_type="TECHNICAL_ERROR",
                confidence=1.0,
                severity="MEDIUM",
                description=f"Frame analysis error: {str(e)}"
            ))
        
        return detections
    
    def _detect_faces(self, gray: np.ndarray, frame: np.ndarray) -> List[DetectionResult]:
        """Detect faces in the frame"""
        detections = []
        
        if self.face_cascade is None:
            # Mock detection for demo
            return [DetectionResult(
                detection_type="FACE_DETECTED",
                confidence=0.9,
                severity="LOW",
                description="Face detection active (mock mode)"
            )]
        
        try:
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                detections.append(DetectionResult(
                    detection_type="NO_FACE_DETECTED",
                    confidence=0.8,
                    severity="HIGH",
                    description="No face detected in frame",
                    evidence_data={"face_count": 0}
                ))
            elif len(faces) == 1:
                detections.append(DetectionResult(
                    detection_type="FACE_DETECTED",
                    confidence=0.9,
                    severity="LOW",
                    description="Single face detected - normal",
                    evidence_data={"face_count": 1}
                ))
            else:
                detections.append(DetectionResult(
                    detection_type="MULTIPLE_FACES_DETECTED",
                    confidence=0.95,
                    severity="CRITICAL",
                    description=f"Multiple faces detected: {len(faces)}",
                    evidence_data={"face_count": len(faces)}
                ))
            
        except Exception as e:
            detections.append(DetectionResult(
                detection_type="FACE_DETECTION_ERROR",
                confidence=1.0,
                severity="MEDIUM",
                description=f"Face detection failed: {str(e)}"
            ))
        
        return detections
    
    def _detect_multiple_persons(self, gray: np.ndarray, frame: np.ndarray) -> List[DetectionResult]:
        """Detect multiple persons in frame"""
        detections = []
        
        try:
            # Use HOG person detector (more advanced detection)
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect people
            people, weights = hog.detectMultiScale(gray, winStride=(8,8))
            
            if len(people) > 1:
                detections.append(DetectionResult(
                    detection_type="MULTIPLE_PERSONS_DETECTED",
                    confidence=0.85,
                    severity="CRITICAL",
                    description=f"Multiple persons detected: {len(people)}",
                    evidence_data={"person_count": len(people)}
                ))
            
        except Exception as e:
            # Fallback - just log the attempt
            detections.append(DetectionResult(
                detection_type="PERSON_DETECTION_INFO",
                confidence=0.5,
                severity="LOW",
                description="Person detection attempted"
            ))
        
        return detections
    
    def _analyze_attention(self, gray: np.ndarray, frame: np.ndarray) -> List[DetectionResult]:
        """Analyze if candidate is looking at camera"""
        detections = []
        
        if self.face_cascade is None or self.eye_cascade is None:
            return []
        
        try:
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) < 2:
                    detections.append(DetectionResult(
                        detection_type="ATTENTION_ISSUE",
                        confidence=0.7,
                        severity="MEDIUM",
                        description="Candidate may not be looking at camera",
                        evidence_data={"eyes_detected": len(eyes)}
                    ))
        
        except Exception as e:
            pass  # Don't report attention analysis errors
        
        return detections

class AudioAnalyzer:
    """Analyze audio for suspicious activity"""
    
    def analyze_audio_chunk(self, audio_base64: str) -> List[DetectionResult]:
        """Analyze audio chunk for violations"""
        detections = []
        
        try:
            # In a real implementation, you would:
            # 1. Decode audio data
            # 2. Analyze for multiple voices
            # 3. Detect background conversations
            # 4. Check for suspicious sounds
            
            # Mock implementation
            import random
            
            # Simulate audio analysis
            if random.random() < 0.1:  # 10% chance of detecting something
                detections.append(DetectionResult(
                    detection_type="MULTIPLE_VOICES_DETECTED",
                    confidence=0.8,
                    severity="HIGH",
                    description="Multiple voices detected in audio",
                    evidence_data={"voice_count": 2}
                ))
            
            if random.random() < 0.05:  # 5% chance
                detections.append(DetectionResult(
                    detection_type="BACKGROUND_CONVERSATION",
                    confidence=0.75,
                    severity="MEDIUM",
                    description="Background conversation detected"
                ))
            
        except Exception as e:
            detections.append(DetectionResult(
                detection_type="AUDIO_ANALYSIS_ERROR",
                confidence=1.0,
                severity="LOW",
                description=f"Audio analysis error: {str(e)}"
            ))
        
        return detections

class TabMonitor:
    """Monitor browser tab switching and window focus"""
    
    def __init__(self):
        self.focus_lost_count = 0
        self.tab_switch_count = 0
        self.last_focus_time = time.time()
    
    def record_focus_lost(self) -> DetectionResult:
        """Record when browser loses focus"""
        self.focus_lost_count += 1
        
        severity = "LOW"
        if self.focus_lost_count > 3:
            severity = "HIGH"
        elif self.focus_lost_count > 1:
            severity = "MEDIUM"
        
        return DetectionResult(
            detection_type="FOCUS_LOST",
            confidence=1.0,
            severity=severity,
            description=f"Browser focus lost (count: {self.focus_lost_count})",
            evidence_data={"total_focus_lost_count": self.focus_lost_count}
        )
    
    def record_tab_switch(self, url_before: str, url_after: str) -> DetectionResult:
        """Record tab switching"""
        self.tab_switch_count += 1
        
        severity = "MEDIUM"
        if self.tab_switch_count > 5:
            severity = "HIGH"
        
        return DetectionResult(
            detection_type="TAB_SWITCH_DETECTED",
            confidence=1.0,
            severity=severity,
            description=f"Tab switch detected (count: {self.tab_switch_count})",
            evidence_data={
                "url_before": url_before,
                "url_after": url_after,
                "total_switches": self.tab_switch_count
            }
        )
    
    def record_visibility_change(self, is_visible: bool) -> Optional[DetectionResult]:
        """Record page visibility changes"""
        if not is_visible:
            return DetectionResult(
                detection_type="PAGE_HIDDEN",
                confidence=1.0,
                severity="MEDIUM",
                description="Page became hidden/minimized"
            )
        return None

class ScreenRecordingMonitor:
    """Monitor for screen recording and screenshot attempts"""
    
    def detect_recording_software(self) -> List[DetectionResult]:
        """Detect running screen recording software (client-side detection)"""
        detections = []
        
        # This would be implemented on the client-side JavaScript
        # Here we provide the structure for server-side processing
        
        common_recording_apps = [
            "OBS Studio", "Camtasia", "ScreenFlow", "QuickTime Player",
            "Loom", "Zoom", "Teams", "Chrome Screen Recorder"
        ]
        
        # Mock detection - in reality, this comes from client
        import random
        if random.random() < 0.02:  # 2% chance
            app_name = random.choice(common_recording_apps)
            detections.append(DetectionResult(
                detection_type="RECORDING_SOFTWARE_DETECTED",
                confidence=0.9,
                severity="CRITICAL",
                description=f"Screen recording software detected: {app_name}",
                evidence_data={"detected_app": app_name}
            ))
        
        return detections

class ViolationScorer:
    """Calculate violation scores and determine actions"""
    
    def __init__(self):
        self.violation_weights = {
            "CRITICAL": 25,
            "HIGH": 15,
            "MEDIUM": 8,
            "LOW": 3
        }
    
    def calculate_session_score(self, events: List[ProctoringEvent]) -> Dict[str, Any]:
        """Calculate overall violation score for session"""
        
        total_score = 0
        violation_breakdown = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for event in events:
            severity = event.severity
            if severity in self.violation_weights:
                score = self.violation_weights[severity]
                total_score += score
                violation_breakdown[severity] += 1
        
        # Determine session status
        if total_score >= 75:
            status = "TERMINATED"
            recommendation = "Session should be terminated due to excessive violations"
        elif total_score >= 50:
            status = "HIGH_RISK"
            recommendation = "Human review required - high violation risk"
        elif total_score >= 25:
            status = "MODERATE_RISK"
            recommendation = "Monitor closely - moderate violations detected"
        else:
            status = "LOW_RISK"
            recommendation = "Normal proctoring - minimal violations"
        
        return {
            "total_score": total_score,
            "status": status,
            "recommendation": recommendation,
            "violation_breakdown": violation_breakdown,
            "total_events": len(events)
        }

class RealTimeProctoringSystem:
    """Main proctoring system coordinator"""
    
    def __init__(self, config: Optional[ProctoringConfiguration] = None):
        self.config = config or ProctoringConfiguration()
        self.face_analyzer = FaceDetectionAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.tab_monitor = TabMonitor()
        self.screen_monitor = ScreenRecordingMonitor()
        self.scorer = ViolationScorer()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def start_proctoring_session(self, session_id: str) -> Dict[str, Any]:
        """Start proctoring for a session"""
        
        self.active_sessions[session_id] = {
            "start_time": datetime.now(),
            "violation_count": 0,
            "last_analysis": datetime.now(),
            "tab_monitor": TabMonitor(),
            "status": "ACTIVE"
        }
        
        # Send initial notification
        notification_service = await get_notification_service()
        await notification_service.send_notification(
            recipient_id="system",
            title="Proctoring Started",
            message=f"Proctoring session {session_id} has been initiated",
            notification_type="SYSTEM_ALERT",
            channel="webhook"
        )
        
        return {
            "session_id": session_id,
            "status": "started",
            "config": {
                "face_detection": self.config.enable_face_detection,
                "audio_analysis": self.config.enable_audio_analysis,
                "tab_monitoring": self.config.enable_tab_monitoring,
                "screen_recording": self.config.enable_screen_recording
            }
        }
    
    async def process_video_frame(
        self,
        session_id: str,
        frame_base64: str,
        candidate_id: str
    ) -> Dict[str, Any]:
        """Process a video frame for violations"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        # Analyze frame
        detections = self.face_analyzer.analyze_frame(frame_base64)
        
        # Process each detection
        events_created = []
        for detection in detections:
            if detection.severity in ["HIGH", "CRITICAL"]:
                # Create proctoring event
                event = ProctoringEvent(
                    id=f"proc_{int(time.time() * 1000)}",
                    session_id=session_id,
                    candidate_id=candidate_id,
                    event_type=detection.detection_type,
                    event_data=detection.evidence_data or {},
                    severity=detection.severity,
                    detected_at=detection.timestamp
                )
                
                # Store in database
                await self._store_proctoring_event(event)
                events_created.append(event)
                
                # Update session violation count
                self.active_sessions[session_id]["violation_count"] += 1
        
        # Check if session should be terminated
        session_data = self.active_sessions[session_id]
        should_terminate = await self._check_termination_criteria(session_id, events_created)
        
        return {
            "detections": [
                {
                    "type": d.detection_type,
                    "severity": d.severity,
                    "description": d.description,
                    "confidence": d.confidence
                } for d in detections
            ],
            "events_created": len(events_created),
            "should_terminate": should_terminate,
            "session_status": session_data["status"]
        }
    
    async def process_audio_chunk(
        self,
        session_id: str,
        audio_base64: str,
        candidate_id: str
    ) -> Dict[str, Any]:
        """Process audio chunk for violations"""
        
        if not self.config.enable_audio_analysis:
            return {"audio_analysis": "disabled"}
        
        detections = self.audio_analyzer.analyze_audio_chunk(audio_base64)
        
        events_created = []
        for detection in detections:
            if detection.severity in ["MEDIUM", "HIGH", "CRITICAL"]:
                event = ProctoringEvent(
                    id=f"audio_{int(time.time() * 1000)}",
                    session_id=session_id,
                    candidate_id=candidate_id,
                    event_type=detection.detection_type,
                    event_data=detection.evidence_data or {},
                    severity=detection.severity,
                    detected_at=detection.timestamp
                )
                
                await self._store_proctoring_event(event)
                events_created.append(event)
        
        return {
            "audio_detections": [
                {
                    "type": d.detection_type,
                    "severity": d.severity,
                    "description": d.description
                } for d in detections
            ],
            "events_created": len(events_created)
        }
    
    async def record_tab_event(
        self,
        session_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        candidate_id: str
    ) -> Dict[str, Any]:
        """Record tab monitoring event"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        tab_monitor = self.active_sessions[session_id]["tab_monitor"]
        detection = None
        
        if event_type == "focus_lost":
            detection = tab_monitor.record_focus_lost()
        elif event_type == "tab_switch":
            detection = tab_monitor.record_tab_switch(
                event_data.get("url_before", ""),
                event_data.get("url_after", "")
            )
        elif event_type == "visibility_change":
            detection = tab_monitor.record_visibility_change(
                event_data.get("is_visible", True)
            )
        
        if detection and detection.severity in ["MEDIUM", "HIGH"]:
            event = ProctoringEvent(
                id=f"tab_{int(time.time() * 1000)}",
                session_id=session_id,
                candidate_id=candidate_id,
                event_type=detection.detection_type,
                event_data=detection.evidence_data or {},
                severity=detection.severity,
                detected_at=detection.timestamp
            )
            
            await self._store_proctoring_event(event)
            
            return {
                "event_recorded": True,
                "severity": detection.severity,
                "description": detection.description
            }
        
        return {"event_recorded": False, "reason": "Below threshold"}
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get proctoring summary for session"""
        
        try:
            db = await get_database()
            events = await db.get_proctoring_events_by_session(session_id)
            
            # Calculate scores
            score_data = self.scorer.calculate_session_score(events)
            
            # Session statistics
            session_data = self.active_sessions.get(session_id, {})
            
            return {
                "session_id": session_id,
                "total_events": len(events),
                "violation_score": score_data["total_score"],
                "status": score_data["status"],
                "recommendation": score_data["recommendation"],
                "violation_breakdown": score_data["violation_breakdown"],
                "session_duration": str(datetime.now() - session_data.get("start_time", datetime.now())),
                "events": [
                    {
                        "type": event.event_type,
                        "severity": event.severity,
                        "timestamp": event.detected_at.isoformat(),
                        "description": event.event_data
                    } for event in events[-10:]  # Last 10 events
                ]
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get session summary: {str(e)}",
                "session_id": session_id
            }
    
    async def end_proctoring_session(self, session_id: str) -> Dict[str, Any]:
        """End proctoring session"""
        
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "ENDED"
            
            # Get final summary
            summary = await self.get_session_summary(session_id)
            
            # Send completion notification
            notification_service = await get_notification_service()
            await notification_service.send_notification(
                recipient_id="system",
                title="Proctoring Ended",
                message=f"Proctoring session {session_id} completed. Violation score: {summary.get('violation_score', 0)}",
                notification_type="SYSTEM_ALERT",
                channel="webhook"
            )
            
            # Clean up session data
            del self.active_sessions[session_id]
            
            return {
                "session_ended": True,
                "final_summary": summary
            }
        
        return {"error": "Session not found"}
    
    async def _store_proctoring_event(self, event: ProctoringEvent):
        """Store proctoring event in database"""
        try:
            db = await get_database()
            await db.record_proctoring_event(event)
        except Exception as e:
            print(f"Error storing proctoring event: {e}")
    
    async def _check_termination_criteria(
        self,
        session_id: str,
        new_events: List[ProctoringEvent]
    ) -> bool:
        """Check if session should be terminated"""
        
        # Check for immediate critical violations
        critical_events = [e for e in new_events if e.severity == "CRITICAL"]
        if critical_events and self.config.auto_terminate_on_critical:
            return True
        
        # Check overall violation score
        try:
            db = await get_database()
            all_events = await db.get_proctoring_events_by_session(session_id)
            score_data = self.scorer.calculate_session_score(all_events)
            
            if score_data["total_score"] >= self.config.violation_threshold_score:
                return True
            
        except Exception as e:
            print(f"Error checking termination criteria: {e}")
        
        return False

# Factory function
def create_proctoring_system(config: Optional[ProctoringConfiguration] = None) -> RealTimeProctoringSystem:
    """Create proctoring system instance"""
    return RealTimeProctoringSystem(config)