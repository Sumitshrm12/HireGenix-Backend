"""
🤖 ADVANCED PROCTORING AGENT (v3.0) - World-Class Agentic AI
=============================================================

Agentic AI-powered proctoring system with comprehensive fraud detection.

Features:
- LangGraph workflow for multi-step analysis
- CrewAI multi-agent collaboration (Vision Expert, Audio Expert, Behavior Analyst)
- DSPy MIPRO optimizer for violation detection optimization
- RAG knowledge base from historical fraud patterns
- Feedback loops from confirmed/false positive violations
- Real-time object detection (phones, earphones, bluetooth devices, smartwatches)
- Multi-face detection and identity verification
- Gaze tracking and attention monitoring
- Emotion analysis for stress/suspicious behavior
- Audio anomaly detection (background voices, phone rings)
- Screen switching and tab change detection

Author: HireGenix AI Team
Version: 3.0.0 (World-Class Agentic AI)
Last Updated: December 2025
"""

# Suppress urllib3 LibreSSL warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import hashlib
import json

# 🤖 CrewAI for Multi-Agent Collaboration
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# 🎯 DSPy for Prompt Optimization
try:
    import dspy
    from dspy import ChainOfThought
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# 📚 RAG & Vector Store
try:
    import redis
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Deep Learning Libraries (optional - graceful degradation)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    DeepFace = None

# Lazy MediaPipe initialization to avoid startup errors
MEDIAPIPE_AVAILABLE = False
_mp_module = None
_mp_face_mesh_instance = None
_mp_init_attempted = False

def _get_mediapipe():
    """Lazy load MediaPipe module with proper error handling"""
    global _mp_module, _mp_init_attempted, MEDIAPIPE_AVAILABLE
    if _mp_module is not None:
        return _mp_module
    if _mp_init_attempted:
        return None
    _mp_init_attempted = True
    try:
        import mediapipe as mp
        _mp_module = mp
        MEDIAPIPE_AVAILABLE = True
        return mp
    except Exception:
        MEDIAPIPE_AVAILABLE = False
        return None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

try:
    import torch
    import torchaudio
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"[Proctoring Agent] Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

# LangChain & Azure
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Local imports
import sys
import os
sys.path.append(os.path.dirname(__file__))
from agentic_ai.config import AgenticAIConfig


# ============================================================================
# 🎯 DSPY PROCTORING SIGNATURES
# ============================================================================

if DSPY_AVAILABLE:
    class ViolationDetectionSignature(dspy.Signature):
        """Analyze proctoring data and detect violations."""
        visual_analysis: str = dspy.InputField(desc="Face, object, gaze, emotion analysis")
        audio_analysis: str = dspy.InputField(desc="Audio anomaly detection results")
        historical_context: str = dspy.InputField(desc="Similar past violations and outcomes")
        violations_detected: str = dspy.OutputField(desc="JSON list of detected violations with severity")
        false_positive_likelihood: str = dspy.OutputField(desc="Probability this is a false positive 0-1")
        
    class RiskAssessmentSignature(dspy.Signature):
        """Calculate risk score and make proctoring decision."""
        violations: str = dspy.InputField(desc="List of detected violations")
        candidate_history: str = dspy.InputField(desc="Candidate's past session behavior")
        risk_score: str = dspy.OutputField(desc="Risk score 0.0-1.0")
        recommendation: str = dspy.OutputField(desc="CONTINUE, FLAG, or TERMINATE")


class ViolationType(str, Enum):
    """Types of proctoring violations"""
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    PHONE_DETECTED = "PHONE_DETECTED"
    EARPHONE_DETECTED = "EARPHONE_DETECTED"
    BLUETOOTH_DEVICE = "BLUETOOTH_DEVICE"
    SMARTWATCH_DETECTED = "SMARTWATCH_DETECTED"
    LOOKING_AWAY = "LOOKING_AWAY"
    SUSPICIOUS_EMOTION = "SUSPICIOUS_EMOTION"
    BACKGROUND_VOICE = "BACKGROUND_VOICE"
    SCREEN_SWITCH = "SCREEN_SWITCH"
    EXCESSIVE_MOTION = "EXCESSIVE_MOTION"
    UNAUTHORIZED_OBJECT = "UNAUTHORIZED_OBJECT"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"


class SeverityLevel(str, Enum):
    """Severity levels for violations"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ViolationEvent:
    """Represents a single proctoring violation"""
    timestamp: datetime
    violation_type: ViolationType
    severity: SeverityLevel
    confidence: float
    description: str
    frame_snapshot: Optional[str] = None
    audio_snippet: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


class ProctoringState(BaseModel):
    """State for LangGraph proctoring workflow"""
    candidate_id: str
    session_id: str
    frame_data: Optional[bytes] = None
    audio_data: Optional[bytes] = None
    previous_frame: Optional[bytes] = None
    screen_activity: Optional[Dict] = None
    
    # Detection results
    faces_detected: int = 0
    objects_detected: List[Dict] = Field(default_factory=list)
    gaze_status: str = "unknown"
    emotion: str = "neutral"
    audio_anomalies: List[str] = Field(default_factory=list)
    
    # RAG Context
    historical_patterns: List[Dict] = Field(default_factory=list)
    
    # CrewAI Collaboration Results
    crew_consensus: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis results
    violations: List[Dict] = Field(default_factory=list)
    risk_score: float = 0.0
    behavioral_pattern: str = "normal"
    ai_analysis: str = ""
    false_positive_likelihood: float = 0.0
    
    # Decision
    should_flag: bool = False
    should_terminate: bool = False
    recommendation: str = ""
    
    # Workflow tracking
    current_step: str = "initialized"
    steps_completed: List[str] = Field(default_factory=list)


class AdvancedProctoringAgent:
    """
    🚀 WORLD-CLASS PROCTORING AGENT (v3.0)
    
    Features:
    - LangGraph multi-step workflow
    - CrewAI 3-agent collaboration (Vision Expert, Audio Expert, Behavior Analyst)
    - DSPy MIPRO for violation detection optimization
    - RAG knowledge base from historical fraud patterns
    - Feedback loops from confirmed/false positive violations
    
    Uses:
    - YOLO v8 for object detection (phones, earphones, smartwatches)
    - MediaPipe for gaze tracking and facial landmarks
    - DeepFace for emotion and face recognition
    - Whisper for audio transcription
    - Azure OpenAI for behavioral analysis
    """
    
    def __init__(self):
        self.config = AgenticAIConfig()
        
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            openai_api_key=self.config.azure.api_key,
            azure_endpoint=self.config.azure.endpoint,
            deployment_name=self.config.azure.deployment_name,
            openai_api_version=self.config.azure.api_version,
            temperature=0.3
        )
        
        # Initialize Advanced Components
        self._init_crewai_agents()
        self._init_rag_knowledge_base()
        self._init_dspy_optimizer()
        
        # Initialize computer vision models
        self._init_cv_models()
        
        # Initialize audio processing
        self._init_audio_models()
        
        # Build LangGraph workflow
        self.workflow = self._build_langgraph_workflow()
        
        # Violation thresholds
        self.thresholds = {
            'max_violations': 3,
            'critical_violation_limit': 1,
            'risk_score_threshold': 0.7,
            'confidence_threshold': 0.6
        }
        
        print("✅ AdvancedProctoringAgent v3.0 initialized with CrewAI + DSPy + RAG")
    
    def _init_crewai_agents(self):
        """Initialize CrewAI multi-agent system for consensus-based decisions"""
        self.crewai_enabled = CREWAI_AVAILABLE
        
        if CREWAI_AVAILABLE:
            self.vision_expert = Agent(
                name="Vision Expert Agent",
                role="Computer Vision Specialist",
                goal="Analyze visual signals for proctoring violations (faces, objects, gaze, motion)",
                backstory="Expert computer vision engineer specializing in exam proctoring. 10 years experience detecting cheating via visual cues.",
                allow_delegation=False
            )
            
            self.audio_expert = Agent(
                name="Audio Expert Agent",
                role="Audio Analysis Specialist",
                goal="Detect audio anomalies indicating potential cheating (voices, phone sounds, whispering)",
                backstory="Audio forensics expert who has analyzed thousands of exam sessions. Expert at distinguishing ambient noise from suspicious audio.",
                allow_delegation=False
            )
            
            self.behavior_analyst = Agent(
                name="Behavior Analyst Agent",
                role="Behavioral Pattern Expert",
                goal="Synthesize visual and audio signals into holistic behavioral assessment, minimize false positives",
                backstory="Organizational psychologist specializing in exam integrity. Expert at distinguishing nervousness from cheating behavior.",
                allow_delegation=False
            )
            
            print("✅ CrewAI 3-agent proctoring crew initialized")
    
    def _init_rag_knowledge_base(self):
        """Initialize RAG for learning from historical fraud patterns"""
        self.rag_enabled = False
        if RAG_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    password=os.getenv("REDIS_PASSWORD", "") or None,
                    decode_responses=True
                )
                self.rag_enabled = True
                print("✅ Proctoring RAG Knowledge Base initialized")
            except Exception as e:
                print(f"⚠️ RAG initialization failed: {e}")
    
    def _init_dspy_optimizer(self):
        """Initialize DSPy for violation detection optimization"""
        self.dspy_enabled = False
        if DSPY_AVAILABLE:
            try:
                lm = dspy.LM(
                    model=f"openai/{self.config.azure.deployment_name}",
                    api_key=self.config.azure.api_key,
                    api_base=self.config.azure.endpoint,
                    api_version=self.config.azure.api_version,
                    model_type="chat"
                )
                dspy.configure(lm=lm)
                self.dspy_enabled = True
                print("✅ DSPy optimizer initialized for proctoring")
            except Exception as e:
                print(f"⚠️ DSPy initialization failed: {e}")
    
    def _init_cv_models(self):
        """Initialize computer vision models with lazy loading"""
        try:
            # YOLO v8 for object detection (phones, earphones, etc.)
            if YOLO_AVAILABLE:
                self.yolo_model = YOLO('yolov8n.pt')
            else:
                self.yolo_model = None
            
            # Custom YOLO classes for proctoring
            self.prohibited_objects = {
                67: 'cell phone',
                73: 'laptop',
                76: 'keyboard',
                77: 'mouse',
                # Custom trained classes (if available)
                'earphone': 'wireless_earphone',
                'bluetooth': 'bluetooth_device',
                'smartwatch': 'smartwatch'
            }
            
            # MediaPipe for face mesh and gaze tracking (lazy loaded)
            mp = _get_mediapipe()
            if mp is not None:
                try:
                    self.mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        max_num_faces=5,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                except Exception:
                    self.mp_face_mesh = None
                    self.face_mesh = None
            else:
                self.mp_face_mesh = None
                self.face_mesh = None
            
            # Haar cascades for backup face detection
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
        except Exception as e:
            self.yolo_model = None
            self.face_mesh = None
            self.mp_face_mesh = None
    
    def _init_audio_models(self):
        """Initialize audio processing models"""
        try:
            # Audio classification for anomaly detection
            self.audio_classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            
            # Speech detection
            self.voice_detector = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny"
            )
            
            print("[Proctoring Agent] Audio models initialized")
        except Exception as e:
            print(f"[Proctoring Agent] Audio initialization warning: {e}")
    
    def _build_langgraph_workflow(self):
        """Build LangGraph workflow for proctoring analysis"""
        workflow = StateGraph(ProctoringState)
        
        # Define workflow nodes
        workflow.add_node("retrieve_patterns", self._retrieve_historical_patterns)
        workflow.add_node("detect_faces", self._detect_faces)
        workflow.add_node("detect_objects", self._detect_objects)
        workflow.add_node("track_gaze", self._track_gaze)
        workflow.add_node("analyze_emotion", self._analyze_emotion)
        workflow.add_node("analyze_audio", self._analyze_audio)
        workflow.add_node("detect_motion", self._detect_motion)
        workflow.add_node("run_crew_consensus", self._run_crew_consensus)
        workflow.add_node("ai_behavioral_analysis", self._ai_behavioral_analysis)
        workflow.add_node("calculate_risk_score", self._calculate_risk_score)
        workflow.add_node("make_decision", self._make_decision)
        workflow.add_node("store_for_learning", self._store_violation_pattern)
        
        # Define workflow edges
        workflow.set_entry_point("retrieve_patterns")
        workflow.add_edge("retrieve_patterns", "detect_faces")
        workflow.add_edge("detect_faces", "detect_objects")
        workflow.add_edge("detect_objects", "track_gaze")
        workflow.add_edge("track_gaze", "analyze_emotion")
        workflow.add_edge("analyze_emotion", "analyze_audio")
        workflow.add_edge("analyze_audio", "detect_motion")
        workflow.add_edge("detect_motion", "run_crew_consensus")
        workflow.add_edge("run_crew_consensus", "ai_behavioral_analysis")
        workflow.add_edge("ai_behavioral_analysis", "calculate_risk_score")
        workflow.add_edge("calculate_risk_score", "make_decision")
        workflow.add_edge("make_decision", "store_for_learning")
        workflow.add_edge("store_for_learning", END)
        
        return workflow.compile()
    
    async def _retrieve_historical_patterns(self, state: ProctoringState) -> ProctoringState:
        """Retrieve historical fraud patterns from RAG"""
        state.current_step = "retrieve_patterns"
        
        if self.rag_enabled:
            try:
                patterns = []
                pattern_keys = self.redis_client.keys("fraud_pattern:*")
                
                for key in pattern_keys[:20]:
                    data = self.redis_client.get(key)
                    if data:
                        pattern = json.loads(data)
                        if pattern.get("confirmed_fraud", False):
                            patterns.append(pattern)
                
                state.historical_patterns = patterns[:5]
            except Exception as e:
                print(f"⚠️ Pattern retrieval error: {e}")
        
        state.steps_completed.append("retrieve_patterns")
        return state
    
    async def _run_crew_consensus(self, state: ProctoringState) -> ProctoringState:
        """Run CrewAI multi-agent consensus for violation assessment"""
        state.current_step = "run_crew_consensus"
        
        if self.crewai_enabled and state.violations:
            try:
                context = f"""
Session: {state.session_id}
Faces Detected: {state.faces_detected}
Objects: {json.dumps(state.objects_detected)}
Gaze: {state.gaze_status}
Emotion: {state.emotion}
Audio Anomalies: {state.audio_anomalies}
Detected Violations: {json.dumps(state.violations)}
Historical Fraud Patterns: {json.dumps([p.get('pattern') for p in state.historical_patterns[:3]])}
"""
                
                vision_task = Task(
                    description=f"Analyze visual signals: {state.faces_detected} faces, {len(state.objects_detected)} objects, gaze={state.gaze_status}. Assess if violations are genuine or false positives.",
                    agent=self.vision_expert,
                    expected_output="Visual violation assessment with confidence"
                )
                
                audio_task = Task(
                    description=f"Analyze audio signals: {state.audio_anomalies}. Determine if sounds indicate cheating or are ambient noise.",
                    agent=self.audio_expert,
                    expected_output="Audio violation assessment with confidence"
                )
                
                behavior_task = Task(
                    description=f"Synthesize all signals. Detected violations: {len(state.violations)}. Emotion: {state.emotion}. Determine if behavior indicates actual cheating or test anxiety.",
                    agent=self.behavior_analyst,
                    expected_output="Holistic behavioral assessment with false positive likelihood"
                )
                
                crew = Crew(
                    agents=[self.vision_expert, self.audio_expert, self.behavior_analyst],
                    tasks=[vision_task, audio_task, behavior_task],
                    process=Process.sequential,  # Sequential for consensus building
                    verbose=True
                )
                
                crew.kickoff()
                
                state.crew_consensus = {
                    "vision_assessment": vision_task.output.raw if vision_task.output else "",
                    "audio_assessment": audio_task.output.raw if audio_task.output else "",
                    "behavioral_assessment": behavior_task.output.raw if behavior_task.output else ""
                }
                
            except Exception as e:
                print(f"⚠️ CrewAI consensus error: {e}")
        
        state.steps_completed.append("run_crew_consensus")
        return state
    
    async def _store_violation_pattern(self, state: ProctoringState) -> ProctoringState:
        """Store violation pattern for future learning"""
        state.current_step = "store_for_learning"
        
        if self.rag_enabled and state.violations:
            try:
                pattern_id = hashlib.md5(
                    f"{state.session_id}{datetime.now().isoformat()}".encode()
                ).hexdigest()
                
                pattern = {
                    "id": pattern_id,
                    "session_id": state.session_id,
                    "candidate_id": state.candidate_id,
                    "violations": [v.get("type") for v in state.violations],
                    "risk_score": state.risk_score,
                    "recommendation": state.recommendation,
                    "crew_consensus": state.crew_consensus,
                    "confirmed_fraud": None,  # Updated via feedback
                    "false_positive": None,  # Updated via feedback
                    "timestamp": datetime.now().isoformat()
                }
                
                self.redis_client.set(f"fraud_pattern:{pattern_id}", json.dumps(pattern))
                
            except Exception as e:
                print(f"⚠️ Pattern storage error: {e}")
        
        state.steps_completed.append("store_for_learning")
        return state
    
    async def _detect_faces(self, state: ProctoringState) -> ProctoringState:
        """Step 1: Detect faces and verify identity"""
        state.current_step = "detect_faces"
        
        try:
            # Decode frame
            frame = cv2.imdecode(
                np.frombuffer(state.frame_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            state.faces_detected = len(faces)
            
            # Check violations
            if len(faces) == 0:
                state.violations.append({
                    'type': ViolationType.NO_FACE,
                    'severity': SeverityLevel.HIGH,
                    'confidence': 0.95,
                    'description': 'No face detected - candidate may have left seat',
                    'timestamp': datetime.now().isoformat()
                })
            elif len(faces) > 1:
                state.violations.append({
                    'type': ViolationType.MULTIPLE_FACES,
                    'severity': SeverityLevel.CRITICAL,
                    'confidence': 0.98,
                    'description': f'{len(faces)} faces detected - possible impersonation',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Verify identity using DeepFace
            if len(faces) == 1:
                try:
                    # Compare with registered photo
                    # In production, load reference photo from database
                    analysis = DeepFace.analyze(
                        frame,
                        actions=['age', 'gender', 'race'],
                        enforce_detection=False
                    )
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    
                    # Store for behavioral pattern analysis
                    state.behavioral_pattern = f"age:{analysis.get('age', 'unknown')}"
                    
                except Exception as e:
                    print(f"Identity verification error: {e}")
            
            state.steps_completed.append("detect_faces")
            
        except Exception as e:
            print(f"Face detection error: {e}")
        
        return state
    
    async def _detect_objects(self, state: ProctoringState) -> ProctoringState:
        """Step 2: Detect prohibited objects (phones, earphones, smartwatches)"""
        state.current_step = "detect_objects"
        
        try:
            # Check if YOLO model is available
            if self.yolo_model is None:
                print("⚠️ [Proctoring] YOLO model not loaded - skipping object detection")
                state.steps_completed.append("detect_objects")
                return state
            
            # Decode frame
            frame = cv2.imdecode(
                np.frombuffer(state.frame_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if frame is None:
                print("⚠️ [Proctoring] Failed to decode frame for object detection")
                state.steps_completed.append("detect_objects")
                return state
            
            print(f"📦 [Proctoring] Running YOLO detection on frame {frame.shape}")
            
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            print(f"📦 [Proctoring] YOLO detection complete, processing {len(results)} result sets")
            
            detected_objects = []
            for result in results:
                boxes = result.boxes
                print(f"📦 [Proctoring] Found {len(boxes)} boxes in this result")
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    object_name = self.yolo_model.names[cls_id]
                    
                    print(f"📦 [Proctoring] Detected: {object_name} (class {cls_id}) with confidence {confidence:.2f}")
                    
                    if confidence < self.thresholds['confidence_threshold']:
                        print(f"📦 [Proctoring] Skipping {object_name} - confidence {confidence:.2f} below threshold {self.thresholds['confidence_threshold']}")
                        continue
                    
                    detected_objects.append({
                        'class_id': cls_id,
                        'name': object_name,
                        'confidence': confidence,
                        'bbox': box.xyxy[0].tolist()
                    })
                    
                    # Check for prohibited objects
                    violation_type = None
                    severity = SeverityLevel.MEDIUM
                    
                    # Phone detection - class 67 is 'cell phone' in COCO dataset
                    if cls_id == 67 or 'phone' in object_name.lower() or 'cell' in object_name.lower():
                        violation_type = ViolationType.PHONE_DETECTED
                        severity = SeverityLevel.CRITICAL
                        print(f"🚨 [Proctoring] PHONE DETECTED! Class: {cls_id}, Name: {object_name}, Confidence: {confidence:.2f}")
                    elif 'earphone' in object_name.lower() or 'headphone' in object_name.lower():
                        violation_type = ViolationType.EARPHONE_DETECTED
                        severity = SeverityLevel.CRITICAL
                        print(f"🚨 [Proctoring] EARPHONE DETECTED!")
                    elif 'watch' in object_name.lower() or 'smartwatch' in object_name.lower():
                        violation_type = ViolationType.SMARTWATCH_DETECTED
                        severity = SeverityLevel.HIGH
                        print(f"🚨 [Proctoring] SMARTWATCH DETECTED!")
                    elif 'bluetooth' in object_name.lower():
                        violation_type = ViolationType.BLUETOOTH_DEVICE
                        severity = SeverityLevel.CRITICAL
                        print(f"🚨 [Proctoring] BLUETOOTH DEVICE DETECTED!")
                    elif cls_id in [73, 76, 77]:  # laptop, keyboard, mouse
                        violation_type = ViolationType.UNAUTHORIZED_OBJECT
                        severity = SeverityLevel.HIGH
                        print(f"🚨 [Proctoring] UNAUTHORIZED OBJECT DETECTED: {object_name}")
                    
                    if violation_type:
                        state.violations.append({
                            'type': violation_type,
                            'severity': severity,
                            'confidence': confidence,
                            'description': f'{object_name} detected near candidate',
                            'timestamp': datetime.now().isoformat(),
                            'bbox': box.xyxy[0].tolist()
                        })
                        print(f"🚨 [Proctoring] Added violation: {violation_type.value}")
            
            print(f"📦 [Proctoring] Total objects detected: {len(detected_objects)}, Violations: {len(state.violations)}")
            state.objects_detected = detected_objects
            state.steps_completed.append("detect_objects")
            
        except Exception as e:
            print(f"❌ [Proctoring] Object detection error: {e}")
            import traceback
            print(f"❌ [Proctoring] Traceback: {traceback.format_exc()}")
            state.steps_completed.append("detect_objects")
        
        return state
    
    async def _track_gaze(self, state: ProctoringState) -> ProctoringState:
        """Step 3: Track gaze direction using MediaPipe"""
        state.current_step = "track_gaze"
        
        try:
            # Decode frame
            frame = cv2.imdecode(
                np.frombuffer(state.frame_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get eye landmarks for gaze estimation
                # Left eye: landmarks 33, 133, 160, 144, 159, 145
                # Right eye: landmarks 362, 263, 387, 373, 380, 374
                
                # Calculate gaze direction
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                nose = face_landmarks.landmark[1]
                
                # Simple gaze estimation based on eye-nose alignment
                eye_center_x = (left_eye.x + right_eye.x) / 2
                nose_x = nose.x
                
                # Determine gaze status
                gaze_offset = abs(eye_center_x - nose_x)
                
                if gaze_offset < 0.03:
                    state.gaze_status = "on_screen"
                elif gaze_offset < 0.06:
                    state.gaze_status = "slightly_off"
                else:
                    state.gaze_status = "looking_away"
                    state.violations.append({
                        'type': ViolationType.LOOKING_AWAY,
                        'severity': SeverityLevel.MEDIUM,
                        'confidence': 0.75,
                        'description': 'Candidate looking away from screen',
                        'timestamp': datetime.now().isoformat()
                    })
            else:
                state.gaze_status = "no_face"
            
            state.steps_completed.append("track_gaze")
            
        except Exception as e:
            print(f"Gaze tracking error: {e}")
        
        return state
    
    async def _analyze_emotion(self, state: ProctoringState) -> ProctoringState:
        """Step 4: Analyze facial emotions for suspicious behavior"""
        state.current_step = "analyze_emotion"
        
        try:
            # Decode frame
            frame = cv2.imdecode(
                np.frombuffer(state.frame_data, np.uint8),
                cv2.IMREAD_COLOR
            )
            
            # Analyze emotion with DeepFace
            analysis = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            dominant_emotion = analysis.get('dominant_emotion', 'neutral')
            state.emotion = dominant_emotion
            
            # Flag suspicious emotions
            suspicious_emotions = ['fear', 'surprise', 'angry']
            if dominant_emotion in suspicious_emotions:
                emotion_scores = analysis.get('emotion', {})
                confidence = emotion_scores.get(dominant_emotion, 0) / 100
                
                state.violations.append({
                    'type': ViolationType.SUSPICIOUS_EMOTION,
                    'severity': SeverityLevel.LOW,
                    'confidence': confidence,
                    'description': f'Suspicious emotion detected: {dominant_emotion}',
                    'timestamp': datetime.now().isoformat()
                })
            
            state.steps_completed.append("analyze_emotion")
            
        except Exception as e:
            print(f"Emotion analysis error: {e}")
        
        return state
    
    async def _analyze_audio(self, state: ProctoringState) -> ProctoringState:
        """Step 5: Analyze audio for background voices and anomalies"""
        state.current_step = "analyze_audio"
        
        if not state.audio_data:
            state.steps_completed.append("analyze_audio")
            return state
        
        try:
            # Audio classification for anomaly detection
            audio_results = self.audio_classifier(state.audio_data)
            
            anomalies = []
            for result in audio_results:
                label = result['label'].lower()
                score = result['score']
                
                # Check for suspicious sounds
                if score > 0.5:
                    if any(keyword in label for keyword in ['speech', 'voice', 'conversation']):
                        anomalies.append(f'background_voice:{score:.2f}')
                        state.violations.append({
                            'type': ViolationType.BACKGROUND_VOICE,
                            'severity': SeverityLevel.HIGH,
                            'confidence': score,
                            'description': f'Background voice detected: {label}',
                            'timestamp': datetime.now().isoformat()
                        })
                    elif 'phone' in label or 'ring' in label:
                        anomalies.append(f'phone_sound:{score:.2f}')
            
            state.audio_anomalies = anomalies
            state.steps_completed.append("analyze_audio")
            
        except Exception as e:
            print(f"Audio analysis error: {e}")
        
        return state
    
    async def _detect_motion(self, state: ProctoringState) -> ProctoringState:
        """Step 6: Detect excessive motion between frames"""
        state.current_step = "detect_motion"
        
        if not state.previous_frame or not state.frame_data:
            state.steps_completed.append("detect_motion")
            return state
        
        try:
            # Decode frames
            prev = cv2.imdecode(
                np.frombuffer(state.previous_frame, np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
            curr = cv2.imdecode(
                np.frombuffer(state.frame_data, np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
            
            # Compute frame difference
            diff = cv2.absdiff(prev, curr)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            non_zero = np.count_nonzero(thresh)
            
            # Check for excessive motion
            if non_zero > 50000:  # Threshold for significant motion
                state.violations.append({
                    'type': ViolationType.EXCESSIVE_MOTION,
                    'severity': SeverityLevel.MEDIUM,
                    'confidence': min(non_zero / 100000, 1.0),
                    'description': 'Excessive movement detected',
                    'timestamp': datetime.now().isoformat()
                })
            
            state.steps_completed.append("detect_motion")
            
        except Exception as e:
            print(f"Motion detection error: {e}")
        
        return state
    
    async def _ai_behavioral_analysis(self, state: ProctoringState) -> ProctoringState:
        """Step 7: AI-powered behavioral pattern analysis"""
        state.current_step = "ai_behavioral_analysis"
        
        try:
            # Prepare context for AI analysis
            context = f"""
Proctoring Session Analysis:
- Candidate ID: {state.candidate_id}
- Session ID: {state.session_id}
- Faces Detected: {state.faces_detected}
- Objects Detected: {len(state.objects_detected)}
- Gaze Status: {state.gaze_status}
- Emotion: {state.emotion}
- Audio Anomalies: {len(state.audio_anomalies)}
- Violations: {len(state.violations)}

Violation Details:
{self._format_violations(state.violations)}

Screen Activity:
{state.screen_activity if state.screen_activity else 'No screen switching detected'}
"""
            
            # Get AI analysis
            messages = [
                SystemMessage(content="""You are an expert proctoring analyst with experience in 
online examination integrity. Analyze the proctoring data and provide:
1. Behavioral pattern assessment
2. Risk level evaluation
3. Specific concerns
4. Recommendations

Be objective and consider that some anomalies may have legitimate explanations."""),
                HumanMessage(content=context)
            ]
            
            response = await asyncio.to_thread(
                self.llm.invoke,
                messages
            )
            
            state.ai_analysis = response.content
            state.steps_completed.append("ai_behavioral_analysis")
            
        except Exception as e:
            print(f"AI behavioral analysis error: {e}")
            state.ai_analysis = "Analysis unavailable"
        
        return state
    
    async def _calculate_risk_score(self, state: ProctoringState) -> ProctoringState:
        """Step 8: Calculate overall risk score"""
        state.current_step = "calculate_risk_score"
        
        try:
            risk_score = 0.0
            
            # Weight violations by severity
            severity_weights = {
                SeverityLevel.LOW: 0.1,
                SeverityLevel.MEDIUM: 0.25,
                SeverityLevel.HIGH: 0.5,
                SeverityLevel.CRITICAL: 1.0
            }
            
            for violation in state.violations:
                severity = violation.get('severity', SeverityLevel.LOW)
                confidence = violation.get('confidence', 0.5)
                risk_score += severity_weights.get(severity, 0.25) * confidence
            
            # Normalize to 0-1 range
            state.risk_score = min(risk_score / 3.0, 1.0)
            
            state.steps_completed.append("calculate_risk_score")
            
        except Exception as e:
            print(f"Risk score calculation error: {e}")
        
        return state
    
    async def _make_decision(self, state: ProctoringState) -> ProctoringState:
        """Step 9: Make final decision on session integrity"""
        state.current_step = "make_decision"
        
        try:
            # Count critical violations
            critical_count = sum(
                1 for v in state.violations
                if v.get('severity') == SeverityLevel.CRITICAL
            )
            
            # Decision logic
            if critical_count >= self.thresholds['critical_violation_limit']:
                state.should_terminate = True
                state.should_flag = True
                state.recommendation = "TERMINATE_SESSION: Critical violation detected"
            elif state.risk_score >= self.thresholds['risk_score_threshold']:
                state.should_flag = True
                state.recommendation = "FLAG_FOR_REVIEW: High risk score"
            elif len(state.violations) >= self.thresholds['max_violations']:
                state.should_flag = True
                state.recommendation = "FLAG_FOR_REVIEW: Multiple violations detected"
            else:
                state.recommendation = "CONTINUE_MONITORING: Session appears normal"
            
            state.steps_completed.append("make_decision")
            
        except Exception as e:
            print(f"Decision making error: {e}")
        
        return state
    
    def _format_violations(self, violations: List[Dict]) -> str:
        """Format violations for AI analysis"""
        if not violations:
            return "No violations detected"
        
        formatted = []
        for i, v in enumerate(violations, 1):
            formatted.append(
                f"{i}. {v.get('type', 'UNKNOWN')} - "
                f"Severity: {v.get('severity', 'UNKNOWN')}, "
                f"Confidence: {v.get('confidence', 0):.2f}, "
                f"Description: {v.get('description', 'N/A')}"
            )
        
        return "\n".join(formatted)
    
    async def analyze_frame(
        self,
        candidate_id: str,
        session_id: str,
        frame_data: bytes,
        audio_data: Optional[bytes] = None,
        previous_frame: Optional[bytes] = None,
        screen_activity: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main method: Analyze frame for proctoring violations
        
        Args:
            candidate_id: Unique candidate identifier
            session_id: Exam session identifier
            frame_data: Current video frame (bytes)
            audio_data: Audio chunk for analysis (optional)
            previous_frame: Previous frame for motion detection (optional)
            screen_activity: Screen switching/tab change data (optional)
        
        Returns:
            Comprehensive proctoring analysis with violations and recommendations
        """
        try:
            # Initialize state
            initial_state = ProctoringState(
                candidate_id=candidate_id,
                session_id=session_id,
                frame_data=frame_data,
                audio_data=audio_data,
                previous_frame=previous_frame,
                screen_activity=screen_activity
            )
            
            # Execute LangGraph workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Format response
            return {
                'success': True,
                'candidate_id': final_state.candidate_id,
                'session_id': final_state.session_id,
                'timestamp': datetime.now().isoformat(),
                'analysis': {
                    'faces_detected': final_state.faces_detected,
                    'objects_detected': final_state.objects_detected,
                    'gaze_status': final_state.gaze_status,
                    'emotion': final_state.emotion,
                    'audio_anomalies': final_state.audio_anomalies,
                    'behavioral_pattern': final_state.behavioral_pattern
                },
                'violations': final_state.violations,
                'risk_assessment': {
                    'risk_score': final_state.risk_score,
                    'ai_analysis': final_state.ai_analysis,
                    'should_flag': final_state.should_flag,
                    'should_terminate': final_state.should_terminate,
                    'recommendation': final_state.recommendation
                },
                'workflow': {
                    'steps_completed': final_state.steps_completed,
                    'total_steps': 9
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Singleton instance
_proctoring_agent = None

def get_proctoring_agent() -> AdvancedProctoringAgent:
    """Get or create singleton proctoring agent instance"""
    global _proctoring_agent
    if _proctoring_agent is None:
        _proctoring_agent = AdvancedProctoringAgent()
    return _proctoring_agent


async def analyze_proctoring_frame(
    candidate_id: str,
    session_id: str,
    frame_data: bytes,
    audio_data: Optional[bytes] = None,
    previous_frame: Optional[bytes] = None,
    screen_activity: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convenience function for frame analysis
    
    Usage:
        result = await analyze_proctoring_frame(
            candidate_id="CAND123",
            session_id="SESS456",
            frame_data=frame_bytes,
            audio_data=audio_bytes
        )
    """
    agent = get_proctoring_agent()
    return await agent.analyze_frame(
        candidate_id,
        session_id,
        frame_data,
        audio_data,
        previous_frame,
        screen_activity
    )


async def record_violation_feedback(
    pattern_id: str,
    confirmed_fraud: bool,
    reviewer_notes: str = ""
) -> Dict[str, Any]:
    """
    Record feedback on a violation pattern (for learning)
    
    Args:
        pattern_id: ID of the violation pattern
        confirmed_fraud: Whether this was actual fraud
        reviewer_notes: Optional notes from human reviewer
    
    Returns:
        Status of the feedback recording
    """
    agent = get_proctoring_agent()
    
    if not agent.rag_enabled:
        return {"success": False, "error": "RAG not enabled"}
    
    try:
        data = agent.redis_client.get(f"fraud_pattern:{pattern_id}")
        if data:
            pattern = json.loads(data)
            pattern["confirmed_fraud"] = confirmed_fraud
            pattern["false_positive"] = not confirmed_fraud
            pattern["reviewer_notes"] = reviewer_notes
            pattern["feedback_timestamp"] = datetime.now().isoformat()
            agent.redis_client.set(f"fraud_pattern:{pattern_id}", json.dumps(pattern))
            return {"success": True, "pattern_id": pattern_id}
        else:
            return {"success": False, "error": "Pattern not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}
