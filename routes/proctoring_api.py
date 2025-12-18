"""
🛡️ PROCTORING API ROUTES
========================

REST API endpoints for AI-powered proctoring during interviews.
Connects frontend to the AdvancedProctoringAgent (CrewAI + YOLO + DeepFace).

Endpoints:
- POST /api/proctoring/analyze-frame - Analyze video frame for violations
- POST /api/proctoring/feedback - Record violation feedback for learning
- GET /api/proctoring/session/{id}/status - Get session proctoring status
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import base64
import asyncio

# Import proctoring agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from proctoring_agent import get_proctoring_agent, analyze_proctoring_frame, record_violation_feedback

router = APIRouter(prefix="/api/proctoring", tags=["Proctoring"])

# ============================================================================
# Request/Response Models
# ============================================================================

class FrameAnalysisRequest(BaseModel):
    """Request to analyze a video frame for proctoring violations"""
    candidate_id: str = Field(..., description="Unique candidate identifier")
    session_id: str = Field(..., description="Interview session ID")
    frame_data: str = Field(..., description="Base64 encoded video frame (JPEG)")
    previous_frame: Optional[str] = Field(None, description="Previous frame for motion detection")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio chunk")
    screen_activity: Optional[Dict[str, Any]] = Field(None, description="Tab switch/screen activity data")

class ViolationDetail(BaseModel):
    """Single violation detected"""
    type: str
    severity: str
    confidence: float
    description: str
    timestamp: str

class AnalysisResult(BaseModel):
    """Analysis result for a frame"""
    faces_detected: int
    objects_detected: List[Dict[str, Any]]
    gaze_status: str
    emotion: str
    audio_anomalies: List[str]
    behavioral_pattern: str

class RiskAssessment(BaseModel):
    """Risk assessment for the session"""
    risk_score: float
    ai_analysis: str
    should_flag: bool
    should_terminate: bool
    recommendation: str

class FrameAnalysisResponse(BaseModel):
    """Response from frame analysis"""
    success: bool
    candidate_id: str
    session_id: str
    timestamp: str
    analysis: Optional[AnalysisResult] = None
    violations: List[ViolationDetail] = []
    risk_assessment: Optional[RiskAssessment] = None
    workflow_steps_completed: int = 0
    error: Optional[str] = None

class ViolationFeedbackRequest(BaseModel):
    """Request to record feedback on a violation"""
    pattern_id: str = Field(..., description="Violation pattern ID")
    confirmed_fraud: bool = Field(..., description="Whether this was actual fraud")
    reviewer_notes: Optional[str] = Field("", description="Human reviewer notes")

class SessionStatusResponse(BaseModel):
    """Proctoring status for a session"""
    session_id: str
    candidate_id: str
    total_frames_analyzed: int
    total_violations: int
    risk_score: float
    is_flagged: bool
    is_terminated: bool
    violations_by_type: Dict[str, int]
    last_analysis_time: Optional[str]

# ============================================================================
# Session State (In-Memory for now, use Redis in production)
# ============================================================================

session_states: Dict[str, Dict] = {}

def get_session_state(session_id: str) -> Dict:
    """Get or create session state"""
    if session_id not in session_states:
        session_states[session_id] = {
            "session_id": session_id,
            "candidate_id": None,
            "total_frames_analyzed": 0,
            "total_violations": 0,
            "risk_score": 0.0,
            "is_flagged": False,
            "is_terminated": False,
            "violations_by_type": {},
            "last_analysis_time": None,
            "violation_history": []
        }
    return session_states[session_id]

# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/analyze-frame", response_model=FrameAnalysisResponse)
async def analyze_frame(request: FrameAnalysisRequest):
    """
    🔍 Analyze a video frame for proctoring violations
    
    Uses the AdvancedProctoringAgent (v3.0) which includes:
    - YOLO v8 for object detection (phones, earphones, smartwatches)
    - MediaPipe for face mesh and gaze tracking
    - DeepFace for emotion and identity verification
    - CrewAI multi-agent consensus for reducing false positives
    - DSPy optimization for violation detection
    - RAG knowledge base for historical fraud patterns
    
    Args:
        request: FrameAnalysisRequest with base64 encoded frame
        
    Returns:
        FrameAnalysisResponse with violations and risk assessment
    """
    try:
        print(f"📥 [Proctoring API] Received frame analysis request for session: {request.session_id}, candidate: {request.candidate_id}")
        print(f"📥 [Proctoring API] Frame data length: {len(request.frame_data)} chars")
        
        # Decode base64 frame to bytes
        try:
            frame_bytes = base64.b64decode(request.frame_data)
            print(f"📥 [Proctoring API] Decoded frame: {len(frame_bytes)} bytes")
        except Exception as e:
            print(f"❌ [Proctoring API] Failed to decode frame: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 frame data: {str(e)}")
        
        # Decode previous frame if provided
        previous_frame_bytes = None
        if request.previous_frame:
            try:
                previous_frame_bytes = base64.b64decode(request.previous_frame)
            except:
                pass  # Ignore if previous frame is invalid
        
        # Decode audio if provided
        audio_bytes = None
        if request.audio_data:
            try:
                audio_bytes = base64.b64decode(request.audio_data)
            except:
                pass
        
        print(f"🔍 [Proctoring API] Running proctoring analysis...")
        
        # Run proctoring analysis
        result = await analyze_proctoring_frame(
            candidate_id=request.candidate_id,
            session_id=request.session_id,
            frame_data=frame_bytes,
            audio_data=audio_bytes,
            previous_frame=previous_frame_bytes,
            screen_activity=request.screen_activity
        )
        
        print(f"📤 [Proctoring API] Analysis complete: success={result.get('success')}, violations={len(result.get('violations', []))}")
        if result.get('violations'):
            print(f"🚨 [Proctoring API] Violations found: {[v.get('type') for v in result.get('violations', [])]}")
        
        # Update session state
        state = get_session_state(request.session_id)
        state["candidate_id"] = request.candidate_id
        state["total_frames_analyzed"] += 1
        state["last_analysis_time"] = datetime.now().isoformat()
        
        if result.get("success"):
            # Update violation counts
            violations = result.get("violations", [])
            state["total_violations"] += len(violations)
            
            for v in violations:
                v_type = v.get("type", "UNKNOWN")
                state["violations_by_type"][v_type] = state["violations_by_type"].get(v_type, 0) + 1
                state["violation_history"].append(v)
            
            # Update risk assessment
            risk = result.get("risk_assessment", {})
            state["risk_score"] = risk.get("risk_score", state["risk_score"])
            state["is_flagged"] = risk.get("should_flag", state["is_flagged"])
            state["is_terminated"] = risk.get("should_terminate", state["is_terminated"])
            
            # Build response
            analysis_data = result.get("analysis", {})
            
            return FrameAnalysisResponse(
                success=True,
                candidate_id=request.candidate_id,
                session_id=request.session_id,
                timestamp=result.get("timestamp", datetime.now().isoformat()),
                analysis=AnalysisResult(
                    faces_detected=analysis_data.get("faces_detected", 0),
                    objects_detected=analysis_data.get("objects_detected", []),
                    gaze_status=analysis_data.get("gaze_status", "unknown"),
                    emotion=analysis_data.get("emotion", "neutral"),
                    audio_anomalies=analysis_data.get("audio_anomalies", []),
                    behavioral_pattern=analysis_data.get("behavioral_pattern", "normal")
                ),
                violations=[
                    ViolationDetail(
                        type=v.get("type", "UNKNOWN"),
                        severity=v.get("severity", "LOW"),
                        confidence=v.get("confidence", 0.0),
                        description=v.get("description", ""),
                        timestamp=v.get("timestamp", datetime.now().isoformat())
                    )
                    for v in violations
                ],
                risk_assessment=RiskAssessment(
                    risk_score=risk.get("risk_score", 0.0),
                    ai_analysis=risk.get("ai_analysis", ""),
                    should_flag=risk.get("should_flag", False),
                    should_terminate=risk.get("should_terminate", False),
                    recommendation=risk.get("recommendation", "CONTINUE_MONITORING")
                ),
                workflow_steps_completed=result.get("workflow", {}).get("total_steps", 0)
            )
        else:
            return FrameAnalysisResponse(
                success=False,
                candidate_id=request.candidate_id,
                session_id=request.session_id,
                timestamp=datetime.now().isoformat(),
                error=result.get("error", "Analysis failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        return FrameAnalysisResponse(
            success=False,
            candidate_id=request.candidate_id,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@router.post("/feedback")
async def submit_violation_feedback(request: ViolationFeedbackRequest):
    """
    📝 Record feedback on a violation pattern for learning
    
    This helps the system learn from:
    - Confirmed fraud cases (true positives)
    - False positive cases (improve detection accuracy)
    
    The feedback is stored in the RAG knowledge base for future reference.
    """
    try:
        result = await record_violation_feedback(
            pattern_id=request.pattern_id,
            confirmed_fraud=request.confirmed_fraud,
            reviewer_notes=request.reviewer_notes
        )
        
        if result.get("success"):
            return {
                "success": True,
                "message": "Feedback recorded successfully",
                "pattern_id": request.pattern_id
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Failed to record feedback")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    📊 Get proctoring status for a session
    
    Returns:
    - Total frames analyzed
    - Total violations detected
    - Current risk score
    - Flag/termination status
    - Violations breakdown by type
    """
    state = get_session_state(session_id)
    
    return SessionStatusResponse(
        session_id=session_id,
        candidate_id=state.get("candidate_id", "unknown"),
        total_frames_analyzed=state.get("total_frames_analyzed", 0),
        total_violations=state.get("total_violations", 0),
        risk_score=state.get("risk_score", 0.0),
        is_flagged=state.get("is_flagged", False),
        is_terminated=state.get("is_terminated", False),
        violations_by_type=state.get("violations_by_type", {}),
        last_analysis_time=state.get("last_analysis_time")
    )

@router.delete("/session/{session_id}")
async def end_proctoring_session(session_id: str):
    """
    🛑 End a proctoring session and cleanup resources
    """
    if session_id in session_states:
        # Get final state for logging
        final_state = session_states[session_id]
        
        # Cleanup
        del session_states[session_id]
        
        return {
            "success": True,
            "session_id": session_id,
            "final_stats": {
                "total_frames_analyzed": final_state.get("total_frames_analyzed", 0),
                "total_violations": final_state.get("total_violations", 0),
                "risk_score": final_state.get("risk_score", 0.0),
                "was_flagged": final_state.get("is_flagged", False),
                "was_terminated": final_state.get("is_terminated", False)
            }
        }
    else:
        return {
            "success": True,
            "session_id": session_id,
            "message": "Session not found or already ended"
        }

@router.get("/health")
async def proctoring_health_check():
    """
    ❤️ Health check for proctoring service
    """
    try:
        agent = get_proctoring_agent()
        return {
            "status": "healthy",
            "agent_version": "3.0",
            "features": {
                "crewai_enabled": agent.crewai_enabled if hasattr(agent, 'crewai_enabled') else False,
                "rag_enabled": agent.rag_enabled if hasattr(agent, 'rag_enabled') else False,
                "dspy_enabled": agent.dspy_enabled if hasattr(agent, 'dspy_enabled') else False,
                "yolo_available": agent.yolo_model is not None if hasattr(agent, 'yolo_model') else False,
                "mediapipe_available": agent.face_mesh is not None if hasattr(agent, 'face_mesh') else False
            },
            "active_sessions": len(session_states)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
