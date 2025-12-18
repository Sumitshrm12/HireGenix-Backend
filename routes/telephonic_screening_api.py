"""
📞 TELEPHONIC SCREENING API ROUTES
FastAPI routes for telephonic screening operations.
Handles call initiation, webhooks, and status monitoring.
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from agentic_ai.telephonic_screening_agent import (
    get_telephonic_screening_orchestrator,
    initiate_telephonic_screening,
    CallStatus,
    CallOutcome,
    ScreeningStage
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telephonic-screening", tags=["Telephonic Screening"])

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class InitiateCallRequest(BaseModel):
    """Request to initiate telephonic screening call"""
    candidate_id: str
    job_id: str
    phone_number: str = Field(..., description="Phone number in E.164 format (e.g., +919876543210)")
    language: str = Field(default="en-IN", description="Language: en-IN, hi-IN, en-US")
    
    # Optional context
    candidate_profile: Optional[Dict[str, Any]] = None
    job_requirements: Optional[Dict[str, Any]] = None
    company_info: Optional[Dict[str, Any]] = None
    resume_analysis: Optional[Dict[str, Any]] = None
    
    # Scheduling
    scheduled_time: Optional[datetime] = None

class ScheduleCallRequest(BaseModel):
    """Request to schedule telephonic screening call"""
    candidate_id: str
    job_id: str
    phone_number: str
    scheduled_time: datetime
    language: str = "en-IN"
    timezone: str = "Asia/Kolkata"
    
    # Optional preferences
    preferred_time_slots: Optional[List[str]] = None  # ["morning", "afternoon", "evening"]
    max_attempts: int = 3
    reminder_before_minutes: int = 15

class CallStatusResponse(BaseModel):
    """Response with call status"""
    call_id: str
    status: str
    stage: str
    questions_asked: int
    duration_seconds: float
    outcome: Optional[str] = None

class CallResultResponse(BaseModel):
    """Complete call result response"""
    success: bool
    call_id: str
    outcome: Optional[str] = None
    overall_score: Optional[float] = None
    recommendation: Optional[str] = None
    next_steps: Optional[List[str]] = None
    data_collected: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class WebhookEvent(BaseModel):
    """Azure Communication Services webhook event"""
    event_type: str
    call_connection_id: Optional[str] = None
    server_call_id: Optional[str] = None
    correlation_id: Optional[str] = None
    event_data: Dict[str, Any] = Field(default_factory=dict)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/initiate", response_model=CallResultResponse)
async def initiate_call(
    request: InitiateCallRequest,
    background_tasks: BackgroundTasks
):
    """
    Initiate a telephonic screening call immediately.
    
    This endpoint triggers an outbound call to the candidate using Azure Communication Services.
    The call follows a structured screening flow with AI-powered conversation.
    """
    try:
        logger.info(f"📞 Initiating telephonic screening for candidate {request.candidate_id}")
        
        # Validate phone number format
        if not request.phone_number.startswith("+"):
            raise HTTPException(
                status_code=400,
                detail="Phone number must be in E.164 format (e.g., +919876543210)"
            )
        
        # Get candidate and job context if not provided
        candidate_profile = request.candidate_profile or {"id": request.candidate_id}
        job_requirements = request.job_requirements or {"id": request.job_id}
        company_info = request.company_info or {}
        
        # Start screening call
        result = await initiate_telephonic_screening(
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            phone_number=request.phone_number,
            candidate_profile=candidate_profile,
            job_requirements=job_requirements,
            company_info=company_info,
            resume_analysis=request.resume_analysis,
            language=request.language
        )
        
        if result.get("success"):
            call_result = result.get("result", {})
            return CallResultResponse(
                success=True,
                call_id=result.get("call_id", ""),
                outcome=call_result.get("outcome"),
                overall_score=call_result.get("overall_score"),
                recommendation=call_result.get("recommendation"),
                next_steps=call_result.get("next_steps"),
                data_collected=call_result.get("data_collected"),
                analysis=call_result.get("analysis")
            )
        else:
            return CallResultResponse(
                success=False,
                call_id=result.get("call_id", ""),
                error=result.get("error", "Unknown error")
            )
            
    except Exception as e:
        logger.error(f"❌ Call initiation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/schedule")
async def schedule_call(request: ScheduleCallRequest):
    """
    Schedule a telephonic screening call for later.
    
    The system will automatically initiate the call at the scheduled time
    and handle retries if the candidate doesn't answer.
    """
    try:
        # In production, this would store in database and use a scheduler
        # For now, return acknowledgment
        
        return {
            "success": True,
            "message": "Call scheduled successfully",
            "scheduled_call": {
                "candidate_id": request.candidate_id,
                "job_id": request.job_id,
                "phone_number": request.phone_number,
                "scheduled_time": request.scheduled_time.isoformat(),
                "timezone": request.timezone,
                "language": request.language,
                "max_attempts": request.max_attempts,
                "reminder_before_minutes": request.reminder_before_minutes
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Schedule error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{call_id}", response_model=CallStatusResponse)
async def get_call_status(call_id: str):
    """
    Get the current status of a telephonic screening call.
    """
    try:
        orchestrator = get_telephonic_screening_orchestrator()
        status = orchestrator.get_call_status(call_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Call not found")
        
        return CallStatusResponse(
            call_id=status["call_id"],
            status=status["status"],
            stage=status["stage"],
            questions_asked=status["questions_asked"],
            duration_seconds=status["duration"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/webhook/{call_id}")
async def handle_webhook(call_id: str, request: Request):
    """
    Handle Azure Communication Services webhook events.
    
    This endpoint receives real-time events from ACS including:
    - CallConnected: Call was answered
    - RecognizeCompleted: Speech was recognized
    - RecognizeFailed: Speech recognition failed
    - CallDisconnected: Call ended
    - PlayCompleted: TTS playback completed
    """
    try:
        body = await request.json()
        
        # Handle array of events (ACS sends array)
        events = body if isinstance(body, list) else [body]
        
        orchestrator = get_telephonic_screening_orchestrator()
        
        for event in events:
            event_type = event.get("type", "")
            event_data = event.get("data", {})
            
            logger.info(f"📨 Webhook event: {event_type} for call {call_id}")
            
            # Handle the event
            result = await orchestrator.handle_call_event(
                call_id=call_id,
                event_type=event_type,
                event_data=event_data
            )
            
            if not result.get("success"):
                logger.warning(f"⚠️ Event handling failed: {result.get('error')}")
        
        return {"status": "processed"}
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        # Return 200 to acknowledge receipt (ACS expects this)
        return {"status": "error", "message": str(e)}

@router.post("/cancel/{call_id}")
async def cancel_call(call_id: str):
    """
    Cancel an active or scheduled call.
    """
    try:
        orchestrator = get_telephonic_screening_orchestrator()
        
        # Check if call exists
        status = orchestrator.get_call_status(call_id)
        if not status:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # If call is active, hang up
        if status["status"] in [CallStatus.IN_PROGRESS.value, CallStatus.RINGING.value]:
            state = orchestrator.active_calls.get(call_id)
            if state and state.call_connection_id:
                await orchestrator.acs_provider.hang_up(state.call_connection_id)
        
        return {
            "success": True,
            "message": "Call cancelled",
            "call_id": call_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Cancel error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{candidate_id}")
async def get_call_history(candidate_id: str, job_id: Optional[str] = None):
    """
    Get telephonic screening call history for a candidate.
    """
    try:
        # In production, this would query from database
        # For now, return placeholder
        
        return {
            "candidate_id": candidate_id,
            "calls": [],
            "total_calls": 0,
            "message": "Call history will be retrieved from database"
        }
        
    except Exception as e:
        logger.error(f"❌ History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-slots/{candidate_id}")
async def get_available_slots(
    candidate_id: str,
    date: Optional[str] = None,
    timezone: str = "Asia/Kolkata"
):
    """
    Get available time slots for scheduling a call.
    
    Returns available slots based on:
    - Recruiter calendar availability
    - Candidate preferred times (if known)
    - Optimal calling hours
    """
    try:
        from datetime import datetime, timedelta
        
        # Generate sample available slots
        # In production, this would check calendars and optimize
        
        base_date = datetime.now()
        if date:
            base_date = datetime.fromisoformat(date)
        
        slots = []
        for day_offset in range(5):  # Next 5 days
            current_date = base_date + timedelta(days=day_offset)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Available slots (10 AM - 6 PM IST)
            for hour in [10, 11, 14, 15, 16, 17]:
                slot_time = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                slots.append({
                    "datetime": slot_time.isoformat(),
                    "display": slot_time.strftime("%d %b %Y, %I:%M %p"),
                    "available": True
                })
        
        return {
            "candidate_id": candidate_id,
            "timezone": timezone,
            "available_slots": slots[:10],  # Return first 10 slots
            "optimal_times": ["10:00 AM", "2:00 PM", "4:00 PM"]
        }
        
    except Exception as e:
        logger.error(f"❌ Slots error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics")
async def get_screening_analytics(
    company_id: Optional[str] = None,
    job_id: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
):
    """
    Get analytics for telephonic screening calls.
    """
    try:
        # In production, this would aggregate from database
        
        return {
            "total_calls": 0,
            "completed_calls": 0,
            "average_duration_seconds": 0,
            "outcome_distribution": {
                "qualified": 0,
                "not_qualified": 0,
                "requires_review": 0,
                "no_answer": 0,
                "callback_requested": 0
            },
            "average_score": 0,
            "calls_by_stage": {},
            "peak_hours": ["10:00 AM", "2:00 PM", "4:00 PM"],
            "language_distribution": {
                "en-IN": 0,
                "hi-IN": 0
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_screening_config():
    """
    Get telephonic screening configuration and capabilities.
    """
    return {
        "supported_languages": [
            {"code": "en-IN", "name": "English (India)", "voices": ["female", "male"]},
            {"code": "hi-IN", "name": "Hindi", "voices": ["female", "male"]},
            {"code": "en-US", "name": "English (US)", "voices": ["female", "male"]}
        ],
        "screening_stages": [stage.value for stage in ScreeningStage],
        "call_outcomes": [outcome.value for outcome in CallOutcome],
        "call_statuses": [status.value for status in CallStatus],
        "default_settings": {
            "max_questions": 12,
            "max_duration_minutes": 15,
            "retry_attempts": 3,
            "retry_interval_minutes": 30
        },
        "azure_services": {
            "communication_services": "Outbound calling",
            "speech_services": "Speech-to-Text, Text-to-Speech",
            "openai": "Conversational AI"
        }
    }

# ============================================================================
# BULK OPERATIONS
# ============================================================================

class BulkCallRequest(BaseModel):
    """Request to initiate bulk telephonic screening calls"""
    job_id: str
    candidate_ids: List[str]
    language: str = "en-IN"
    stagger_minutes: int = 5  # Time between calls
    max_concurrent: int = 3

@router.post("/bulk/initiate")
async def initiate_bulk_calls(
    request: BulkCallRequest,
    background_tasks: BackgroundTasks
):
    """
    Initiate telephonic screening for multiple candidates.
    
    Calls are staggered to avoid overwhelming the system and
    to give recruiters time to review results.
    """
    try:
        if len(request.candidate_ids) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 candidates per bulk operation"
            )
        
        # In production, this would queue calls
        job_id = f"bulk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "success": True,
            "bulk_job_id": job_id,
            "total_candidates": len(request.candidate_ids),
            "estimated_completion_minutes": len(request.candidate_ids) * request.stagger_minutes,
            "status": "queued",
            "message": f"Bulk screening job created. Calls will be initiated with {request.stagger_minutes} minute intervals."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Bulk initiate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bulk/status/{bulk_job_id}")
async def get_bulk_job_status(bulk_job_id: str):
    """
    Get status of a bulk screening job.
    """
    try:
        # In production, this would query from job queue/database
        
        return {
            "bulk_job_id": bulk_job_id,
            "status": "in_progress",
            "total_candidates": 0,
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "failed": 0,
            "results": []
        }
        
    except Exception as e:
        logger.error(f"❌ Bulk status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
