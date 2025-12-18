"""
🎤 AGENTIC INTERVIEW API - Real-Time Interview Enhancement
Exposes the AgenticAIIntegrationLayer for RealtimeInterviewer integration.

This API allows the frontend RealtimeInterviewer to:
1. Initialize interview sessions with agentic AI modules
2. Get personalized openings from memory layer
3. Process responses with deep sensing
4. Get AI-enhanced questions with drill-down
5. Apply human behavior simulation to responses
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging

# Import the integration layer
from agentic_ai.integration_layer import (
    get_integration_layer,
    InterviewMode,
    ProcessingStage
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agentic-interview", tags=["agentic-interview"])


class InitSessionRequest(BaseModel):
    """Request to initialize an agentic interview session"""
    session_id: str
    candidate_id: str
    job_id: str
    company_id: str
    interview_mode: str = InterviewMode.STANDARD.value
    candidate_profile: Optional[Dict[str, Any]] = None
    job_requirements: Optional[Dict[str, Any]] = None
    company_info: Optional[Dict[str, Any]] = None
    round_type: Optional[str] = "behavioral"
    round_data: Optional[Dict[str, Any]] = None


class ProcessResponseRequest(BaseModel):
    """Request to process a candidate's response"""
    session_id: str
    answer: str
    audio_features: Optional[Dict[str, Any]] = None  # Voice stress, tone, etc.
    video_features: Optional[Dict[str, Any]] = None  # Facial expressions, engagement


class GenerateQuestionRequest(BaseModel):
    """Request to generate next question"""
    session_id: str
    topic: Optional[str] = None
    enable_drill_down: bool = True


class HumanizeTextRequest(BaseModel):
    """Request to humanize AI text"""
    text: str
    context: Optional[Dict[str, Any]] = None
    style: str = "warm"  # warm, professional, casual


class GetSystemInstructionsRequest(BaseModel):
    """Request for enhanced system instructions from backend"""
    session_id: str
    round_type: str
    round_data: Optional[Dict[str, Any]] = None
    candidate_profile: Optional[Dict[str, Any]] = None
    company_info: Optional[Dict[str, Any]] = None
    job_requirements: Optional[Dict[str, Any]] = None

class CreateSessionRequest(BaseModel):
    """Request to create interview session for RealtimeInterviewer"""
    interview_id: str
    candidate_id: str
    job_title: str
    job_description: Optional[str] = None
    job_requirements: Optional[List[str]] = []
    job_skills: Optional[List[str]] = []
    round_type: str
    round_data: Optional[Dict[str, Any]] = None
    company_name: str
    company_intelligence: Optional[Dict[str, Any]] = None
    candidate_profile: Optional[Dict[str, Any]] = None
    evaluation_criteria: Optional[List[str]] = []
    language: str = "en"

class TranscriptRequest(BaseModel):
    """Request to add transcript entry"""
    role: str
    text: str
    timestamp: str

class GuidanceRequest(BaseModel):
    """Request for agentic guidance on next question"""
    user_response: str
    current_stage: str
    questions_asked: int


@router.post("/init-session")
async def init_agentic_session(request: InitSessionRequest):
    """
    🚀 Initialize an agentic interview session
    
    This connects all the AI modules for the interview:
    - Memory Layer: Remembers previous interactions
    - Adaptation Engine: Adjusts difficulty in real-time
    - Behavior Simulator: Makes responses human-like
    - Drill-Down Engine: Deep probing questions
    - Enhanced Sensing: Engagement/stress tracking
    
    Returns session context and available modules.
    """
    try:
        logger.info(f"🎤 Initializing agentic session: {request.session_id}")
        
        layer = get_integration_layer()
        
        result = await layer.initialize_session(
            session_id=request.session_id,
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            company_id=request.company_id,
            interview_mode=request.interview_mode,
            candidate_profile=request.candidate_profile,
            job_requirements=request.job_requirements,
            company_info=request.company_info
        )
        
        if result.get("success"):
            logger.info(f"✅ Session initialized with modules: {result.get('modules_initialized')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Session init error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# REALTIME INTERVIEWER ENDPOINTS - Called by Frontend RealtimeInterviewer
# ============================================================================

@router.post("/session/create")
async def create_realtime_session(request: CreateSessionRequest):
    """
    🎤 Create interview session for RealtimeInterviewer
    
    Called by frontend when RealtimeInterviewer component mounts.
    Returns session_id and system_instructions for Azure OpenAI Realtime API.
    
    CRITICAL: Returns HireGenix-branded instructions (NOT ChatGPT)
    """
    try:
        import uuid
        session_id = f"realtime_{request.interview_id}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🎤 Creating RealtimeInterviewer session: {session_id}")
        
        layer = get_integration_layer()
        
        # Initialize agentic session with all modules
        await layer.initialize_session(
            session_id=session_id,
            candidate_id=request.candidate_id,
            job_id=request.interview_id,
            company_id=request.company_name,
            interview_mode=InterviewMode.STANDARD.value,
            candidate_profile=request.candidate_profile,
            job_requirements={"skills": request.job_skills, "requirements": request.job_requirements},
            company_info=request.company_intelligence
        )
        
        # Generate HireGenix AI Interviewer system instructions
        system_instructions = _build_hiregenix_instructions(request)
        
        logger.info(f"✅ Session created: {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "system_instructions": system_instructions,
            "modules_available": layer.get_available_modules()
        }
        
    except Exception as e:
        logger.error(f"❌ Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/transcript")
async def add_transcript_entry(session_id: str, request: TranscriptRequest):
    """
    📝 Add transcript entry for agentic analysis
    
    Called by RealtimeInterviewer after each speech turn.
    """
    try:
        layer = get_integration_layer()
        
        # Process through agentic modules if user response
        if request.role.lower() in ["user", "candidate", "you"]:
            try:
                await layer.process_response(
                    session_id=session_id,
                    answer=request.text,
                    audio_data=None,
                    video_features=None
                )
            except:
                pass
        
        return {"success": True, "session_id": session_id}
        
    except Exception as e:
        logger.error(f"❌ Transcript error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/guidance")
async def get_agentic_guidance(session_id: str, request: GuidanceRequest):
    """
    🧠 Get agentic guidance for next question
    
    Uses drill-down, adaptation, and behavior modules.
    """
    try:
        layer = get_integration_layer()
        guidance = None
        
        # Check drill-down engine
        if layer.drill_down_engine:
            try:
                result = await layer.drill_down_engine.analyze_response(
                    response=request.user_response,
                    topic=request.current_stage,
                    depth=request.questions_asked
                )
                if result.get("should_drill_down"):
                    guidance = result.get("follow_up_question")
            except:
                pass
        
        return {
            "success": True,
            "guidance": guidance,
            "questions_asked": request.questions_asked
        }
        
    except Exception as e:
        logger.error(f"❌ Guidance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _build_hiregenix_instructions(request: CreateSessionRequest) -> str:
    """Build HireGenix AI Interviewer system instructions"""
    
    candidate_name = request.candidate_profile.get("name", "Candidate") if request.candidate_profile else "Candidate"
    candidate_skills = ", ".join(request.candidate_profile.get("skills", [])) if request.candidate_profile else ""
    
    company_mission = request.company_intelligence.get("mission", "") if request.company_intelligence else ""
    company_values = ", ".join(request.company_intelligence.get("values", [])) if request.company_intelligence else ""
    
    round_name = request.round_data.get("name", request.round_type) if request.round_data else request.round_type
    
    return f"""You are the HireGenix AI Interviewer, conducting professional interviews for {request.company_name}.

🚨 CRITICAL IDENTITY RULES:
- You are "HireGenix AI Interviewer" - NEVER say you are ChatGPT, OpenAI, GPT, or any other AI
- If asked who you are, say: "I'm your HireGenix AI Interviewer, here to conduct your {round_name} for {request.company_name}"
- If asked "Are you ChatGPT?", say: "No, I'm the HireGenix AI Interviewer"
- You work for {request.company_name} through HireGenix's AI interview platform

📋 INTERVIEW CONTEXT:
- Candidate: {candidate_name}
- Position: {request.job_title}
- Company: {request.company_name}
- Round: {round_name}
- Type: {request.round_type}
{f"- Candidate Skills: {candidate_skills}" if candidate_skills else ""}

🏢 COMPANY CONTEXT:
{f"- Mission: {company_mission}" if company_mission else ""}
{f"- Values: {company_values}" if company_values else ""}

📝 JOB DETAILS:
{f"- Description: {request.job_description[:500]}..." if request.job_description and len(request.job_description) > 500 else f"- Description: {request.job_description}" if request.job_description else ""}
{f"- Requirements: {', '.join(request.job_requirements)}" if request.job_requirements else ""}
{f"- Skills: {', '.join(request.job_skills)}" if request.job_skills else ""}

🎯 INTERVIEW INSTRUCTIONS:
1. Start with a warm, professional greeting
2. Introduce yourself as "HireGenix AI Interviewer"
3. Ask ONE question at a time
4. Listen carefully before following up
5. Be professional but warm
6. Acknowledge good answers positively
7. NEVER break character
8. At the end, thank the candidate

🎭 VOICE & TONE:
- Professional yet warm
- Encouraging and supportive
- Clear and concise
- Natural conversational flow

Begin with a warm greeting and introduce yourself as the HireGenix AI Interviewer."""


@router.post("/generate-opening")
async def generate_personalized_opening(session_id: str):
    """
    🎬 Generate a personalized opening using memory and behavior modules
    
    If candidate has history, references their previous interviews.
    Applies human behavior simulation for warmth.
    """
    try:
        layer = get_integration_layer()
        result = await layer.generate_opening(session_id=session_id)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Opening generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-response")
async def process_candidate_response(request: ProcessResponseRequest):
    """
    🧠 Process candidate response through all sensing modules
    
    Analyzes:
    - Voice stress and tone (if audio features provided)
    - Facial expressions and engagement (if video features)
    - Answer quality and relevance
    - Behavioral patterns
    
    Returns sensing summary and adaptation recommendations.
    """
    try:
        layer = get_integration_layer()
        
        result = await layer.process_response(
            session_id=request.session_id,
            answer=request.answer,
            audio_data=None,  # Audio bytes would need separate handling
            video_features=request.video_features
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Response processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-question")
async def generate_next_question(request: GenerateQuestionRequest):
    """
    ❓ Generate the next interview question
    
    Uses:
    - Drill-down engine for deep probing
    - Adaptation signals for difficulty adjustment
    - Behavior simulator for human-like phrasing
    
    Returns enhanced, natural question with context.
    """
    try:
        layer = get_integration_layer()
        
        result = await layer.generate_next_question(
            session_id=request.session_id,
            topic=request.topic,
            enable_drill_down=request.enable_drill_down
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Question generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/humanize-text")
async def humanize_ai_text(request: HumanizeTextRequest):
    """
    🎭 Make AI text more human-like using Behavior Simulator
    
    Adds:
    - Natural speech patterns
    - Acknowledgments and empathy
    - Thinking pauses
    - Warmth and personality
    """
    try:
        layer = get_integration_layer()
        
        if layer.behavior_simulator:
            result = await layer.behavior_simulator.make_human_like(
                text=request.text,
                context=request.context or {}
            )
            return {
                "success": True,
                "original": request.text,
                "humanized": result.get("result", request.text)
            }
        else:
            return {
                "success": True,
                "original": request.text,
                "humanized": request.text,
                "note": "Behavior simulator not available"
            }
        
    except Exception as e:
        logger.error(f"❌ Humanize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-enhanced-instructions")
async def get_enhanced_system_instructions(request: GetSystemInstructionsRequest):
    """
    📋 Get enhanced system instructions for OpenAI Realtime API
    
    Combines:
    - Round-specific personas from backend
    - Memory context if available
    - Company intelligence
    - Adaptation signals
    
    Returns comprehensive system prompt for the AI.
    """
    try:
        layer = get_integration_layer()
        
        # Check if session exists, if not create context
        session_state = layer.sessions.get(request.session_id)
        
        # Build enhanced context
        enhanced_context = {
            "round_type": request.round_type,
            "round_data": request.round_data,
            "has_memory": False,
            "adaptation_mode": "balanced",
            "behavior_style": "warm_professional"
        }
        
        # Get memory context if available
        if layer.memory_layer and session_state:
            try:
                memory_context = await layer.memory_layer.retrieve_context(
                    candidate_id=session_state.candidate_id,
                    company_id=session_state.company_id,
                    job_id=session_state.job_id
                )
                if memory_context.get("has_history"):
                    enhanced_context["has_memory"] = True
                    enhanced_context["previous_topics"] = memory_context.get("topics_discussed", [])
                    enhanced_context["personality_insights"] = memory_context.get("personality_insights", {})
            except:
                pass
        
        # Get adaptation signals if available
        if session_state and session_state.adaptation_signals:
            enhanced_context["adaptation_mode"] = session_state.adaptation_signals.get("mode", "balanced")
        
        # Build persona based on round type - matching frontend but from backend
        round_personas = {
            "phone_screening": {
                "name": "Sarah",
                "style": "warm, friendly, encouraging",
                "focus": "basic fit, motivation, communication",
                "tips": ["Keep it conversational", "Assess cultural fit", "Make candidate comfortable"]
            },
            "behavioral": {
                "name": "Priya", 
                "style": "curious, thorough, supportive",
                "focus": "STAR methodology, specific examples, past experiences",
                "tips": ["Probe for details", "Ask 'What specifically did YOU do?'", "Look for quantifiable results"]
            },
            "technical": {
                "name": "Emily",
                "style": "collegial, analytical, patient",
                "focus": "technical depth, problem-solving, implementation details",
                "tips": ["Start with fundamentals", "Increase complexity gradually", "Let them think aloud"]
            },
            "culture_fit": {
                "name": "Ananya",
                "style": "warm, values-focused, insightful",
                "focus": "values alignment, teamwork, growth mindset",
                "tips": ["Explore ideal work environment", "Ask about conflict handling", "Assess adaptability"]
            },
            "final": {
                "name": "Rachel",
                "style": "comprehensive, strategic, thorough",
                "focus": "overall fit, readiness, commitment",
                "tips": ["Cover gaps from previous rounds", "Discuss growth trajectory", "Be thorough but respectful"]
            }
        }
        
        persona = round_personas.get(request.round_type.lower(), round_personas["behavioral"])
        enhanced_context["persona"] = persona
        
        # Get available modules status
        enhanced_context["modules_available"] = layer.get_available_modules()
        
        return {
            "success": True,
            "session_id": request.session_id,
            "enhanced_context": enhanced_context
        }
        
    except Exception as e:
        logger.error(f"❌ Get instructions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/end-session")
async def end_interview_session(session_id: str, final_notes: Optional[str] = None):
    """
    🏁 End the interview session and generate report
    
    Returns:
    - Session summary
    - Behavioral trajectory
    - Module performance metrics
    - Q&A assessment if applicable
    """
    try:
        layer = get_integration_layer()
        
        result = await layer.end_session(
            session_id=session_id,
            final_notes=final_notes
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Session end error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/modules-status")
async def get_modules_status():
    """
    📊 Get status of all agentic AI modules
    
    Returns availability of:
    - Memory Layer
    - Adaptation Engine
    - Behavior Simulator
    - Drill-Down Engine
    - Enhanced Sensing
    - Voice Processor
    - etc.
    """
    try:
        layer = get_integration_layer()
        
        return {
            "success": True,
            "modules": layer.get_available_modules(),
            "registry": {
                name: {
                    "available": info["module"] is not None,
                    "capabilities": info["capabilities"],
                    "required_for": [m.value for m in info["required_for"]]
                }
                for name, info in layer.module_registry.items()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Modules status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
