"""
📞 AGENTIC TELEPHONIC SCREENING AGENT
AI-powered outbound phone call screening using Azure GPT-4o-realtime-mini.
Ultra-low latency speech-to-speech conversations for phone interviews.

Features:
- Azure OpenAI GPT-4o-realtime-mini for speech-to-speech (WebSocket)
- Azure Communication Services for telephony
- Sub-200ms latency voice responses
- Natural interruption handling
- Voice Activity Detection (VAD)
- Multi-language support (Hindi + English)
- Adaptive questioning based on responses
- Real-time sentiment analysis
- Call recording and analysis
- Integration with existing prescreening pipeline

Tech Stack:
- Azure OpenAI Realtime API (gpt-4o-realtime-preview)
- Azure Communication Services for telephony
- WebSocket for bidirectional audio streaming
- LangGraph for workflow orchestration
"""

import os
import json
import asyncio
import uuid
import websockets
import base64
import struct
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
import io
import wave

from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Azure SDKs
try:
    from azure.communication.callautomation import (
        CallAutomationClient,
        PhoneNumberIdentifier,
        MediaStreamingOptions,
        MediaStreamingTransportType,
        MediaStreamingContentType,
        MediaStreamingAudioChannelType
    )
    AZURE_COMM_AVAILABLE = True
except ImportError:
    AZURE_COMM_AVAILABLE = False
    logging.warning("⚠️ Azure Communication Services SDK not installed")

# Import existing modules
try:
    from .voice_native_processor import get_voice_processor, VoiceNativeProcessor
    from .human_behavior_simulator import get_human_behavior_simulator
    from .memory_layer import get_memory_layer, PersistentMemoryLayer
    from .real_time_adaptation_engine import get_adaptation_engine
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class CallStatus(str, Enum):
    """Status of telephonic screening call"""
    SCHEDULED = "scheduled"
    INITIATED = "initiated"
    RINGING = "ringing"
    CONNECTED = "connected"
    STREAMING = "streaming"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    VOICEMAIL = "voicemail"
    CANCELLED = "cancelled"

class ScreeningStage(str, Enum):
    """Stages of telephonic screening"""
    GREETING = "greeting"
    INTRODUCTION = "introduction"
    AVAILABILITY_CHECK = "availability_check"
    EXPERIENCE_VERIFICATION = "experience_verification"
    SKILL_ASSESSMENT = "skill_assessment"
    MOTIVATION_CHECK = "motivation_check"
    SALARY_EXPECTATION = "salary_expectation"
    QUESTIONS_FROM_CANDIDATE = "questions_from_candidate"
    CLOSING = "closing"
    POST_CALL_ANALYSIS = "post_call_analysis"

class CallOutcome(str, Enum):
    """Outcome of telephonic screening"""
    QUALIFIED = "qualified"
    NOT_QUALIFIED = "not_qualified"
    REQUIRES_REVIEW = "requires_review"
    CALLBACK_REQUESTED = "callback_requested"
    NOT_INTERESTED = "not_interested"
    INCOMPLETE = "incomplete"

class RealtimeEventType(str, Enum):
    """Azure OpenAI Realtime API event types"""
    SESSION_CREATE = "session.create"
    SESSION_CREATED = "session.created"
    SESSION_UPDATE = "session.update"
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    RESPONSE_CREATE = "response.create"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_DONE = "response.done"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
    ERROR = "error"

# ============================================================================
# STATE MODELS
# ============================================================================

class TelephonicScreeningState(BaseModel):
    """State for telephonic screening workflow"""
    # Session identifiers
    call_id: str = ""
    session_id: str = ""
    candidate_id: str = ""
    job_id: str = ""
    company_id: str = ""
    
    # Call details
    phone_number: str = ""
    call_status: str = CallStatus.SCHEDULED.value
    call_connection_id: str = ""
    server_call_id: str = ""
    recording_url: str = ""
    
    # Context
    candidate_profile: Dict[str, Any] = Field(default_factory=dict)
    job_requirements: Dict[str, Any] = Field(default_factory=dict)
    company_info: Dict[str, Any] = Field(default_factory=dict)
    resume_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # Language settings
    language: str = "en-IN"  # en-IN, hi-IN, en-US
    voice: str = "alloy"  # Azure Realtime voices: alloy, echo, shimmer
    
    # Realtime session
    realtime_ws: Optional[Any] = None
    realtime_session_id: str = ""
    
    # Conversation state
    current_stage: str = ScreeningStage.GREETING.value
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_question: str = ""
    current_response: str = ""
    questions_asked: int = 0
    max_questions: int = 12
    
    # Transcripts
    ai_transcript: str = ""
    user_transcript: str = ""
    full_transcript: List[Dict[str, str]] = Field(default_factory=list)
    
    # Screening data collected
    availability_confirmed: bool = False
    notice_period: Optional[str] = None
    current_ctc: Optional[str] = None
    expected_ctc: Optional[str] = None
    relocation_willing: Optional[bool] = None
    verified_experience: Dict[str, Any] = Field(default_factory=dict)
    skill_responses: List[Dict[str, Any]] = Field(default_factory=list)
    motivation_insights: List[str] = Field(default_factory=list)
    candidate_questions: List[str] = Field(default_factory=list)
    
    # Voice analysis
    voice_metrics: Dict[str, Any] = Field(default_factory=dict)
    sentiment_trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0
    engagement_score: float = 0.0
    
    # Scoring
    overall_score: float = 0.0
    stage_scores: Dict[str, float] = Field(default_factory=dict)
    red_flags: List[str] = Field(default_factory=list)
    green_flags: List[str] = Field(default_factory=list)
    
    # Outcome
    outcome: str = ""
    recommendation: str = ""
    next_steps: List[str] = Field(default_factory=list)
    
    # Timing
    call_start_time: Optional[datetime] = None
    call_end_time: Optional[datetime] = None
    call_duration_seconds: int = 0
    
    # Audio buffers
    input_audio_buffer: bytes = b""
    output_audio_buffer: bytes = b""
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    retry_count: int = 0
    
    class Config:
        arbitrary_types_allowed = True

# ============================================================================
# AZURE OPENAI REALTIME CLIENT
# ============================================================================

class AzureRealtimeClient:
    """
    Azure OpenAI GPT-4o-realtime-mini client for speech-to-speech.
    Uses WebSocket for ultra-low latency bidirectional audio streaming.
    """
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_REALTIME_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_REALTIME_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_REALTIME_DEPLOYMENT", "gpt-4o-realtime-preview")
        self.api_version = "2024-10-01-preview"
        
        self.ws = None
        self.session_id = None
        self.is_connected = False
        
        # Audio settings for phone calls (8kHz mono for telephony)
        self.audio_format = "pcm16"  # 16-bit PCM
        self.sample_rate = 24000  # 24kHz for realtime API
        self.phone_sample_rate = 8000  # 8kHz for phone
        
        # Event handlers
        self.on_audio_delta: Optional[Callable] = None
        self.on_transcript: Optional[Callable] = None
        self.on_response_done: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        logger.info("✅ Azure OpenAI Realtime Client initialized")
    
    def _get_ws_url(self) -> str:
        """Get WebSocket URL for Azure OpenAI Realtime API"""
        if not self.endpoint:
            raise ValueError("AZURE_OPENAI_REALTIME_ENDPOINT not configured")
        
        # Format: wss://{endpoint}/openai/realtime?api-version={version}&deployment={deployment}
        base_url = self.endpoint.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base_url}/openai/realtime?api-version={self.api_version}&deployment={self.deployment}"
    
    async def connect(self, system_prompt: str, voice: str = "alloy") -> bool:
        """Establish WebSocket connection to Azure OpenAI Realtime API"""
        try:
            ws_url = self._get_ws_url()
            
            headers = {
                "api-key": self.api_key,
                "OpenAI-Beta": "realtime=v1"
            }
            
            logger.info(f"🔌 Connecting to Azure OpenAI Realtime: {self.deployment}")
            
            self.ws = await websockets.connect(
                ws_url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.is_connected = True
            logger.info("✅ WebSocket connected")
            
            # Configure session
            await self._configure_session(system_prompt, voice)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            self.is_connected = False
            return False
    
    async def _configure_session(self, system_prompt: str, voice: str = "alloy"):
        """Configure the realtime session"""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": system_prompt,
                "voice": voice,  # alloy, echo, shimmer
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.7,
                "max_response_output_tokens": 1024
            }
        }
        
        await self.ws.send(json.dumps(session_config))
        logger.info("✅ Session configured with VAD and audio transcription")
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to the realtime API"""
        if not self.is_connected or not self.ws:
            return
        
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            
            await self.ws.send(json.dumps(event))
            
        except Exception as e:
            logger.error(f"❌ Error sending audio: {e}")
    
    async def commit_audio(self):
        """Commit the audio buffer to trigger processing"""
        if not self.is_connected or not self.ws:
            return
        
        try:
            event = {"type": "input_audio_buffer.commit"}
            await self.ws.send(json.dumps(event))
        except Exception as e:
            logger.error(f"❌ Error committing audio: {e}")
    
    async def send_text(self, text: str, role: str = "user"):
        """Send text message (for system messages or text fallback)"""
        if not self.is_connected or not self.ws:
            return
        
        try:
            event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": text
                        }
                    ]
                }
            }
            
            await self.ws.send(json.dumps(event))
            
            # Trigger response
            await self.ws.send(json.dumps({"type": "response.create"}))
            
        except Exception as e:
            logger.error(f"❌ Error sending text: {e}")
    
    async def receive_events(self) -> Dict[str, Any]:
        """Receive and process events from the realtime API"""
        if not self.is_connected or not self.ws:
            return {"type": "error", "error": "Not connected"}
        
        try:
            message = await self.ws.recv()
            event = json.loads(message)
            return event
            
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            return {"type": "connection_closed"}
        except Exception as e:
            return {"type": "error", "error": str(e)}
    
    async def process_events_loop(self, state: TelephonicScreeningState):
        """Main event processing loop"""
        while self.is_connected:
            event = await self.receive_events()
            event_type = event.get("type", "")
            
            if event_type == "response.audio.delta":
                # Audio chunk received
                audio_b64 = event.get("delta", "")
                if audio_b64 and self.on_audio_delta:
                    audio_data = base64.b64decode(audio_b64)
                    await self.on_audio_delta(audio_data, state)
            
            elif event_type == "response.audio_transcript.delta":
                # AI transcript delta
                transcript_delta = event.get("delta", "")
                state.ai_transcript += transcript_delta
            
            elif event_type == "conversation.item.input_audio_transcription.completed":
                # User speech transcribed
                transcript = event.get("transcript", "")
                if transcript:
                    state.user_transcript = transcript
                    state.full_transcript.append({
                        "role": "user",
                        "content": transcript,
                        "timestamp": datetime.now().isoformat()
                    })
                    if self.on_transcript:
                        await self.on_transcript(transcript, "user", state)
            
            elif event_type == "response.done":
                # Response complete
                response = event.get("response", {})
                if self.on_response_done:
                    await self.on_response_done(response, state)
                
                # Store AI response in transcript
                if state.ai_transcript:
                    state.full_transcript.append({
                        "role": "assistant",
                        "content": state.ai_transcript,
                        "timestamp": datetime.now().isoformat()
                    })
                    state.ai_transcript = ""
            
            elif event_type == "error":
                error = event.get("error", {})
                logger.error(f"❌ Realtime API error: {error}")
                if self.on_error:
                    await self.on_error(error, state)
            
            elif event_type == "session.created":
                self.session_id = event.get("session", {}).get("id", "")
                logger.info(f"✅ Session created: {self.session_id}")
            
            elif event_type == "connection_closed":
                logger.info("📞 Connection closed")
                break
    
    async def disconnect(self):
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            logger.info("📞 Disconnected from Azure OpenAI Realtime")

# ============================================================================
# AZURE COMMUNICATION SERVICES PROVIDER (Enhanced for Media Streaming)
# ============================================================================

class AzureCommunicationProvider:
    """
    Azure Communication Services for telephony with media streaming.
    Bridges phone calls with Azure OpenAI Realtime API.
    """
    
    def __init__(self):
        self.connection_string = os.getenv("AZURE_COMMUNICATION_CONNECTION_STRING")
        self.phone_number = os.getenv("AZURE_COMMUNICATION_PHONE_NUMBER")
        self.callback_url = os.getenv("AZURE_COMMUNICATION_CALLBACK_URL")
        self.media_streaming_url = os.getenv("AZURE_COMMUNICATION_MEDIA_STREAMING_URL")
        
        self.client = None
        if AZURE_COMM_AVAILABLE and self.connection_string:
            try:
                self.client = CallAutomationClient.from_connection_string(self.connection_string)
                logger.info("✅ Azure Communication Services initialized")
            except Exception as e:
                logger.error(f"❌ ACS initialization error: {e}")
        else:
            logger.warning("⚠️ Azure Communication Services not configured")
    
    async def initiate_call_with_media_streaming(
        self,
        phone_number: str,
        callback_uri: str,
        media_streaming_uri: str
    ) -> Dict[str, Any]:
        """
        Initiate outbound call with bidirectional media streaming.
        This enables real-time audio to be streamed to/from Azure OpenAI Realtime.
        """
        if not self.client:
            return {"success": False, "error": "Azure Communication Services not configured"}
        
        try:
            target = PhoneNumberIdentifier(phone_number)
            caller = PhoneNumberIdentifier(self.phone_number)
            
            # Configure media streaming for realtime audio
            media_streaming_options = MediaStreamingOptions(
                transport_url=media_streaming_uri,
                transport_type=MediaStreamingTransportType.WEBSOCKET,
                content_type=MediaStreamingContentType.AUDIO,
                audio_channel_type=MediaStreamingAudioChannelType.MIXED,
                start_media_streaming=True
            )
            
            call_connection_properties = self.client.create_call(
                target_participant=target,
                source_caller_id_number=caller,
                callback_url=callback_uri,
                media_streaming=media_streaming_options
            )
            
            return {
                "success": True,
                "call_connection_id": call_connection_properties.call_connection_id,
                "server_call_id": call_connection_properties.server_call_id
            }
            
        except Exception as e:
            logger.error(f"❌ Call initiation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def start_media_streaming(self, call_connection_id: str) -> Dict[str, Any]:
        """Start media streaming on an active call"""
        if not self.client:
            return {"success": False, "error": "Not connected"}
        
        try:
            call_connection = self.client.get_call_connection(call_connection_id)
            call_connection.start_media_streaming()
            return {"success": True}
        except Exception as e:
            logger.error(f"❌ Start media streaming failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def stop_media_streaming(self, call_connection_id: str) -> Dict[str, Any]:
        """Stop media streaming"""
        if not self.client:
            return {"success": False, "error": "Not connected"}
        
        try:
            call_connection = self.client.get_call_connection(call_connection_id)
            call_connection.stop_media_streaming()
            return {"success": True}
        except Exception as e:
            logger.error(f"❌ Stop media streaming failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def hang_up(self, call_connection_id: str) -> Dict[str, Any]:
        """End the call"""
        if not self.client:
            return {"success": False, "error": "Not connected"}
        
        try:
            call_connection = self.client.get_call_connection(call_connection_id)
            call_connection.hang_up(is_for_everyone=True)
            return {"success": True}
        except Exception as e:
            logger.error(f"❌ Hang up failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def start_recording(self, server_call_id: str) -> Dict[str, Any]:
        """Start call recording"""
        if not self.client:
            return {"success": False, "error": "Not connected"}
        
        try:
            recording_properties = self.client.start_recording(
                call_locator=server_call_id,
                recording_content_type="audio",
                recording_channel_type="mixed",
                recording_format_type="mp3"
            )
            
            return {
                "success": True,
                "recording_id": recording_properties.recording_id
            }
        except Exception as e:
            logger.error(f"❌ Recording start failed: {e}")
            return {"success": False, "error": str(e)}

# ============================================================================
# REALTIME SCREENING ORCHESTRATOR
# ============================================================================

class RealtimeScreeningOrchestrator:
    """
    Main orchestrator for realtime telephonic screening.
    Uses Azure GPT-4o-realtime-mini for speech-to-speech conversations.
    """
    
    def __init__(self):
        logger.info("📞 Initializing Realtime Telephonic Screening Orchestrator...")
        
        # Initialize components
        self.acs_provider = AzureCommunicationProvider()
        self.realtime_client = AzureRealtimeClient()
        
        # Try to get existing modules
        try:
            self.memory_layer = get_memory_layer()
        except:
            self.memory_layer = None
        
        try:
            self.behavior_simulator = get_human_behavior_simulator()
        except:
            self.behavior_simulator = None
        
        # Active calls tracking
        self.active_calls: Dict[str, TelephonicScreeningState] = {}
        
        logger.info("✅ Realtime Telephonic Screening Orchestrator initialized")
    
    def _build_screening_system_prompt(self, state: TelephonicScreeningState) -> str:
        """Build comprehensive system prompt for screening"""
        candidate_name = state.candidate_profile.get("name", "the candidate")
        company_name = state.company_info.get("name", "our company")
        job_title = state.job_requirements.get("title", "the position")
        job_description = state.job_requirements.get("description", "")
        required_skills = state.job_requirements.get("skills", [])
        resume_summary = state.resume_analysis.get("summary", "")
        
        language_instruction = ""
        if state.language == "hi-IN":
            language_instruction = """
LANGUAGE: Speak in Hindi (Hinglish is okay - mix of Hindi and English technical terms).
Use natural Hindi phrases like "अच्छा", "बढ़िया", "समझ गया/गई" for acknowledgments.
"""
        else:
            language_instruction = """
LANGUAGE: Speak in English with an Indian context. Use natural phrases.
"""
        
        return f"""You are an AI phone screener conducting a telephonic interview for {company_name}.
You are calling {candidate_name} regarding the {job_title} position.

{language_instruction}

JOB DETAILS:
- Title: {job_title}
- Description: {job_description}
- Required Skills: {', '.join(required_skills) if required_skills else 'Not specified'}

CANDIDATE BACKGROUND:
{resume_summary if resume_summary else 'Resume details will be discussed during the call.'}

YOUR PERSONALITY:
- Professional but friendly
- Patient and encouraging
- Natural conversational flow (not robotic)
- Use filler words naturally ("So...", "Well...", "Hmm, I see...")
- Acknowledge responses before asking next question
- Keep responses concise (phone conversations should be brief)

SCREENING FLOW:
1. GREETING: Confirm you're speaking with the right person, introduce yourself
2. CONFIRM TIME: Ask if they have 10-15 minutes for a quick screening call
3. AVAILABILITY: Ask about notice period, current employment status, availability to join
4. EXPERIENCE: Verify key experience points from their resume
5. SKILLS: Ask about proficiency in 2-3 key required skills with examples
6. MOTIVATION: Why are they interested in this role?
7. SALARY: Ask about current CTC and expected CTC
8. QUESTIONS: Ask if they have any questions about the role
9. CLOSING: Thank them and explain next steps

IMPORTANT RULES:
- Listen carefully to responses
- If answer is unclear, ask for clarification politely
- Note any red flags (inconsistencies, evasive answers, attitude issues)
- Note green flags (enthusiasm, good communication, relevant experience)
- Keep the call focused (10-15 minutes max)
- Don't interrupt unless they're going off-topic
- Be culturally sensitive (Indian context)

Start with a warm greeting and confirm the candidate's identity."""
    
    async def start_screening_call(
        self,
        candidate_id: str,
        job_id: str,
        phone_number: str,
        candidate_profile: Dict[str, Any],
        job_requirements: Dict[str, Any],
        company_info: Dict[str, Any],
        resume_analysis: Optional[Dict[str, Any]] = None,
        language: str = "en-IN",
        voice: str = "alloy"
    ) -> Dict[str, Any]:
        """Start a new realtime telephonic screening call"""
        
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Create initial state
        state = TelephonicScreeningState(
            call_id=call_id,
            session_id=session_id,
            candidate_id=candidate_id,
            job_id=job_id,
            company_id=company_info.get("id", ""),
            phone_number=phone_number,
            candidate_profile=candidate_profile,
            job_requirements=job_requirements,
            company_info=company_info,
            resume_analysis=resume_analysis or {},
            language=language,
            voice=voice,
            call_start_time=datetime.now()
        )
        
        # Store active call
        self.active_calls[call_id] = state
        
        try:
            # 1. Build system prompt
            system_prompt = self._build_screening_system_prompt(state)
            
            # 2. Connect to Azure OpenAI Realtime
            realtime_connected = await self.realtime_client.connect(
                system_prompt=system_prompt,
                voice=voice
            )
            
            if not realtime_connected:
                raise Exception("Failed to connect to Azure OpenAI Realtime API")
            
            state.call_status = CallStatus.CONNECTED.value
            state.realtime_session_id = self.realtime_client.session_id
            
            # 3. Set up event handlers
            self.realtime_client.on_audio_delta = self._handle_audio_output
            self.realtime_client.on_transcript = self._handle_transcript
            self.realtime_client.on_response_done = self._handle_response_done
            self.realtime_client.on_error = self._handle_error
            
            # 4. Initiate phone call with media streaming
            callback_url = f"{os.getenv('AZURE_COMMUNICATION_CALLBACK_URL', '')}/api/telephonic-screening/webhook/{call_id}"
            media_streaming_url = f"{os.getenv('AZURE_COMMUNICATION_MEDIA_STREAMING_URL', '')}/api/telephonic-screening/media/{call_id}"
            
            call_result = await self.acs_provider.initiate_call_with_media_streaming(
                phone_number=phone_number,
                callback_uri=callback_url,
                media_streaming_uri=media_streaming_url
            )
            
            if call_result.get("success"):
                state.call_connection_id = call_result.get("call_connection_id", "")
                state.server_call_id = call_result.get("server_call_id", "")
                state.call_status = CallStatus.RINGING.value
                logger.info(f"✅ Call initiated: {state.call_connection_id}")
                
                # Start recording
                await self.acs_provider.start_recording(state.server_call_id)
                
            else:
                raise Exception(call_result.get("error", "Failed to initiate call"))
            
            # 5. Start processing events (this runs in background)
            asyncio.create_task(self.realtime_client.process_events_loop(state))
            
            return {
                "success": True,
                "call_id": call_id,
                "session_id": session_id,
                "status": state.call_status,
                "message": "Call initiated successfully"
            }
            
        except Exception as e:
            logger.error(f"❌ Screening call error: {e}")
            state.call_status = CallStatus.FAILED.value
            state.errors.append(str(e))
            
            return {
                "success": False,
                "call_id": call_id,
                "error": str(e)
            }
    
    async def handle_incoming_audio(self, call_id: str, audio_data: bytes):
        """Handle incoming audio from phone call (via media streaming)"""
        if call_id not in self.active_calls:
            return
        
        state = self.active_calls[call_id]
        
        # Resample audio if needed (phone is 8kHz, realtime API is 24kHz)
        resampled_audio = self._resample_audio(audio_data, 8000, 24000)
        
        # Send to Azure OpenAI Realtime
        await self.realtime_client.send_audio(resampled_audio)
    
    async def _handle_audio_output(self, audio_data: bytes, state: TelephonicScreeningState):
        """Handle audio output from Azure OpenAI Realtime"""
        # Resample from 24kHz to 8kHz for phone
        resampled_audio = self._resample_audio(audio_data, 24000, 8000)
        
        # Store in output buffer
        state.output_audio_buffer += resampled_audio
        
        # In real implementation, stream this to the phone call via ACS media streaming
        # This would be handled by the media streaming WebSocket connection
    
    async def _handle_transcript(self, transcript: str, role: str, state: TelephonicScreeningState):
        """Handle transcribed text"""
        logger.info(f"📝 {role.upper()}: {transcript}")
        
        # Store in conversation history
        state.conversation_history.append({
            "role": role,
            "content": transcript,
            "stage": state.current_stage,
            "timestamp": datetime.now().isoformat()
        })
        
        # Analyze response if from user
        if role == "user" and transcript:
            await self._analyze_response(transcript, state)
    
    async def _handle_response_done(self, response: Dict[str, Any], state: TelephonicScreeningState):
        """Handle completed AI response"""
        state.questions_asked += 1
        
        # Check if we should move to next stage
        await self._update_screening_stage(state)
    
    async def _handle_error(self, error: Dict[str, Any], state: TelephonicScreeningState):
        """Handle errors from realtime API"""
        error_msg = error.get("message", str(error))
        state.errors.append(error_msg)
        logger.error(f"❌ Realtime error: {error_msg}")
    
    async def _analyze_response(self, response: str, state: TelephonicScreeningState):
        """Analyze candidate's response for insights"""
        # Extract key information based on current stage
        stage = state.current_stage
        
        # Simple keyword extraction (can be enhanced with NLP)
        response_lower = response.lower()
        
        # Notice period detection
        if stage == ScreeningStage.AVAILABILITY_CHECK.value:
            if "immediate" in response_lower:
                state.notice_period = "Immediate"
            elif "month" in response_lower:
                # Extract number
                import re
                match = re.search(r'(\d+)\s*month', response_lower)
                if match:
                    state.notice_period = f"{match.group(1)} months"
        
        # CTC detection
        if stage == ScreeningStage.SALARY_EXPECTATION.value:
            import re
            ctc_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lpa|lac)', response_lower)
            if ctc_match:
                if not state.current_ctc:
                    state.current_ctc = f"{ctc_match.group(1)} LPA"
                else:
                    state.expected_ctc = f"{ctc_match.group(1)} LPA"
        
        # Sentiment analysis (simple)
        positive_words = ['great', 'excited', 'love', 'passionate', 'enthusiastic', 'enjoy']
        negative_words = ['hate', 'dislike', 'boring', 'difficult', 'problem']
        
        pos_count = sum(1 for word in positive_words if word in response_lower)
        neg_count = sum(1 for word in negative_words if word in response_lower)
        
        if pos_count > neg_count:
            state.sentiment_trajectory.append({
                "stage": stage,
                "sentiment": "positive",
                "timestamp": datetime.now().isoformat()
            })
            state.engagement_score = min(1.0, state.engagement_score + 0.1)
        elif neg_count > pos_count:
            state.sentiment_trajectory.append({
                "stage": stage,
                "sentiment": "negative",
                "timestamp": datetime.now().isoformat()
            })
    
    async def _update_screening_stage(self, state: TelephonicScreeningState):
        """Update screening stage based on progress"""
        stage_order = [
            ScreeningStage.GREETING.value,
            ScreeningStage.INTRODUCTION.value,
            ScreeningStage.AVAILABILITY_CHECK.value,
            ScreeningStage.EXPERIENCE_VERIFICATION.value,
            ScreeningStage.SKILL_ASSESSMENT.value,
            ScreeningStage.MOTIVATION_CHECK.value,
            ScreeningStage.SALARY_EXPECTATION.value,
            ScreeningStage.QUESTIONS_FROM_CANDIDATE.value,
            ScreeningStage.CLOSING.value
        ]
        
        current_index = stage_order.index(state.current_stage) if state.current_stage in stage_order else 0
        questions_per_stage = state.max_questions // len(stage_order)
        
        # Move to next stage based on questions asked
        expected_stage_index = min(state.questions_asked // max(1, questions_per_stage), len(stage_order) - 1)
        
        if expected_stage_index > current_index:
            state.current_stage = stage_order[expected_stage_index]
            logger.info(f"📍 Stage updated to: {state.current_stage}")
    
    def _resample_audio(self, audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Simple audio resampling (linear interpolation)"""
        if from_rate == to_rate:
            return audio_data
        
        # Convert bytes to samples (16-bit PCM)
        samples = struct.unpack(f'{len(audio_data)//2}h', audio_data)
        
        # Calculate resampling ratio
        ratio = to_rate / from_rate
        new_length = int(len(samples) * ratio)
        
        # Linear interpolation
        resampled = []
        for i in range(new_length):
            src_idx = i / ratio
            idx = int(src_idx)
            frac = src_idx - idx
            
            if idx + 1 < len(samples):
                sample = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
            else:
                sample = samples[-1]
            
            resampled.append(max(-32768, min(32767, sample)))
        
        return struct.pack(f'{len(resampled)}h', *resampled)
    
    async def end_call(self, call_id: str) -> Dict[str, Any]:
        """End the screening call and generate results"""
        if call_id not in self.active_calls:
            return {"success": False, "error": "Call not found"}
        
        state = self.active_calls[call_id]
        
        try:
            # Disconnect realtime API
            await self.realtime_client.disconnect()
            
            # Hang up phone call
            if state.call_connection_id:
                await self.acs_provider.hang_up(state.call_connection_id)
            
            state.call_status = CallStatus.COMPLETED.value
            state.call_end_time = datetime.now()
            
            if state.call_start_time:
                state.call_duration_seconds = int((state.call_end_time - state.call_start_time).total_seconds())
            
            # Calculate outcome
            state = self._calculate_outcome(state)
            
            # Store in memory
            if self.memory_layer:
                try:
                    await self.memory_layer.store_interaction(
                        candidate_id=state.candidate_id,
                        company_id=state.company_id,
                        job_id=state.job_id,
                        interaction={
                            "type": "telephonic_screening_realtime",
                            "call_id": state.call_id,
                            "duration": state.call_duration_seconds,
                            "outcome": state.outcome,
                            "score": state.overall_score,
                            "full_transcript": state.full_transcript,
                            "stage_scores": state.stage_scores
                        },
                        behavioral_signals={
                            "sentiment_trajectory": state.sentiment_trajectory,
                            "confidence_score": state.confidence_score,
                            "engagement_score": state.engagement_score
                        }
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Memory storage failed: {e}")
            
            # Create result
            result = self._create_screening_result(state)
            
            # Cleanup
            del self.active_calls[call_id]
            
            return {
                "success": True,
                "call_id": call_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"❌ Error ending call: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_outcome(self, state: TelephonicScreeningState) -> TelephonicScreeningState:
        """Calculate screening outcome"""
        # Calculate engagement score
        if state.conversation_history:
            user_responses = [h for h in state.conversation_history if h.get("role") == "user"]
            avg_response_length = sum(len(h.get("content", "")) for h in user_responses) / max(1, len(user_responses))
            state.engagement_score = min(1.0, avg_response_length / 100)
        
        # Calculate confidence score based on sentiment
        positive_sentiments = sum(1 for s in state.sentiment_trajectory if s.get("sentiment") == "positive")
        total_sentiments = len(state.sentiment_trajectory)
        state.confidence_score = positive_sentiments / max(1, total_sentiments)
        
        # Overall score
        state.overall_score = (state.engagement_score * 0.4 + state.confidence_score * 0.3 + 0.3) * 100
        
        # Add flags
        score_adjustment = len(state.green_flags) * 5 - len(state.red_flags) * 10
        state.overall_score = min(100, max(0, state.overall_score + score_adjustment))
        
        # Determine outcome
        if state.overall_score >= 75:
            state.outcome = CallOutcome.QUALIFIED.value
            state.recommendation = "Recommend for technical interview"
            state.next_steps = ["Schedule technical interview", "Send interview preparation materials"]
        elif state.overall_score >= 60:
            state.outcome = CallOutcome.REQUIRES_REVIEW.value
            state.recommendation = "Manual review recommended"
            state.next_steps = ["Review call recording", "Discuss with hiring manager"]
        else:
            state.outcome = CallOutcome.NOT_QUALIFIED.value
            state.recommendation = "Does not meet minimum requirements"
            state.next_steps = ["Send rejection email", "Add to talent pool for future roles"]
        
        return state
    
    def _create_screening_result(self, state: TelephonicScreeningState) -> Dict[str, Any]:
        """Create final screening result"""
        return {
            "call_id": state.call_id,
            "candidate_id": state.candidate_id,
            "job_id": state.job_id,
            "outcome": state.outcome,
            "overall_score": state.overall_score,
            "stage_scores": state.stage_scores,
            "call_duration_seconds": state.call_duration_seconds,
            "questions_asked": state.questions_asked,
            "data_collected": {
                "availability_confirmed": state.availability_confirmed,
                "notice_period": state.notice_period,
                "current_ctc": state.current_ctc,
                "expected_ctc": state.expected_ctc,
                "relocation_willing": state.relocation_willing
            },
            "analysis": {
                "red_flags": state.red_flags,
                "green_flags": state.green_flags,
                "confidence_score": state.confidence_score,
                "engagement_score": state.engagement_score,
                "sentiment_trajectory": state.sentiment_trajectory
            },
            "recommendation": state.recommendation,
            "next_steps": state.next_steps,
            "recording_url": state.recording_url,
            "full_transcript": state.full_transcript,
            "conversation_summary": {
                "total_turns": len(state.conversation_history),
                "stages_completed": list(set(h.get("stage", "") for h in state.conversation_history))
            }
        }
    
    def get_call_status(self, call_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a call"""
        if call_id not in self.active_calls:
            return None
        
        state = self.active_calls[call_id]
        return {
            "call_id": call_id,
            "status": state.call_status,
            "stage": state.current_stage,
            "questions_asked": state.questions_asked,
            "duration": (datetime.now() - state.call_start_time).total_seconds() if state.call_start_time else 0,
            "transcript_length": len(state.full_transcript)
        }
    
    async def handle_call_event(
        self,
        call_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Azure Communication Services webhook events"""
        if call_id not in self.active_calls:
            return {"success": False, "error": "Call not found"}
        
        state = self.active_calls[call_id]
        
        if event_type == "CallConnected":
            state.call_status = CallStatus.STREAMING.value
            logger.info(f"✅ Call connected, streaming started: {call_id}")
            
        elif event_type == "MediaStreamingStarted":
            state.call_status = CallStatus.IN_PROGRESS.value
            logger.info(f"🎤 Media streaming active: {call_id}")
            
        elif event_type == "CallDisconnected":
            await self.end_call(call_id)
            
        self.active_calls[call_id] = state
        return {"success": True, "state": state.current_stage}

# ============================================================================
# BACKWARD COMPATIBILITY - Keep old class names working
# ============================================================================

TelephonicScreeningOrchestrator = RealtimeScreeningOrchestrator
AzureSpeechService = None  # Deprecated - using realtime API instead
ScreeningConversationEngine = None  # Deprecated - using realtime API instead

# ============================================================================
# SINGLETON AND FACTORY
# ============================================================================

_telephonic_orchestrator = None

def get_telephonic_screening_orchestrator() -> RealtimeScreeningOrchestrator:
    """Get singleton telephonic screening orchestrator"""
    global _telephonic_orchestrator
    if _telephonic_orchestrator is None:
        _telephonic_orchestrator = RealtimeScreeningOrchestrator()
    return _telephonic_orchestrator

async def initiate_telephonic_screening(
    candidate_id: str,
    job_id: str,
    phone_number: str,
    candidate_profile: Dict[str, Any],
    job_requirements: Dict[str, Any],
    company_info: Dict[str, Any],
    resume_analysis: Optional[Dict[str, Any]] = None,
    language: str = "en-IN",
    voice: str = "alloy"
) -> Dict[str, Any]:
    """Convenience function to start telephonic screening"""
    orchestrator = get_telephonic_screening_orchestrator()
    return await orchestrator.start_screening_call(
        candidate_id=candidate_id,
        job_id=job_id,
        phone_number=phone_number,
        candidate_profile=candidate_profile,
        job_requirements=job_requirements,
        company_info=company_info,
        resume_analysis=resume_analysis,
        language=language,
        voice=voice
    )
