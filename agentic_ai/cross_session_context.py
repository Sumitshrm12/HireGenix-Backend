"""
🔄 CROSS-SESSION CONTEXT SHARING - Multi-Round Interview Intelligence
Enables seamless context passing between interview rounds (HR, Technical, Cultural, Final).

Features:
- Round-to-round handoff summaries
- Accumulated insights aggregation
- Contradiction detection across rounds
- Progressive depth tracking
- Interviewer-specific context
- Candidate journey mapping
- Decision consensus building
- Historical pattern matching

Tech Stack:
- LangGraph for state management
- Redis for persistent storage
- Semantic embeddings for context matching
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewRound(str, Enum):
    """Types of interview rounds"""
    SCREENING = "screening"
    HR = "hr"
    TECHNICAL_1 = "technical_1"
    TECHNICAL_2 = "technical_2"
    CODING = "coding"
    SYSTEM_DESIGN = "system_design"
    CULTURAL = "cultural"
    MANAGER = "manager"
    FINAL = "final"
    OFFER = "offer"


class HandoffPriority(str, Enum):
    """Priority of handoff notes"""
    CRITICAL = "critical"  # Must be addressed in next round
    HIGH = "high"  # Important to explore
    NORMAL = "normal"  # Good to know
    LOW = "low"  # Background context


class CrossSessionState(BaseModel):
    """State for cross-session context management"""
    # Session identifiers
    candidate_id: str = ""
    job_id: str = ""
    company_id: str = ""
    session_id: str = ""
    
    # Round information
    current_round: str = InterviewRound.SCREENING.value
    previous_rounds: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Aggregated context
    candidate_profile: Dict[str, Any] = Field(default_factory=dict)
    verified_skills: Dict[str, float] = Field(default_factory=dict)
    unverified_claims: List[str] = Field(default_factory=list)
    
    # Handoff information
    handoff_notes: List[Dict[str, Any]] = Field(default_factory=list)
    areas_to_probe: List[Dict[str, Any]] = Field(default_factory=list)
    red_flags: List[Dict[str, Any]] = Field(default_factory=list)
    green_flags: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Cross-round analysis
    consistency_score: float = 0.0
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    emerging_patterns: List[str] = Field(default_factory=list)
    
    # Journey tracking
    engagement_trajectory: List[float] = Field(default_factory=list)
    confidence_trajectory: List[float] = Field(default_factory=list)
    skill_depth_by_round: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Consensus building
    round_decisions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cumulative_score: float = 0.0
    hiring_probability: float = 0.5
    
    # Context for current round
    briefing_for_interviewer: str = ""
    suggested_questions: List[str] = Field(default_factory=list)
    topics_covered: List[str] = Field(default_factory=list)
    topics_to_cover: List[str] = Field(default_factory=list)


class CrossSessionContextManager:
    """
    Manages context sharing between interview rounds,
    ensuring seamless handoffs and progressive evaluation.
    """
    
    def __init__(self):
        logger.info("🔄 Initializing Cross-Session Context Manager...")
        
        # LLM for intelligent context processing
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.3
        )
        
        # Redis for persistent cross-session storage
        self.redis_client = None
        self._init_redis()
        
        # Sentence transformer for semantic matching
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build context workflow
        self.workflow = self._build_context_workflow()
        
        # Round progression map
        self.round_sequence = {
            InterviewRound.SCREENING.value: [InterviewRound.HR.value, InterviewRound.TECHNICAL_1.value],
            InterviewRound.HR.value: [InterviewRound.TECHNICAL_1.value],
            InterviewRound.TECHNICAL_1.value: [InterviewRound.TECHNICAL_2.value, InterviewRound.CODING.value],
            InterviewRound.TECHNICAL_2.value: [InterviewRound.SYSTEM_DESIGN.value, InterviewRound.CULTURAL.value],
            InterviewRound.CODING.value: [InterviewRound.SYSTEM_DESIGN.value],
            InterviewRound.SYSTEM_DESIGN.value: [InterviewRound.MANAGER.value],
            InterviewRound.CULTURAL.value: [InterviewRound.MANAGER.value],
            InterviewRound.MANAGER.value: [InterviewRound.FINAL.value],
            InterviewRound.FINAL.value: [InterviewRound.OFFER.value]
        }
        
        logger.info("✅ Cross-Session Context Manager initialized")
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}. Using in-memory fallback.")
            self.redis_client = None
            self._fallback_store = {}
    
    def _generate_context_key(self, candidate_id: str, job_id: str) -> str:
        """Generate unique key for candidate-job context"""
        return f"hiregenix:cross_session:{candidate_id}:{job_id}"
    
    def _build_context_workflow(self) -> StateGraph:
        """Build LangGraph workflow for context management"""
        workflow = StateGraph(CrossSessionState)
        
        # Define nodes
        workflow.add_node("load_previous_context", self._load_previous_context)
        workflow.add_node("analyze_cross_round_consistency", self._analyze_cross_round_consistency)
        workflow.add_node("aggregate_skill_verification", self._aggregate_skill_verification)
        workflow.add_node("generate_handoff_briefing", self._generate_handoff_briefing)
        workflow.add_node("suggest_questions", self._suggest_questions)
        workflow.add_node("update_hiring_probability", self._update_hiring_probability)
        workflow.add_node("save_context", self._save_context)
        
        # Define edges
        workflow.set_entry_point("load_previous_context")
        workflow.add_edge("load_previous_context", "analyze_cross_round_consistency")
        workflow.add_edge("analyze_cross_round_consistency", "aggregate_skill_verification")
        workflow.add_edge("aggregate_skill_verification", "generate_handoff_briefing")
        workflow.add_edge("generate_handoff_briefing", "suggest_questions")
        workflow.add_edge("suggest_questions", "update_hiring_probability")
        workflow.add_edge("update_hiring_probability", "save_context")
        workflow.add_edge("save_context", END)
        
        return workflow.compile()
    
    async def _load_previous_context(self, state: CrossSessionState) -> CrossSessionState:
        """Load context from previous interview rounds"""
        try:
            context_key = self._generate_context_key(state.candidate_id, state.job_id)
            
            if self.redis_client:
                stored_data = await self.redis_client.get(context_key)
                if stored_data:
                    previous_context = json.loads(stored_data)
                    state.previous_rounds = previous_context.get("rounds", [])
                    state.candidate_profile = previous_context.get("profile", {})
                    state.verified_skills = previous_context.get("verified_skills", {})
                    state.red_flags = previous_context.get("red_flags", [])
                    state.green_flags = previous_context.get("green_flags", [])
                    state.topics_covered = previous_context.get("topics_covered", [])
            elif hasattr(self, '_fallback_store'):
                if context_key in self._fallback_store:
                    previous_context = self._fallback_store[context_key]
                    state.previous_rounds = previous_context.get("rounds", [])
                    state.candidate_profile = previous_context.get("profile", {})
                    state.verified_skills = previous_context.get("verified_skills", {})
                    state.red_flags = previous_context.get("red_flags", [])
                    state.green_flags = previous_context.get("green_flags", [])
                    state.topics_covered = previous_context.get("topics_covered", [])
            
            logger.info(f"📥 Loaded context from {len(state.previous_rounds)} previous rounds")
            return state
            
        except Exception as e:
            logger.error(f"❌ Context load error: {e}")
            return state
    
    async def _analyze_cross_round_consistency(self, state: CrossSessionState) -> CrossSessionState:
        """Analyze consistency across interview rounds"""
        if len(state.previous_rounds) < 1:
            state.consistency_score = 1.0  # First round, assume consistent
            return state
        
        try:
            # Collect all answers/statements from previous rounds
            all_statements = []
            for round_data in state.previous_rounds:
                statements = round_data.get("key_statements", [])
                all_statements.extend([{
                    "round": round_data.get("round_type"),
                    "statement": s
                } for s in statements])
            
            if not all_statements:
                state.consistency_score = 1.0
                return state
            
            prompt = f"""
            Analyze the consistency of this candidate's statements across interview rounds.
            
            STATEMENTS BY ROUND:
            {json.dumps(all_statements, indent=2)}
            
            Check for:
            1. Contradicting facts (different dates, numbers, companies)
            2. Inconsistent skill claims
            3. Changing stories about same events
            4. Varying confidence levels on same topics
            
            Return JSON:
            {{
                "consistency_score": 0.85,
                "contradictions": [
                    {{
                        "round_1": "technical_1",
                        "statement_1": "what they said",
                        "round_2": "cultural",
                        "statement_2": "conflicting statement",
                        "severity": "minor|moderate|major"
                    }}
                ],
                "patterns": ["pattern1", "pattern2"],
                "analysis": "Brief analysis"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.consistency_score = result.get("consistency_score", 0.8)
            state.contradictions = result.get("contradictions", [])
            state.emerging_patterns = result.get("patterns", [])
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Consistency analysis error: {e}")
            state.consistency_score = 0.7
            return state
    
    async def _aggregate_skill_verification(self, state: CrossSessionState) -> CrossSessionState:
        """Aggregate skill verification across rounds"""
        try:
            # Collect skill assessments from all rounds
            skill_assessments = {}
            for round_data in state.previous_rounds:
                round_skills = round_data.get("skills_assessed", {})
                round_type = round_data.get("round_type", "unknown")
                
                for skill, score in round_skills.items():
                    if skill not in skill_assessments:
                        skill_assessments[skill] = []
                    skill_assessments[skill].append({
                        "round": round_type,
                        "score": score
                    })
            
            # Calculate weighted averages (later rounds weighted higher)
            for skill, assessments in skill_assessments.items():
                if len(assessments) == 1:
                    state.verified_skills[skill] = assessments[0]["score"]
                else:
                    # Later rounds weighted more heavily
                    weights = [1.0 + (i * 0.2) for i in range(len(assessments))]
                    total_weight = sum(weights)
                    weighted_sum = sum(a["score"] * w for a, w in zip(assessments, weights))
                    state.verified_skills[skill] = weighted_sum / total_weight
            
            # Track skill depth by round
            for round_data in state.previous_rounds:
                round_type = round_data.get("round_type", "unknown")
                state.skill_depth_by_round[round_type] = round_data.get("skills_assessed", {})
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Skill aggregation error: {e}")
            return state
    
    async def _generate_handoff_briefing(self, state: CrossSessionState) -> CrossSessionState:
        """Generate briefing for the next interviewer"""
        try:
            prompt = f"""
            Generate a briefing for the next interviewer in this interview process.
            
            CANDIDATE PROFILE:
            {json.dumps(state.candidate_profile, indent=2)}
            
            CURRENT ROUND: {state.current_round}
            
            PREVIOUS ROUNDS:
            {json.dumps([{
                "round": r.get("round_type"),
                "decision": r.get("decision"),
                "highlights": r.get("highlights", [])
            } for r in state.previous_rounds], indent=2)}
            
            VERIFIED SKILLS:
            {json.dumps(state.verified_skills, indent=2)}
            
            RED FLAGS:
            {json.dumps(state.red_flags, indent=2)}
            
            GREEN FLAGS:
            {json.dumps(state.green_flags, indent=2)}
            
            TOPICS ALREADY COVERED:
            {state.topics_covered}
            
            CONSISTENCY SCORE: {state.consistency_score}
            {f"CONTRADICTIONS FOUND: {json.dumps(state.contradictions)}" if state.contradictions else ""}
            
            Generate a concise, actionable briefing that:
            1. Summarizes what's known about the candidate
            2. Highlights what to probe further
            3. Notes any concerns to address
            4. Suggests focus areas for this round type
            
            Return JSON:
            {{
                "briefing": "2-3 paragraph briefing for interviewer",
                "key_strengths": ["strength1", "strength2"],
                "areas_to_probe": [
                    {{
                        "topic": "topic name",
                        "reason": "why probe this",
                        "priority": "critical|high|normal"
                    }}
                ],
                "topics_to_avoid": ["already thoroughly covered topics"],
                "interviewer_tips": ["tip1", "tip2"]
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.briefing_for_interviewer = result.get("briefing", "")
            state.areas_to_probe = result.get("areas_to_probe", [])
            state.topics_to_cover = [p["topic"] for p in result.get("areas_to_probe", [])]
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Briefing generation error: {e}")
            state.briefing_for_interviewer = "No previous context available. Standard interview recommended."
            return state
    
    async def _suggest_questions(self, state: CrossSessionState) -> CrossSessionState:
        """Suggest specific questions for this round"""
        try:
            # Determine focus based on round type
            round_focus = {
                InterviewRound.SCREENING.value: "basic qualifications, salary expectations, availability",
                InterviewRound.HR.value: "cultural fit, career goals, work style",
                InterviewRound.TECHNICAL_1.value: "core technical skills, problem-solving approach",
                InterviewRound.TECHNICAL_2.value: "deep technical expertise, complex scenarios",
                InterviewRound.CODING.value: "live coding ability, code quality, debugging",
                InterviewRound.SYSTEM_DESIGN.value: "architecture skills, scalability thinking",
                InterviewRound.CULTURAL.value: "values alignment, collaboration style",
                InterviewRound.MANAGER.value: "team fit, growth potential, expectations",
                InterviewRound.FINAL.value: "executive presence, strategic thinking"
            }
            
            focus = round_focus.get(state.current_round, "general assessment")
            
            prompt = f"""
            Generate targeted questions for this interview round.
            
            ROUND TYPE: {state.current_round}
            ROUND FOCUS: {focus}
            
            CANDIDATE PROFILE:
            {json.dumps(state.candidate_profile, indent=2)}
            
            AREAS TO PROBE (from previous rounds):
            {json.dumps(state.areas_to_probe, indent=2)}
            
            ALREADY COVERED TOPICS:
            {state.topics_covered}
            
            UNVERIFIED CLAIMS:
            {state.unverified_claims}
            
            Generate 5-7 targeted questions that:
            1. Fit the round type and focus
            2. Address areas flagged by previous interviewers
            3. Don't repeat thoroughly covered topics
            4. Probe unverified claims
            5. Are natural and conversational
            
            Return JSON:
            {{
                "questions": [
                    {{
                        "question": "The actual question",
                        "purpose": "What we're trying to learn",
                        "follow_up_if_surface": "Follow-up if answer is shallow"
                    }}
                ]
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.suggested_questions = [
                q["question"] for q in result.get("questions", [])
            ]
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Question suggestion error: {e}")
            return state
    
    async def _update_hiring_probability(self, state: CrossSessionState) -> CrossSessionState:
        """Update cumulative hiring probability"""
        try:
            if not state.previous_rounds:
                state.hiring_probability = 0.5
                return state
            
            # Collect round decisions
            positive_signals = 0
            negative_signals = 0
            
            for round_data in state.previous_rounds:
                decision = round_data.get("decision", "proceed")
                if decision in ["strong_yes", "yes"]:
                    positive_signals += 1.5 if decision == "strong_yes" else 1.0
                elif decision in ["no", "strong_no"]:
                    negative_signals += 1.5 if decision == "strong_no" else 1.0
                # "proceed" is neutral
            
            total_rounds = len(state.previous_rounds)
            
            # Calculate base probability
            if total_rounds > 0:
                net_signal = positive_signals - negative_signals
                # Sigmoid-like scaling
                state.hiring_probability = 0.5 + (net_signal / (total_rounds * 2)) * 0.4
            
            # Adjust for consistency
            state.hiring_probability *= state.consistency_score
            
            # Cap between 0.1 and 0.95
            state.hiring_probability = max(0.1, min(0.95, state.hiring_probability))
            
            # Calculate cumulative score
            skill_avg = sum(state.verified_skills.values()) / len(state.verified_skills) if state.verified_skills else 0.5
            state.cumulative_score = (state.hiring_probability * 0.6 + skill_avg * 0.4) * 100
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Probability update error: {e}")
            return state
    
    async def _save_context(self, state: CrossSessionState) -> CrossSessionState:
        """Save context for future rounds"""
        try:
            context_key = self._generate_context_key(state.candidate_id, state.job_id)
            
            context_to_save = {
                "candidate_id": state.candidate_id,
                "job_id": state.job_id,
                "rounds": state.previous_rounds,
                "profile": state.candidate_profile,
                "verified_skills": state.verified_skills,
                "red_flags": state.red_flags,
                "green_flags": state.green_flags,
                "topics_covered": state.topics_covered,
                "consistency_score": state.consistency_score,
                "hiring_probability": state.hiring_probability,
                "cumulative_score": state.cumulative_score,
                "last_updated": datetime.now().isoformat()
            }
            
            if self.redis_client:
                await self.redis_client.set(
                    context_key,
                    json.dumps(context_to_save),
                    ex=60 * 60 * 24 * 30  # 30 days TTL
                )
            elif hasattr(self, '_fallback_store'):
                self._fallback_store[context_key] = context_to_save
            
            logger.info(f"💾 Saved cross-session context for {state.candidate_id}")
            return state
            
        except Exception as e:
            logger.error(f"❌ Context save error: {e}")
            return state
    
    async def get_context_for_round(
        self,
        candidate_id: str,
        job_id: str,
        round_type: str,
        candidate_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get context for a specific interview round"""
        try:
            initial_state = CrossSessionState(
                candidate_id=candidate_id,
                job_id=job_id,
                current_round=round_type,
                candidate_profile=candidate_profile or {}
            )
            
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "success": True,
                "briefing": final_state.briefing_for_interviewer,
                "suggested_questions": final_state.suggested_questions,
                "areas_to_probe": final_state.areas_to_probe,
                "verified_skills": final_state.verified_skills,
                "red_flags": final_state.red_flags,
                "green_flags": final_state.green_flags,
                "topics_covered": final_state.topics_covered,
                "topics_to_cover": final_state.topics_to_cover,
                "consistency_score": final_state.consistency_score,
                "hiring_probability": final_state.hiring_probability,
                "cumulative_score": final_state.cumulative_score,
                "previous_rounds_count": len(final_state.previous_rounds)
            }
            
        except Exception as e:
            logger.error(f"❌ Get context error: {e}")
            return {
                "success": False,
                "error": str(e),
                "briefing": "Unable to load previous context. Standard interview recommended."
            }
    
    async def record_round_completion(
        self,
        candidate_id: str,
        job_id: str,
        round_type: str,
        round_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Record completion of an interview round"""
        try:
            context_key = self._generate_context_key(candidate_id, job_id)
            
            # Load existing context
            existing_context = {}
            if self.redis_client:
                stored = await self.redis_client.get(context_key)
                if stored:
                    existing_context = json.loads(stored)
            elif hasattr(self, '_fallback_store') and context_key in self._fallback_store:
                existing_context = self._fallback_store[context_key]
            
            # Add new round data
            rounds = existing_context.get("rounds", [])
            round_entry = {
                "round_type": round_type,
                "timestamp": datetime.now().isoformat(),
                "decision": round_data.get("decision", "proceed"),
                "skills_assessed": round_data.get("skills_assessed", {}),
                "key_statements": round_data.get("key_statements", []),
                "highlights": round_data.get("highlights", []),
                "concerns": round_data.get("concerns", []),
                "interviewer_notes": round_data.get("notes", "")
            }
            rounds.append(round_entry)
            
            # Update context
            existing_context["rounds"] = rounds
            existing_context["last_updated"] = datetime.now().isoformat()
            
            # Merge topics covered
            topics = existing_context.get("topics_covered", [])
            topics.extend(round_data.get("topics_covered", []))
            existing_context["topics_covered"] = list(set(topics))
            
            # Merge red/green flags
            red_flags = existing_context.get("red_flags", [])
            red_flags.extend(round_data.get("red_flags", []))
            existing_context["red_flags"] = red_flags
            
            green_flags = existing_context.get("green_flags", [])
            green_flags.extend(round_data.get("green_flags", []))
            existing_context["green_flags"] = green_flags
            
            # Save updated context
            if self.redis_client:
                await self.redis_client.set(
                    context_key,
                    json.dumps(existing_context),
                    ex=60 * 60 * 24 * 30
                )
            elif hasattr(self, '_fallback_store'):
                self._fallback_store[context_key] = existing_context
            
            return {
                "success": True,
                "rounds_completed": len(rounds),
                "next_suggested_rounds": self.round_sequence.get(round_type, [])
            }
            
        except Exception as e:
            logger.error(f"❌ Round recording error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_candidate_journey(
        self,
        candidate_id: str,
        job_id: str
    ) -> Dict[str, Any]:
        """Get complete interview journey for a candidate"""
        try:
            context_key = self._generate_context_key(candidate_id, job_id)
            
            if self.redis_client:
                stored = await self.redis_client.get(context_key)
                if stored:
                    context = json.loads(stored)
                else:
                    return {"success": True, "journey": [], "message": "No interview history found"}
            elif hasattr(self, '_fallback_store') and context_key in self._fallback_store:
                context = self._fallback_store[context_key]
            else:
                return {"success": True, "journey": [], "message": "No interview history found"}
            
            # Build journey visualization
            journey = []
            for i, round_data in enumerate(context.get("rounds", [])):
                journey.append({
                    "step": i + 1,
                    "round": round_data.get("round_type"),
                    "date": round_data.get("timestamp"),
                    "decision": round_data.get("decision"),
                    "highlights": round_data.get("highlights", [])[:3],
                    "concerns": round_data.get("concerns", [])[:2]
                })
            
            return {
                "success": True,
                "candidate_id": candidate_id,
                "job_id": job_id,
                "journey": journey,
                "total_rounds": len(journey),
                "hiring_probability": context.get("hiring_probability", 0.5),
                "cumulative_score": context.get("cumulative_score", 50),
                "verified_skills": context.get("verified_skills", {}),
                "status": self._determine_journey_status(journey)
            }
            
        except Exception as e:
            logger.error(f"❌ Journey retrieval error: {e}")
            return {"success": False, "error": str(e)}
    
    def _determine_journey_status(self, journey: List[Dict]) -> str:
        """Determine overall journey status"""
        if not journey:
            return "not_started"
        
        last_decision = journey[-1].get("decision", "proceed")
        if last_decision in ["strong_no", "no"]:
            return "rejected"
        elif last_decision == "strong_yes" and journey[-1].get("round") == "final":
            return "offer_pending"
        else:
            return "in_progress"


# Singleton instance
_cross_session_manager = None

def get_cross_session_manager() -> CrossSessionContextManager:
    """Get singleton cross-session manager instance"""
    global _cross_session_manager
    if _cross_session_manager is None:
        _cross_session_manager = CrossSessionContextManager()
    return _cross_session_manager
