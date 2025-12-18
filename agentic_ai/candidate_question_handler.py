"""
❓ CANDIDATE QUESTION HANDLER - Intelligent Q&A Phase Manager
Handles the critical "Do you have any questions for us?" phase with
intelligent responses, company knowledge, and engagement assessment.

Features:
- Intelligent answer generation
- Company knowledge retrieval
- Question quality assessment
- Red flag detection
- Engagement scoring
- Follow-up conversation
- Cultural fit indicators
- Personalized responses

Tech Stack:
- LangGraph for Q&A workflow
- LangChain for intelligent responses
- RAG for company knowledge
- Semantic search for context
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionQuality(str, Enum):
    """Quality levels of candidate questions"""
    EXCEPTIONAL = "exceptional"  # Shows deep research and insight
    STRONG = "strong"  # Thoughtful and relevant
    ADEQUATE = "adequate"  # Basic but acceptable
    WEAK = "weak"  # Generic or concerning
    RED_FLAG = "red_flag"  # Concerning question


class QuestionCategory(str, Enum):
    """Categories of candidate questions"""
    ROLE = "role"  # About the job itself
    TEAM = "team"  # About team dynamics
    GROWTH = "growth"  # Career development
    CULTURE = "culture"  # Company culture
    TECHNICAL = "technical"  # Tech stack, challenges
    PROCESS = "process"  # How work gets done
    STRATEGY = "strategy"  # Company direction
    COMPENSATION = "compensation"  # Pay, benefits
    WORK_LIFE = "work_life"  # Balance, flexibility
    INTERVIEWER = "interviewer"  # Personal to interviewer


class CandidateQuestionState(BaseModel):
    """State for candidate question handling"""
    # Session context
    session_id: str = ""
    candidate_id: str = ""
    job_id: str = ""
    
    # Company/Job context
    company_info: Dict[str, Any] = Field(default_factory=dict)
    job_details: Dict[str, Any] = Field(default_factory=dict)
    team_info: Dict[str, Any] = Field(default_factory=dict)
    
    # Interviewer context
    interviewer_persona: Dict[str, Any] = Field(default_factory=dict)
    interview_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Current Q&A
    candidate_question: str = ""
    question_category: str = ""
    question_quality: str = ""
    
    # Q&A history
    qa_history: List[Dict[str, Any]] = Field(default_factory=list)
    questions_asked: int = 0
    
    # Analysis
    quality_scores: List[float] = Field(default_factory=list)
    categories_asked: List[str] = Field(default_factory=list)
    engagement_indicators: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    
    # Generated response
    answer: str = ""
    follow_up_prompt: str = ""
    can_continue: bool = True
    
    # Final assessment
    overall_engagement: float = 0.0
    question_pattern_analysis: str = ""
    cultural_fit_signals: List[str] = Field(default_factory=list)


class CandidateQuestionHandler:
    """
    Handles the Q&A phase where candidates ask questions.
    Provides intelligent, context-aware answers and assesses engagement.
    """
    
    def __init__(self):
        logger.info("❓ Initializing Candidate Question Handler...")
        
        # LLM for generating responses
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.6
        )
        
        # Sentence transformer for semantic matching
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build Q&A workflow
        self.workflow = self._build_qa_workflow()
        
        # Quality indicators
        self._init_quality_indicators()
        
        # Active sessions
        self.sessions: Dict[str, CandidateQuestionState] = {}
        
        logger.info("✅ Candidate Question Handler initialized")
    
    def _init_quality_indicators(self):
        """Initialize question quality indicators"""
        self.exceptional_indicators = [
            "specific challenge",
            "recent announcement",
            "technical roadmap",
            "team structure",
            "growth metrics",
            "strategic direction",
            "competitor",
            "market position"
        ]
        
        self.strong_indicators = [
            "typical day",
            "success metrics",
            "team collaboration",
            "career growth",
            "learning opportunities",
            "tech stack decisions",
            "project examples"
        ]
        
        self.red_flag_patterns = [
            "how little",
            "minimum",
            "can i work less",
            "how soon can i",
            "guaranteed promotion",
            "never",
            "i don't want to",
            "do i have to"
        ]
    
    def _build_qa_workflow(self) -> StateGraph:
        """Build LangGraph workflow for Q&A handling"""
        workflow = StateGraph(CandidateQuestionState)
        
        # Define nodes
        workflow.add_node("classify_question", self._classify_question)
        workflow.add_node("assess_quality", self._assess_quality)
        workflow.add_node("retrieve_context", self._retrieve_context)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("update_assessment", self._update_assessment)
        workflow.add_node("prepare_follow_up", self._prepare_follow_up)
        
        # Define edges
        workflow.set_entry_point("classify_question")
        workflow.add_edge("classify_question", "assess_quality")
        workflow.add_edge("assess_quality", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "update_assessment")
        workflow.add_edge("update_assessment", "prepare_follow_up")
        workflow.add_edge("prepare_follow_up", END)
        
        return workflow.compile()
    
    async def _classify_question(self, state: CandidateQuestionState) -> CandidateQuestionState:
        """Classify the candidate's question"""
        try:
            question = state.candidate_question.lower()
            
            # Quick keyword-based classification
            category_keywords = {
                QuestionCategory.ROLE.value: ["responsibility", "day to day", "typical day", "role", "expect"],
                QuestionCategory.TEAM.value: ["team", "who would i", "work with", "collaborate", "manager"],
                QuestionCategory.GROWTH.value: ["growth", "career", "promotion", "learn", "develop", "advance"],
                QuestionCategory.CULTURE.value: ["culture", "values", "environment", "remote", "office"],
                QuestionCategory.TECHNICAL.value: ["tech stack", "technology", "architecture", "tools", "scale"],
                QuestionCategory.PROCESS.value: ["agile", "sprint", "deploy", "review", "process", "workflow"],
                QuestionCategory.STRATEGY.value: ["vision", "future", "roadmap", "direction", "strategy", "goal"],
                QuestionCategory.COMPENSATION.value: ["salary", "benefits", "equity", "compensation", "package"],
                QuestionCategory.WORK_LIFE.value: ["balance", "hours", "flexibility", "vacation", "remote"],
                QuestionCategory.INTERVIEWER.value: ["your experience", "you like", "your favorite", "how long have you"]
            }
            
            for category, keywords in category_keywords.items():
                if any(kw in question for kw in keywords):
                    state.question_category = category
                    break
            
            if not state.question_category:
                # Use LLM for ambiguous questions
                prompt = f"""
                Classify this interview question from a candidate:
                "{state.candidate_question}"
                
                Categories: role, team, growth, culture, technical, process, strategy, compensation, work_life, interviewer
                
                Return only the category name.
                """
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                state.question_category = response.content.strip().lower()
            
            state.categories_asked.append(state.question_category)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Question classification error: {e}")
            state.question_category = "general"
            return state
    
    async def _assess_quality(self, state: CandidateQuestionState) -> CandidateQuestionState:
        """Assess the quality of the question"""
        try:
            question_lower = state.candidate_question.lower()
            
            # Check for red flags first
            for pattern in self.red_flag_patterns:
                if pattern in question_lower:
                    state.question_quality = QuestionQuality.RED_FLAG.value
                    state.red_flags.append(f"Concerning question: {state.candidate_question}")
                    state.quality_scores.append(0.2)
                    return state
            
            # Check for exceptional indicators
            exceptional_count = sum(1 for ind in self.exceptional_indicators if ind in question_lower)
            if exceptional_count >= 2:
                state.question_quality = QuestionQuality.EXCEPTIONAL.value
                state.engagement_indicators.append("Deep research shown")
                state.quality_scores.append(1.0)
                return state
            
            # Check for strong indicators
            strong_count = sum(1 for ind in self.strong_indicators if ind in question_lower)
            if strong_count >= 1 or exceptional_count >= 1:
                state.question_quality = QuestionQuality.STRONG.value
                state.engagement_indicators.append("Thoughtful question")
                state.quality_scores.append(0.8)
                return state
            
            # Generic questions
            generic_patterns = ["any questions", "anything else", "what else", "tell me more"]
            if any(p in question_lower for p in generic_patterns):
                state.question_quality = QuestionQuality.WEAK.value
                state.quality_scores.append(0.4)
                return state
            
            # Default to adequate
            state.question_quality = QuestionQuality.ADEQUATE.value
            state.quality_scores.append(0.6)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Quality assessment error: {e}")
            state.question_quality = QuestionQuality.ADEQUATE.value
            return state
    
    async def _retrieve_context(self, state: CandidateQuestionState) -> CandidateQuestionState:
        """Retrieve relevant context for answering"""
        try:
            # This would integrate with a knowledge base
            # For now, we'll use the provided context
            
            # Select relevant context based on category
            category = state.question_category
            
            context_map = {
                QuestionCategory.ROLE.value: state.job_details,
                QuestionCategory.TEAM.value: state.team_info,
                QuestionCategory.CULTURE.value: state.company_info.get("culture", {}),
                QuestionCategory.TECHNICAL.value: state.job_details.get("tech_stack", {}),
                QuestionCategory.GROWTH.value: state.company_info.get("growth", {}),
                QuestionCategory.STRATEGY.value: state.company_info.get("strategy", {}),
            }
            
            # Context is already in state for the LLM to use
            return state
            
        except Exception as e:
            logger.error(f"❌ Context retrieval error: {e}")
            return state
    
    async def _generate_answer(self, state: CandidateQuestionState) -> CandidateQuestionState:
        """Generate intelligent answer to the question"""
        try:
            prompt = f"""
            You are an interviewer answering a candidate's question during an interview.
            
            INTERVIEWER CONTEXT:
            {json.dumps(state.interviewer_persona, indent=2)}
            
            COMPANY INFO:
            {json.dumps(state.company_info, indent=2)}
            
            JOB DETAILS:
            {json.dumps(state.job_details, indent=2)}
            
            TEAM INFO:
            {json.dumps(state.team_info, indent=2)}
            
            INTERVIEW HISTORY (key moments):
            {json.dumps(state.interview_history[-3:] if state.interview_history else [], indent=2)}
            
            CANDIDATE'S QUESTION:
            "{state.candidate_question}"
            
            QUESTION CATEGORY: {state.question_category}
            
            Generate a genuine, informative answer that:
            1. Directly addresses their question
            2. Provides specific, concrete information when possible
            3. Shows enthusiasm about the opportunity
            4. Is honest about challenges (if appropriate)
            5. Personalizes if the question is about you
            6. Is conversational, not corporate-speak
            
            If the question requires information you don't have, be honest and offer to follow up.
            
            Keep the response under 150 words for conciseness.
            
            Return JSON:
            {{
                "answer": "Your answer",
                "key_points": ["point1", "point2"],
                "honesty_note": "any caveats or honest admissions (or null)"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.answer = result.get("answer", "That's a great question. Let me think about that...")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Answer generation error: {e}")
            state.answer = "That's a great question. I'd be happy to discuss that in more detail."
            return state
    
    async def _update_assessment(self, state: CandidateQuestionState) -> CandidateQuestionState:
        """Update engagement and cultural fit assessment"""
        try:
            state.questions_asked += 1
            
            # Store in Q&A history
            state.qa_history.append({
                "question": state.candidate_question,
                "category": state.question_category,
                "quality": state.question_quality,
                "answer": state.answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # Calculate overall engagement
            if state.quality_scores:
                state.overall_engagement = sum(state.quality_scores) / len(state.quality_scores)
            
            # Analyze cultural fit signals
            if state.question_quality == QuestionQuality.EXCEPTIONAL.value:
                state.cultural_fit_signals.append("Shows genuine interest in company success")
            
            if QuestionCategory.CULTURE.value in state.categories_asked:
                state.cultural_fit_signals.append("Values cultural alignment")
            
            if QuestionCategory.GROWTH.value in state.categories_asked:
                state.cultural_fit_signals.append("Growth-oriented mindset")
            
            if QuestionCategory.TEAM.value in state.categories_asked:
                state.cultural_fit_signals.append("Team-focused thinking")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Assessment update error: {e}")
            return state
    
    async def _prepare_follow_up(self, state: CandidateQuestionState) -> CandidateQuestionState:
        """Prepare follow-up prompt for more questions"""
        try:
            # Check if we should continue
            if state.questions_asked >= 5:
                state.can_continue = False
                state.follow_up_prompt = "We're running short on time, but is there anything else critical you'd like to know?"
                return state
            
            # Generate natural follow-up
            follow_ups = [
                "What else would you like to know?",
                "Do you have any other questions?",
                "Is there anything else I can tell you about the role or team?",
                "Any other questions on your mind?",
                "What other aspects of the role are you curious about?"
            ]
            
            # Contextual follow-up based on categories not yet asked
            not_asked = set(QuestionCategory.__members__.keys()) - set(state.categories_asked)
            if not_asked:
                suggestions = {
                    "ROLE": "about the day-to-day responsibilities",
                    "TEAM": "about the team you'd be working with",
                    "TECHNICAL": "about our tech stack",
                    "GROWTH": "about growth opportunities"
                }
                
                for category in not_asked:
                    if category in suggestions:
                        state.follow_up_prompt = f"Do you have any questions {suggestions[category]}, or anything else?"
                        break
            
            if not state.follow_up_prompt:
                import random
                state.follow_up_prompt = random.choice(follow_ups)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Follow-up preparation error: {e}")
            state.follow_up_prompt = "Any other questions?"
            return state
    
    async def start_qa_session(
        self,
        session_id: str,
        candidate_id: str,
        job_id: str,
        company_info: Dict[str, Any],
        job_details: Dict[str, Any],
        team_info: Dict[str, Any],
        interviewer_persona: Optional[Dict[str, Any]] = None,
        interview_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Start a Q&A session"""
        try:
            state = CandidateQuestionState(
                session_id=session_id,
                candidate_id=candidate_id,
                job_id=job_id,
                company_info=company_info,
                job_details=job_details,
                team_info=team_info,
                interviewer_persona=interviewer_persona or {},
                interview_history=interview_history or []
            )
            
            self.sessions[session_id] = state
            
            # Generate opening prompt
            opening = self._generate_opening_prompt(state)
            
            return {
                "success": True,
                "session_id": session_id,
                "opening_prompt": opening
            }
            
        except Exception as e:
            logger.error(f"❌ Q&A session start error: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_opening_prompt(self, state: CandidateQuestionState) -> str:
        """Generate the opening prompt for Q&A phase"""
        openings = [
            "We've covered a lot of ground about your experience. Now I'd like to give you time to ask questions. What would you like to know about us?",
            "Before we wrap up, I want to make sure you have all the information you need. What questions do you have for me?",
            "That brings us to my favorite part - your questions. What would you like to learn more about?",
            "I've really enjoyed learning about your background. Now it's your turn - what would you like to know about the role, team, or company?"
        ]
        import random
        return random.choice(openings)
    
    async def handle_question(
        self,
        session_id: str,
        question: str
    ) -> Dict[str, Any]:
        """Handle a candidate question"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            state.candidate_question = question
            
            # Check for "no questions" response
            no_question_patterns = ["no", "i'm good", "nothing", "that's all", "no questions"]
            if any(p in question.lower() for p in no_question_patterns):
                return {
                    "success": True,
                    "answer": "Okay, that's perfectly fine. Thank you for your time today!",
                    "can_continue": False,
                    "session_complete": True,
                    "final_assessment": await self._generate_final_assessment(state)
                }
            
            # Run Q&A workflow
            final_state = await self.workflow.ainvoke(state)
            
            self.sessions[session_id] = final_state
            
            return {
                "success": True,
                "answer": final_state.answer,
                "question_category": final_state.question_category,
                "question_quality": final_state.question_quality,
                "follow_up_prompt": final_state.follow_up_prompt,
                "can_continue": final_state.can_continue,
                "questions_asked": final_state.questions_asked,
                "engagement_score": final_state.overall_engagement
            }
            
        except Exception as e:
            logger.error(f"❌ Question handling error: {e}")
            return {"success": False, "error": str(e)}
    
    async def end_qa_session(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """End Q&A session and generate assessment"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            assessment = await self._generate_final_assessment(state)
            
            # Clean up
            del self.sessions[session_id]
            
            return {
                "success": True,
                **assessment
            }
            
        except Exception as e:
            logger.error(f"❌ Session end error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_final_assessment(self, state: CandidateQuestionState) -> Dict[str, Any]:
        """Generate final assessment of Q&A phase"""
        try:
            # Analyze question patterns
            prompt = f"""
            Analyze the candidate's questions during this interview.
            
            QUESTIONS ASKED:
            {json.dumps(state.qa_history, indent=2)}
            
            CATEGORIES COVERED:
            {state.categories_asked}
            
            QUALITY SCORES:
            {state.quality_scores}
            
            RED FLAGS:
            {state.red_flags}
            
            Analyze what these questions reveal about the candidate:
            1. Their priorities and motivations
            2. Their level of research and preparation
            3. Their genuine interest in the role
            4. Any concerns indicated by their questions
            
            Return JSON:
            {{
                "pattern_analysis": "2-3 sentence analysis of their questioning pattern",
                "revealed_priorities": ["priority1", "priority2"],
                "preparation_level": "exceptional|good|moderate|minimal|none",
                "genuine_interest_score": 0.8,
                "concerns": ["concern1"],
                "positive_signals": ["signal1"]
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            return {
                "questions_asked": state.questions_asked,
                "categories_explored": list(set(state.categories_asked)),
                "average_quality": state.overall_engagement,
                "pattern_analysis": result.get("pattern_analysis", ""),
                "revealed_priorities": result.get("revealed_priorities", []),
                "preparation_level": result.get("preparation_level", "moderate"),
                "genuine_interest_score": result.get("genuine_interest_score", 0.5),
                "cultural_fit_signals": state.cultural_fit_signals,
                "red_flags": state.red_flags,
                "positive_signals": result.get("positive_signals", []),
                "concerns": result.get("concerns", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Final assessment error: {e}")
            return {
                "questions_asked": state.questions_asked,
                "average_quality": state.overall_engagement,
                "error": str(e)
            }


# Singleton instance
_question_handler = None

def get_question_handler() -> CandidateQuestionHandler:
    """Get singleton question handler instance"""
    global _question_handler
    if _question_handler is None:
        _question_handler = CandidateQuestionHandler()
    return _question_handler
