"""
👥 PANEL INTERVIEW MODE - Multi-Persona Interview Agent
Simulates panel interviews with multiple AI personas interviewing simultaneously,
each with distinct perspectives, expertise areas, and questioning styles.

Features:
- Multiple interviewer personas
- Coordinated questioning
- Perspective triangulation
- Cross-examiner collaboration
- Role-specific deep dives
- Consensus building
- Natural turn-taking
- Diverse evaluation viewpoints

Tech Stack:
- LangGraph for multi-agent coordination
- LangChain for persona simulation
- Redis for state synchronization
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
import logging

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterviewerRole(str, Enum):
    """Panel interviewer roles"""
    TECHNICAL_LEAD = "technical_lead"
    HIRING_MANAGER = "hiring_manager"
    TEAM_MEMBER = "team_member"
    HR_REPRESENTATIVE = "hr_representative"
    SENIOR_ENGINEER = "senior_engineer"
    PRODUCT_MANAGER = "product_manager"
    CTO = "cto"
    CULTURE_ADVOCATE = "culture_advocate"


class QuestionCategory(str, Enum):
    """Question categories by role"""
    TECHNICAL_DEPTH = "technical_depth"
    SYSTEM_DESIGN = "system_design"
    TEAM_FIT = "team_fit"
    LEADERSHIP = "leadership"
    PROBLEM_SOLVING = "problem_solving"
    CULTURE = "culture"
    GROWTH = "growth"
    COMMUNICATION = "communication"


class PanelInterviewState(BaseModel):
    """State for panel interview coordination"""
    # Session info
    session_id: str = ""
    candidate_id: str = ""
    job_id: str = ""
    
    # Candidate context
    candidate_profile: Dict[str, Any] = Field(default_factory=dict)
    job_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Panel configuration
    panel_members: List[Dict[str, Any]] = Field(default_factory=list)
    current_speaker: str = ""
    speaking_order: List[str] = Field(default_factory=list)
    
    # Conversation state
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_question: str = ""
    current_answer: str = ""
    
    # Topic tracking
    topics_by_interviewer: Dict[str, List[str]] = Field(default_factory=dict)
    covered_topics: List[str] = Field(default_factory=list)
    pending_topics: List[str] = Field(default_factory=list)
    
    # Evaluation tracking
    evaluations_by_interviewer: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    follow_up_requests: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Coordination signals
    handoff_reason: str = ""
    should_follow_up: bool = False
    follow_up_from: str = ""
    
    # Final outputs
    next_question: str = ""
    next_speaker: str = ""
    speaker_introduction: str = ""
    panel_consensus: Dict[str, Any] = Field(default_factory=dict)


class InterviewerPersona:
    """Individual interviewer persona with distinct characteristics"""
    
    def __init__(
        self,
        role: InterviewerRole,
        name: str,
        expertise: List[str],
        style: str,
        focus_areas: List[str]
    ):
        self.role = role
        self.name = name
        self.expertise = expertise
        self.style = style
        self.focus_areas = focus_areas
        
        # Persona-specific prompts
        self.persona_prompts = self._build_persona_prompts()
    
    def _build_persona_prompts(self) -> Dict[str, str]:
        """Build persona-specific system prompts"""
        base_personas = {
            InterviewerRole.TECHNICAL_LEAD: f"""
You are {self.name}, a Technical Lead with deep expertise in {', '.join(self.expertise)}.
Your interview style is {self.style}.
You focus on: {', '.join(self.focus_areas)}.

In panel interviews, you:
- Ask deep technical questions that reveal true understanding
- Follow up on vague answers with specific probes
- Look for problem-solving approach, not just correct answers
- Evaluate architecture and design thinking
""",
            InterviewerRole.HIRING_MANAGER: f"""
You are {self.name}, a Hiring Manager responsible for team growth and culture.
Your interview style is {self.style}.
You focus on: {', '.join(self.focus_areas)}.

In panel interviews, you:
- Assess team fit and collaboration style
- Explore career goals and growth trajectory
- Look for leadership potential and initiative
- Consider the candidate's impact on team dynamics
""",
            InterviewerRole.TEAM_MEMBER: f"""
You are {self.name}, a Team Member who would work directly with the candidate.
Your interview style is {self.style}.
You focus on: {', '.join(self.focus_areas)}.

In panel interviews, you:
- Ask practical, day-to-day work questions
- Share team culture and gauge interest
- Look for collaboration and communication skills
- Assess technical competence for daily tasks
""",
            InterviewerRole.SENIOR_ENGINEER: f"""
You are {self.name}, a Senior Engineer with extensive hands-on experience.
Your interview style is {self.style}.
You focus on: {', '.join(self.focus_areas)}.

In panel interviews, you:
- Dive deep into implementation details
- Ask about debugging and troubleshooting approaches
- Explore edge cases and production experience
- Look for engineering maturity and best practices
""",
            InterviewerRole.PRODUCT_MANAGER: f"""
You are {self.name}, a Product Manager who collaborates closely with engineering.
Your interview style is {self.style}.
You focus on: {', '.join(self.focus_areas)}.

In panel interviews, you:
- Assess product thinking and user empathy
- Explore communication with non-technical stakeholders
- Look for understanding of business context
- Evaluate prioritization and tradeoff thinking
""",
            InterviewerRole.HR_REPRESENTATIVE: f"""
You are {self.name}, an HR Representative ensuring a fair and comprehensive interview.
Your interview style is {self.style}.
You focus on: {', '.join(self.focus_areas)}.

In panel interviews, you:
- Ensure the candidate has opportunity to showcase strengths
- Ask about values alignment and work preferences
- Explore motivation and career aspirations
- Maintain a welcoming atmosphere
""",
            InterviewerRole.CTO: f"""
You are {self.name}, the CTO evaluating senior technical talent.
Your interview style is {self.style}.
You focus on: {', '.join(self.focus_areas)}.

In panel interviews, you:
- Assess strategic technical thinking
- Explore experience with scale and complexity
- Look for technical leadership indicators
- Evaluate fit with company technical vision
""",
            InterviewerRole.CULTURE_ADVOCATE: f"""
You are {self.name}, a Culture Advocate ensuring values alignment.
Your interview style is {self.style}.
You focus on: {', '.join(self.focus_areas)}.

In panel interviews, you:
- Explore alignment with company values
- Ask about handling conflict and feedback
- Look for diversity of thought and inclusion
- Assess authentic cultural fit
"""
        }
        
        return {
            "system": base_personas.get(self.role, base_personas[InterviewerRole.TEAM_MEMBER])
        }


class PanelInterviewMode:
    """
    Multi-persona panel interview orchestrator.
    Coordinates multiple AI interviewers for comprehensive evaluation.
    """
    
    def __init__(self):
        logger.info("👥 Initializing Panel Interview Mode...")
        
        # LLM for all personas
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.7  # Higher temperature for persona variety
        )
        
        # Available personas
        self.persona_templates = self._create_persona_templates()
        
        # Build coordination workflow
        self.workflow = self._build_panel_workflow()
        
        # Active sessions
        self.sessions: Dict[str, PanelInterviewState] = {}
        
        logger.info("✅ Panel Interview Mode initialized")
    
    def _create_persona_templates(self) -> Dict[str, InterviewerPersona]:
        """Create templates for different interviewer personas"""
        return {
            "alex_tech_lead": InterviewerPersona(
                role=InterviewerRole.TECHNICAL_LEAD,
                name="Alex",
                expertise=["system design", "distributed systems", "cloud architecture"],
                style="thorough and curious",
                focus_areas=["scalability", "code quality", "technical depth"]
            ),
            "sarah_hiring_mgr": InterviewerPersona(
                role=InterviewerRole.HIRING_MANAGER,
                name="Sarah",
                expertise=["team building", "engineering management", "career development"],
                style="warm but direct",
                focus_areas=["leadership", "growth mindset", "team dynamics"]
            ),
            "mike_senior_eng": InterviewerPersona(
                role=InterviewerRole.SENIOR_ENGINEER,
                name="Mike",
                expertise=["backend development", "databases", "performance optimization"],
                style="practical and detail-oriented",
                focus_areas=["implementation", "debugging", "best practices"]
            ),
            "lisa_product_mgr": InterviewerPersona(
                role=InterviewerRole.PRODUCT_MANAGER,
                name="Lisa",
                expertise=["product strategy", "user research", "agile methodologies"],
                style="collaborative and thoughtful",
                focus_areas=["product thinking", "user focus", "communication"]
            ),
            "jordan_hr": InterviewerPersona(
                role=InterviewerRole.HR_REPRESENTATIVE,
                name="Jordan",
                expertise=["talent acquisition", "employee experience", "culture"],
                style="supportive and structured",
                focus_areas=["values", "work style", "career goals"]
            ),
            "david_cto": InterviewerPersona(
                role=InterviewerRole.CTO,
                name="David",
                expertise=["technical strategy", "architecture", "engineering excellence"],
                style="strategic and insightful",
                focus_areas=["vision", "scale", "technical leadership"]
            )
        }
    
    def _build_panel_workflow(self) -> StateGraph:
        """Build LangGraph workflow for panel coordination"""
        workflow = StateGraph(PanelInterviewState)
        
        # Define nodes
        workflow.add_node("analyze_answer", self._analyze_answer)
        workflow.add_node("collect_evaluations", self._collect_evaluations)
        workflow.add_node("determine_next_speaker", self._determine_next_speaker)
        workflow.add_node("generate_question", self._generate_question)
        workflow.add_node("prepare_handoff", self._prepare_handoff)
        
        # Define edges
        workflow.set_entry_point("analyze_answer")
        workflow.add_edge("analyze_answer", "collect_evaluations")
        workflow.add_edge("collect_evaluations", "determine_next_speaker")
        workflow.add_edge("determine_next_speaker", "generate_question")
        workflow.add_edge("generate_question", "prepare_handoff")
        workflow.add_edge("prepare_handoff", END)
        
        return workflow.compile()
    
    async def _analyze_answer(self, state: PanelInterviewState) -> PanelInterviewState:
        """Analyze the candidate's answer from current speaker's perspective"""
        try:
            if not state.current_answer:
                return state
            
            current_persona = None
            for member in state.panel_members:
                if member["persona_id"] == state.current_speaker:
                    current_persona = self.persona_templates.get(member["persona_id"])
                    break
            
            if not current_persona:
                return state
            
            prompt = f"""
            {current_persona.persona_prompts["system"]}
            
            You just asked: "{state.current_question}"
            
            Candidate answered: "{state.current_answer}"
            
            Analyze this answer from your perspective as {current_persona.name}.
            
            Consider:
            1. Did they address your question fully?
            2. What stood out (positive or negative)?
            3. Do you need to follow up?
            4. Should another panel member explore something?
            
            Return JSON:
            {{
                "answer_quality": 0.8,
                "key_observations": ["observation1"],
                "strengths_shown": ["strength1"],
                "concerns": ["concern1"],
                "needs_follow_up": true|false,
                "follow_up_topic": "topic if needed",
                "suggest_handoff_to": "role to explore further (or null)",
                "handoff_reason": "why handoff"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            # Store in conversation history
            state.conversation_history.append({
                "speaker": state.current_speaker,
                "question": state.current_question,
                "answer": state.current_answer,
                "analysis": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if follow-up needed
            if result.get("needs_follow_up"):
                state.should_follow_up = True
                state.follow_up_from = state.current_speaker
            
            # Handle handoff suggestion
            if result.get("suggest_handoff_to"):
                state.follow_up_requests.append({
                    "from": state.current_speaker,
                    "to_role": result.get("suggest_handoff_to"),
                    "topic": result.get("handoff_reason"),
                    "priority": "normal"
                })
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Answer analysis error: {e}")
            return state
    
    async def _collect_evaluations(self, state: PanelInterviewState) -> PanelInterviewState:
        """Collect evaluations from all panel members"""
        try:
            # Get silent observations from other panel members
            for member in state.panel_members:
                if member["persona_id"] == state.current_speaker:
                    continue  # Skip current speaker
                
                persona = self.persona_templates.get(member["persona_id"])
                if not persona:
                    continue
                
                prompt = f"""
                {persona.persona_prompts["system"]}
                
                You are observing a panel interview. The candidate was asked:
                "{state.current_question}"
                
                And answered:
                "{state.current_answer}"
                
                As {persona.name} (observing, not asking), note:
                1. Anything relevant to your focus areas?
                2. Any follow-up you'd like to ask?
                
                Return JSON (brief):
                {{
                    "relevance_to_me": 0.6,
                    "note": "Brief observation",
                    "want_to_follow_up": true|false,
                    "follow_up_question": "question if wanted"
                }}
                """
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                result = json.loads(response.content.replace("```json", "").replace("```", ""))
                
                # Store evaluation
                if member["persona_id"] not in state.evaluations_by_interviewer:
                    state.evaluations_by_interviewer[member["persona_id"]] = {
                        "observations": [],
                        "overall_score": 0.0
                    }
                
                state.evaluations_by_interviewer[member["persona_id"]]["observations"].append({
                    "turn": len(state.conversation_history),
                    "note": result.get("note", ""),
                    "relevance": result.get("relevance_to_me", 0.5)
                })
                
                if result.get("want_to_follow_up"):
                    state.follow_up_requests.append({
                        "from": member["persona_id"],
                        "question": result.get("follow_up_question"),
                        "priority": "normal"
                    })
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Evaluation collection error: {e}")
            return state
    
    async def _determine_next_speaker(self, state: PanelInterviewState) -> PanelInterviewState:
        """Determine who should ask the next question"""
        try:
            # Priority 1: If current speaker needs to follow up
            if state.should_follow_up and state.follow_up_from:
                state.next_speaker = state.follow_up_from
                state.handoff_reason = "follow_up"
                return state
            
            # Priority 2: If there are handoff requests
            if state.follow_up_requests:
                # Sort by priority, take first
                request = state.follow_up_requests.pop(0)
                state.next_speaker = request.get("from", state.speaking_order[0])
                state.handoff_reason = "requested_by_panel"
                return state
            
            # Priority 3: Rotate through panel
            current_idx = -1
            for i, member in enumerate(state.panel_members):
                if member["persona_id"] == state.current_speaker:
                    current_idx = i
                    break
            
            next_idx = (current_idx + 1) % len(state.panel_members)
            state.next_speaker = state.panel_members[next_idx]["persona_id"]
            state.handoff_reason = "rotation"
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Next speaker determination error: {e}")
            state.next_speaker = state.panel_members[0]["persona_id"]
            return state
    
    async def _generate_question(self, state: PanelInterviewState) -> PanelInterviewState:
        """Generate the next question from the determined speaker"""
        try:
            next_persona = self.persona_templates.get(state.next_speaker)
            if not next_persona:
                state.next_question = "Can you tell me more about your experience?"
                return state
            
            # Build context from conversation
            recent_exchanges = state.conversation_history[-3:] if state.conversation_history else []
            
            # Check for pending follow-up from this speaker
            pending_followup = None
            for req in state.follow_up_requests:
                if req.get("from") == state.next_speaker:
                    pending_followup = req
                    break
            
            prompt = f"""
            {next_persona.persona_prompts["system"]}
            
            CANDIDATE PROFILE:
            {json.dumps(state.candidate_profile, indent=2)}
            
            JOB REQUIREMENTS:
            {json.dumps(state.job_requirements, indent=2)}
            
            RECENT CONVERSATION:
            {json.dumps(recent_exchanges, indent=2)}
            
            TOPICS ALREADY COVERED:
            {state.covered_topics}
            
            YOUR PENDING TOPICS:
            {state.topics_by_interviewer.get(state.next_speaker, [])}
            
            {f"SPECIFIC FOLLOW-UP REQUESTED: {pending_followup['question']}" if pending_followup else ""}
            
            Generate your next question as {next_persona.name}.
            
            Guidelines:
            1. Don't repeat covered topics unless following up
            2. Build on previous answers naturally
            3. Stay true to your persona and focus areas
            4. Make it conversational, not interrogation-style
            
            Return JSON:
            {{
                "question": "Your question",
                "topic": "Main topic this covers",
                "intent": "What you're trying to learn",
                "natural_lead_in": "A brief sentence to connect to previous discussion"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            # Build the full question with lead-in
            lead_in = result.get("natural_lead_in", "")
            question = result.get("question", "Tell me more about yourself.")
            
            if lead_in:
                state.next_question = f"{lead_in} {question}"
            else:
                state.next_question = question
            
            # Track topic
            topic = result.get("topic", "general")
            if topic not in state.covered_topics:
                state.covered_topics.append(topic)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Question generation error: {e}")
            state.next_question = "Can you elaborate on that?"
            return state
    
    async def _prepare_handoff(self, state: PanelInterviewState) -> PanelInterviewState:
        """Prepare the speaker introduction and handoff"""
        try:
            next_persona = self.persona_templates.get(state.next_speaker)
            if not next_persona:
                return state
            
            # Generate natural handoff
            if state.handoff_reason == "follow_up":
                state.speaker_introduction = ""  # Same speaker, no intro needed
            elif state.handoff_reason == "requested_by_panel":
                state.speaker_introduction = f"Thanks. I'd like to hand off to {next_persona.name} who had a follow-up."
            else:
                # Natural rotation
                intros = [
                    f"Let me pass it over to {next_persona.name}.",
                    f"{next_persona.name}, would you like to jump in?",
                    f"Thanks for that. {next_persona.name}?",
                    f"I'll let {next_persona.name} take the next question."
                ]
                state.speaker_introduction = random.choice(intros)
            
            # Reset follow-up flag
            state.should_follow_up = False
            state.follow_up_from = ""
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Handoff preparation error: {e}")
            return state
    
    async def create_panel(
        self,
        session_id: str,
        candidate_id: str,
        job_id: str,
        panel_config: List[str],  # List of persona IDs
        candidate_profile: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a panel interview session"""
        try:
            # Validate personas
            panel_members = []
            for persona_id in panel_config:
                if persona_id in self.persona_templates:
                    persona = self.persona_templates[persona_id]
                    panel_members.append({
                        "persona_id": persona_id,
                        "name": persona.name,
                        "role": persona.role.value
                    })
            
            if not panel_members:
                return {"success": False, "error": "No valid personas provided"}
            
            # Assign initial topics to each interviewer
            topics_by_interviewer = {}
            for member in panel_members:
                persona = self.persona_templates[member["persona_id"]]
                topics_by_interviewer[member["persona_id"]] = list(persona.focus_areas)
            
            state = PanelInterviewState(
                session_id=session_id,
                candidate_id=candidate_id,
                job_id=job_id,
                candidate_profile=candidate_profile,
                job_requirements=job_requirements,
                panel_members=panel_members,
                current_speaker=panel_members[0]["persona_id"],
                speaking_order=[m["persona_id"] for m in panel_members],
                topics_by_interviewer=topics_by_interviewer
            )
            
            self.sessions[session_id] = state
            
            # Generate first question
            first_persona = self.persona_templates[panel_members[0]["persona_id"]]
            
            return {
                "success": True,
                "session_id": session_id,
                "panel": panel_members,
                "first_speaker": {
                    "persona_id": panel_members[0]["persona_id"],
                    "name": first_persona.name,
                    "role": first_persona.role.value
                },
                "panel_introduction": self._generate_panel_introduction(panel_members)
            }
            
        except Exception as e:
            logger.error(f"❌ Panel creation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_panel_introduction(self, panel_members: List[Dict]) -> str:
        """Generate introduction for the panel"""
        intros = []
        for member in panel_members:
            persona = self.persona_templates.get(member["persona_id"])
            if persona:
                role_desc = {
                    InterviewerRole.TECHNICAL_LEAD: "our Technical Lead",
                    InterviewerRole.HIRING_MANAGER: "the Hiring Manager",
                    InterviewerRole.TEAM_MEMBER: "a member of the team you'd be joining",
                    InterviewerRole.SENIOR_ENGINEER: "a Senior Engineer",
                    InterviewerRole.PRODUCT_MANAGER: "our Product Manager",
                    InterviewerRole.HR_REPRESENTATIVE: "from our People team",
                    InterviewerRole.CTO: "our CTO",
                    InterviewerRole.CULTURE_ADVOCATE: "our Culture Advocate"
                }
                intros.append(f"{persona.name}, {role_desc.get(persona.role, 'on the team')}")
        
        return f"Welcome to your panel interview! Today you'll be meeting with {', '.join(intros[:-1])}, and {intros[-1]}. Each of us will have some questions for you, and we encourage you to ask us questions too. Let's get started!"
    
    async def get_first_question(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get the first question from the panel"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            first_persona = self.persona_templates[state.current_speaker]
            
            prompt = f"""
            {first_persona.persona_prompts["system"]}
            
            You are starting a panel interview. This is the first question.
            
            CANDIDATE PROFILE:
            {json.dumps(state.candidate_profile, indent=2)}
            
            JOB: {json.dumps(state.job_requirements, indent=2)}
            
            Generate a warm, engaging opening question that:
            1. Welcomes the candidate
            2. Relates to your focus areas
            3. Is easy to answer to build confidence
            4. Sets a positive tone
            
            Return JSON:
            {{
                "opening": "Brief introduction of yourself",
                "question": "Your opening question",
                "topic": "Main topic"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            full_question = f"Hi, I'm {first_persona.name}. {result.get('opening', '')} {result.get('question', 'Tell me about yourself.')}"
            
            state.current_question = full_question
            self.sessions[session_id] = state
            
            return {
                "success": True,
                "speaker": {
                    "persona_id": state.current_speaker,
                    "name": first_persona.name,
                    "role": first_persona.role.value
                },
                "question": full_question,
                "topic": result.get("topic", "introduction")
            }
            
        except Exception as e:
            logger.error(f"❌ First question error: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_answer_and_get_next(
        self,
        session_id: str,
        answer: str
    ) -> Dict[str, Any]:
        """Process answer and get next question"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            state.current_answer = answer
            
            # Run coordination workflow
            final_state = await self.workflow.ainvoke(state)
            
            # Update current speaker
            final_state.current_speaker = final_state.next_speaker
            final_state.current_question = final_state.next_question
            
            self.sessions[session_id] = final_state
            
            next_persona = self.persona_templates.get(final_state.next_speaker)
            
            return {
                "success": True,
                "speaker_introduction": final_state.speaker_introduction,
                "speaker": {
                    "persona_id": final_state.next_speaker,
                    "name": next_persona.name if next_persona else "Interviewer",
                    "role": next_persona.role.value if next_persona else "interviewer"
                },
                "question": final_state.next_question,
                "handoff_reason": final_state.handoff_reason,
                "topics_covered": len(final_state.covered_topics)
            }
            
        except Exception as e:
            logger.error(f"❌ Answer processing error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_panel_consensus(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get final consensus from all panel members"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            
            # Collect individual evaluations
            evaluations = []
            
            for member in state.panel_members:
                persona = self.persona_templates.get(member["persona_id"])
                if not persona:
                    continue
                
                prompt = f"""
                {persona.persona_prompts["system"]}
                
                The panel interview has concluded.
                
                CONVERSATION SUMMARY:
                {json.dumps(state.conversation_history[-5:], indent=2)}
                
                YOUR OBSERVATIONS:
                {json.dumps(state.evaluations_by_interviewer.get(member["persona_id"], {}), indent=2)}
                
                Provide your final evaluation as {persona.name}.
                
                Return JSON:
                {{
                    "overall_score": 0.75,
                    "recommendation": "strong_yes|yes|maybe|no|strong_no",
                    "key_strengths": ["strength1"],
                    "concerns": ["concern1"],
                    "brief_summary": "2-3 sentence summary"
                }}
                """
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                result = json.loads(response.content.replace("```json", "").replace("```", ""))
                
                evaluations.append({
                    "evaluator": persona.name,
                    "role": persona.role.value,
                    **result
                })
            
            # Build consensus
            avg_score = sum(e.get("overall_score", 0.5) for e in evaluations) / len(evaluations)
            
            # Count recommendations
            rec_counts = {}
            for e in evaluations:
                rec = e.get("recommendation", "maybe")
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
            
            consensus_rec = max(rec_counts.keys(), key=lambda k: rec_counts[k])
            
            return {
                "success": True,
                "individual_evaluations": evaluations,
                "consensus": {
                    "average_score": avg_score,
                    "recommendation": consensus_rec,
                    "recommendation_distribution": rec_counts,
                    "unanimous": len(set(e.get("recommendation") for e in evaluations)) == 1
                },
                "interview_summary": {
                    "total_questions": len(state.conversation_history),
                    "topics_covered": state.covered_topics,
                    "panel_size": len(state.panel_members)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Consensus generation error: {e}")
            return {"success": False, "error": str(e)}


# Singleton instance
_panel_interview = None

def get_panel_interview() -> PanelInterviewMode:
    """Get singleton panel interview instance"""
    global _panel_interview
    if _panel_interview is None:
        _panel_interview = PanelInterviewMode()
    return _panel_interview
