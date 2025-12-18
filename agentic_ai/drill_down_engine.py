"""
🔬 DRILL-DOWN QUESTION ENGINE - Multi-Level Deep Probing System
Implements intelligent follow-up questioning that probes 3-4 levels deep on topics
to verify actual knowledge vs surface-level understanding.

Features:
- Automatic depth detection of answers
- Multi-level probing (3-4 levels deep)
- Socratic questioning methodology
- Edge case exploration
- Contradiction detection
- Skill verification through depth
- Knowledge vs experience differentiation
- Technical accuracy validation

Tech Stack:
- LangGraph for conversation state
- LangChain for intelligent probing
- Semantic analysis for depth detection
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerDepth(str, Enum):
    """Depth level of an answer"""
    SURFACE = "surface"  # Basic, textbook answer
    MODERATE = "moderate"  # Some depth, shows understanding
    DEEP = "deep"  # Detailed, shows experience
    EXPERT = "expert"  # Nuanced, shows mastery


class ProbeType(str, Enum):
    """Types of probing questions"""
    CLARIFICATION = "clarification"  # "What do you mean by...?"
    ELABORATION = "elaboration"  # "Can you tell me more about...?"
    SPECIFICITY = "specificity"  # "Can you give a specific example?"
    JUSTIFICATION = "justification"  # "Why did you choose...?"
    EDGE_CASE = "edge_case"  # "What happens when...?"
    ALTERNATIVE = "alternative"  # "What other approaches...?"
    CHALLENGE = "challenge"  # "But what about...?"
    IMPLEMENTATION = "implementation"  # "How exactly would you...?"
    CONSEQUENCE = "consequence"  # "What are the implications of...?"
    COMPARISON = "comparison"  # "How does this compare to...?"


class DrillDownState(BaseModel):
    """State for drill-down questioning"""
    # Original question context
    original_topic: str
    original_question: str
    job_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Answer history
    answers: List[Dict[str, Any]] = Field(default_factory=list)
    current_answer: str = ""
    
    # Depth tracking
    current_depth: int = 0
    max_depth: int = 4
    depth_assessment: str = AnswerDepth.SURFACE.value
    
    # Probing decisions
    should_probe_deeper: bool = True
    probe_type: str = ProbeType.ELABORATION.value
    probe_reason: str = ""
    
    # Generated content
    next_question: str = ""
    probe_context: str = ""
    
    # Skill verification
    skill_being_verified: str = ""
    confidence_in_skill: float = 0.0
    verification_notes: List[str] = Field(default_factory=list)
    
    # Contradiction detection
    contradictions_found: List[Dict[str, str]] = Field(default_factory=list)
    
    # Summary
    depth_reached: int = 0
    knowledge_type: str = "unknown"  # theoretical, practical, expert
    final_assessment: str = ""


class DrillDownQuestionEngine:
    """
    Intelligent probing engine that goes multiple levels deep
    on topics to verify genuine understanding and experience.
    """
    
    def __init__(self):
        logger.info("🔬 Initializing Drill-Down Question Engine...")
        
        # Azure OpenAI for intelligent probing
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.5
        )
        
        # Build drill-down workflow
        self.workflow = self._build_drill_down_workflow()
        
        # Probe templates by type
        self._init_probe_templates()
        
        logger.info("✅ Drill-Down Question Engine initialized")
    
    def _init_probe_templates(self):
        """Initialize probe question templates"""
        self.probe_templates = {
            ProbeType.CLARIFICATION: [
                "When you say '{term}', what exactly do you mean by that?",
                "Could you clarify what you mean by '{term}'?",
                "I want to make sure I understand - what do you mean when you say '{term}'?",
            ],
            ProbeType.ELABORATION: [
                "That's interesting. Can you tell me more about how that worked?",
                "I'd like to dig deeper into that. What else can you share?",
                "Can you elaborate on the {aspect} aspect of that?",
            ],
            ProbeType.SPECIFICITY: [
                "Can you walk me through a specific example of when you did this?",
                "What's a concrete instance where you applied this?",
                "Give me the specifics - what exactly did you do in that situation?",
            ],
            ProbeType.JUSTIFICATION: [
                "Why did you choose that particular approach?",
                "What was your reasoning behind that decision?",
                "What made you decide to go in that direction?",
            ],
            ProbeType.EDGE_CASE: [
                "What happens when {edge_case}?",
                "How would you handle a situation where {edge_case}?",
                "What if {edge_case}? How does that change things?",
            ],
            ProbeType.ALTERNATIVE: [
                "What other approaches did you consider?",
                "Were there alternatives you evaluated? Why didn't you go with those?",
                "If you couldn't use that approach, what would you do instead?",
            ],
            ProbeType.CHALLENGE: [
                "But what about {counterpoint}? How would you address that?",
                "I've seen that approach fail when {counterpoint}. How did you handle that?",
                "Some would argue that {counterpoint}. What's your take?",
            ],
            ProbeType.IMPLEMENTATION: [
                "Let's get into the weeds. How exactly would you implement that?",
                "Walk me through the technical implementation step by step.",
                "What are the actual code/system changes needed to do that?",
            ],
            ProbeType.CONSEQUENCE: [
                "What were the consequences of that decision?",
                "What trade-offs did you accept with that approach?",
                "How did that decision impact other parts of the system/team?",
            ],
            ProbeType.COMPARISON: [
                "How does this compare to {alternative}?",
                "Why this over {alternative}?",
                "What are the differences between your approach and {alternative}?",
            ]
        }
    
    def _build_drill_down_workflow(self) -> StateGraph:
        """Build LangGraph workflow for drill-down questioning"""
        workflow = StateGraph(DrillDownState)
        
        # Define nodes
        workflow.add_node("assess_answer_depth", self._assess_answer_depth)
        workflow.add_node("detect_contradictions", self._detect_contradictions)
        workflow.add_node("determine_probe_type", self._determine_probe_type)
        workflow.add_node("generate_probe_question", self._generate_probe_question)
        workflow.add_node("update_skill_verification", self._update_skill_verification)
        workflow.add_node("generate_summary", self._generate_summary)
        
        # Define edges
        workflow.set_entry_point("assess_answer_depth")
        workflow.add_edge("assess_answer_depth", "detect_contradictions")
        workflow.add_edge("detect_contradictions", "determine_probe_type")
        
        # Conditional edge - probe deeper or summarize
        workflow.add_conditional_edges(
            "determine_probe_type",
            self._should_continue_probing,
            {
                "probe": "generate_probe_question",
                "summarize": "generate_summary"
            }
        )
        
        workflow.add_edge("generate_probe_question", "update_skill_verification")
        workflow.add_edge("update_skill_verification", END)
        workflow.add_edge("generate_summary", END)
        
        return workflow.compile()
    
    async def _assess_answer_depth(self, state: DrillDownState) -> DrillDownState:
        """Assess the depth of the current answer"""
        try:
            prompt = f"""
            Assess the depth of this interview answer.
            
            QUESTION: {state.original_question}
            ANSWER: {state.current_answer}
            
            Depth levels:
            - surface: Basic, textbook answer. Could be memorized or Googled quickly.
            - moderate: Shows some understanding, but lacks specific details or examples.
            - deep: Provides specific details, examples, shows practical experience.
            - expert: Nuanced, mentions trade-offs, edge cases, shows mastery.
            
            Also identify:
            1. Key terms that could be probed
            2. Claims that need verification
            3. Missing details that would strengthen the answer
            
            Return JSON:
            {{
                "depth": "surface|moderate|deep|expert",
                "confidence": 0.8,
                "key_terms": ["term1", "term2"],
                "claims_to_verify": ["claim1"],
                "missing_details": ["detail1"],
                "knowledge_type": "theoretical|practical|expert",
                "reasoning": "Why this depth level"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.depth_assessment = result.get("depth", "surface")
            state.knowledge_type = result.get("knowledge_type", "theoretical")
            
            # Store answer with metadata
            state.answers.append({
                "depth_level": state.current_depth,
                "answer": state.current_answer,
                "depth_assessment": state.depth_assessment,
                "key_terms": result.get("key_terms", []),
                "claims": result.get("claims_to_verify", []),
                "missing": result.get("missing_details", [])
            })
            
            state.current_depth += 1
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Depth assessment error: {e}")
            state.depth_assessment = "moderate"
            return state
    
    async def _detect_contradictions(self, state: DrillDownState) -> DrillDownState:
        """Detect contradictions between current and previous answers"""
        if len(state.answers) < 2:
            return state
        
        try:
            previous_answers = [a["answer"] for a in state.answers[:-1]]
            current = state.answers[-1]["answer"]
            
            prompt = f"""
            Check for contradictions between these interview answers from the same candidate.
            
            PREVIOUS ANSWERS:
            {json.dumps(previous_answers, indent=2)}
            
            CURRENT ANSWER:
            {current}
            
            Identify any contradictions where the candidate says something that conflicts
            with what they said before.
            
            Return JSON:
            {{
                "contradictions": [
                    {{
                        "previous_statement": "what they said before",
                        "current_statement": "what they're saying now",
                        "severity": "minor|moderate|major"
                    }}
                ],
                "has_contradictions": true|false
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            if result.get("has_contradictions"):
                state.contradictions_found.extend(result.get("contradictions", []))
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Contradiction detection error: {e}")
            return state
    
    async def _determine_probe_type(self, state: DrillDownState) -> DrillDownState:
        """Determine the best type of probing question"""
        depth = state.depth_assessment
        current_level = state.current_depth
        contradictions = state.contradictions_found
        
        # Decision logic for probe type
        if contradictions and len(contradictions) > 0:
            # Address contradictions first
            state.probe_type = ProbeType.CLARIFICATION.value
            state.probe_reason = "Need to clarify contradicting statements"
        elif depth == AnswerDepth.SURFACE.value:
            # Surface answers need specificity
            if current_level == 1:
                state.probe_type = ProbeType.SPECIFICITY.value
                state.probe_reason = "Answer is too general, need specific example"
            else:
                state.probe_type = ProbeType.IMPLEMENTATION.value
                state.probe_reason = "Need to verify practical knowledge"
        elif depth == AnswerDepth.MODERATE.value:
            # Moderate answers need justification or edge cases
            if current_level <= 2:
                state.probe_type = ProbeType.JUSTIFICATION.value
                state.probe_reason = "Understand reasoning behind approach"
            else:
                state.probe_type = ProbeType.EDGE_CASE.value
                state.probe_reason = "Test edge case handling"
        elif depth == AnswerDepth.DEEP.value:
            # Deep answers can be challenged or compared
            state.probe_type = ProbeType.ALTERNATIVE.value
            state.probe_reason = "Explore alternative approaches considered"
        else:
            # Expert answers - explore trade-offs
            state.probe_type = ProbeType.CONSEQUENCE.value
            state.probe_reason = "Understand trade-off analysis"
        
        # Check if we should continue probing
        state.should_probe_deeper = (
            current_level < state.max_depth and
            depth != AnswerDepth.EXPERT.value
        )
        
        return state
    
    def _should_continue_probing(self, state: DrillDownState) -> str:
        """Determine if we should probe deeper or summarize"""
        if state.should_probe_deeper and state.current_depth < state.max_depth:
            return "probe"
        return "summarize"
    
    async def _generate_probe_question(self, state: DrillDownState) -> DrillDownState:
        """Generate the probing question"""
        try:
            # Get context from recent answers
            recent_answer = state.answers[-1] if state.answers else {}
            key_terms = recent_answer.get("key_terms", [])
            claims = recent_answer.get("claims", [])
            missing = recent_answer.get("missing", [])
            
            prompt = f"""
            Generate a probing follow-up question for an interview.
            
            TOPIC: {state.original_topic}
            ORIGINAL QUESTION: {state.original_question}
            CANDIDATE'S ANSWER: {state.current_answer}
            
            PROBE TYPE: {state.probe_type}
            REASON FOR PROBING: {state.probe_reason}
            CURRENT DEPTH LEVEL: {state.current_depth} of {state.max_depth}
            
            KEY TERMS TO EXPLORE: {key_terms}
            CLAIMS TO VERIFY: {claims}
            MISSING DETAILS: {missing}
            
            {f"CONTRADICTIONS TO ADDRESS: {json.dumps(state.contradictions_found)}" if state.contradictions_found else ""}
            
            Generate a natural, conversational follow-up question that:
            1. Probes deeper into the topic
            2. Sounds like a curious human interviewer
            3. Isn't aggressive but is persistent
            4. Targets the identified probe type
            
            Return JSON:
            {{
                "question": "The probing question",
                "context": "Brief intro if needed",
                "expected_depth_if_answered_well": "moderate|deep|expert"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            context = result.get("context", "")
            question = result.get("question", "")
            
            if context:
                state.next_question = f"{context} {question}"
            else:
                state.next_question = question
            
            state.probe_context = result.get("expected_depth_if_answered_well", "moderate")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Probe question generation error: {e}")
            # Fallback to template
            templates = self.probe_templates.get(
                ProbeType(state.probe_type),
                self.probe_templates[ProbeType.ELABORATION]
            )
            state.next_question = templates[0].format(
                term=state.original_topic,
                aspect="key",
                edge_case="conditions change",
                alternative="other common approaches",
                counterpoint="scalability concerns"
            )
            return state
    
    async def _update_skill_verification(self, state: DrillDownState) -> DrillDownState:
        """Update skill verification based on depth achieved"""
        depth_scores = {
            AnswerDepth.SURFACE.value: 0.25,
            AnswerDepth.MODERATE.value: 0.5,
            AnswerDepth.DEEP.value: 0.75,
            AnswerDepth.EXPERT.value: 0.95
        }
        
        # Calculate confidence based on depth and consistency
        base_confidence = depth_scores.get(state.depth_assessment, 0.5)
        
        # Penalize for contradictions
        contradiction_penalty = len(state.contradictions_found) * 0.1
        
        state.confidence_in_skill = max(0, base_confidence - contradiction_penalty)
        
        # Add verification note
        state.verification_notes.append(
            f"Depth {state.current_depth}: {state.depth_assessment} ({state.probe_type})"
        )
        
        return state
    
    async def _generate_summary(self, state: DrillDownState) -> DrillDownState:
        """Generate summary of the drill-down session"""
        try:
            prompt = f"""
            Summarize this drill-down interview session.
            
            TOPIC: {state.original_topic}
            SKILL BEING VERIFIED: {state.skill_being_verified or state.original_topic}
            
            ANSWERS AT EACH DEPTH:
            {json.dumps(state.answers, indent=2)}
            
            CONTRADICTIONS FOUND:
            {json.dumps(state.contradictions_found, indent=2) if state.contradictions_found else "None"}
            
            Generate a summary that:
            1. Assesses the candidate's true level of knowledge
            2. Distinguishes between theoretical and practical knowledge
            3. Notes any red flags or concerns
            4. Provides a skill confidence score
            
            Return JSON:
            {{
                "skill_verified": "{state.skill_being_verified or state.original_topic}",
                "knowledge_level": "beginner|intermediate|advanced|expert",
                "knowledge_type": "theoretical|practical|expert",
                "confidence_score": 0.75,
                "key_strengths": ["strength1"],
                "concerns": ["concern1"],
                "summary": "Brief narrative summary"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.depth_reached = state.current_depth
            state.knowledge_type = result.get("knowledge_type", state.knowledge_type)
            state.confidence_in_skill = result.get("confidence_score", state.confidence_in_skill)
            state.final_assessment = result.get("summary", "")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Summary generation error: {e}")
            state.final_assessment = f"Reached depth {state.current_depth}, knowledge appears {state.knowledge_type}"
            return state
    
    async def start_drill_down(
        self,
        topic: str,
        initial_question: str,
        initial_answer: str,
        skill_to_verify: Optional[str] = None,
        max_depth: int = 4,
        job_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Start a drill-down questioning session"""
        try:
            initial_state = DrillDownState(
                original_topic=topic,
                original_question=initial_question,
                current_answer=initial_answer,
                skill_being_verified=skill_to_verify or topic,
                max_depth=max_depth,
                job_context=job_context or {}
            )
            
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "success": True,
                "topic": topic,
                "next_question": final_state.next_question,
                "probe_type": final_state.probe_type,
                "current_depth": final_state.current_depth,
                "depth_assessment": final_state.depth_assessment,
                "should_continue": final_state.should_probe_deeper,
                "skill_confidence": final_state.confidence_in_skill,
                "contradictions": final_state.contradictions_found,
                "summary": final_state.final_assessment if not final_state.should_probe_deeper else None
            }
            
        except Exception as e:
            logger.error(f"❌ Drill-down start error: {e}")
            return {
                "success": False,
                "error": str(e),
                "next_question": f"Can you tell me more about {topic}?"
            }
    
    async def continue_drill_down(
        self,
        state_dict: Dict[str, Any],
        new_answer: str
    ) -> Dict[str, Any]:
        """Continue an existing drill-down session with a new answer"""
        try:
            # Reconstruct state from dict
            state = DrillDownState(**state_dict)
            state.current_answer = new_answer
            
            final_state = await self.workflow.ainvoke(state)
            
            return {
                "success": True,
                "topic": final_state.original_topic,
                "next_question": final_state.next_question,
                "probe_type": final_state.probe_type,
                "current_depth": final_state.current_depth,
                "depth_assessment": final_state.depth_assessment,
                "should_continue": final_state.should_probe_deeper,
                "skill_confidence": final_state.confidence_in_skill,
                "contradictions": final_state.contradictions_found,
                "knowledge_type": final_state.knowledge_type,
                "summary": final_state.final_assessment if not final_state.should_probe_deeper else None,
                "state": final_state.dict()  # Return state for next iteration
            }
            
        except Exception as e:
            logger.error(f"❌ Drill-down continue error: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Singleton instance
_drill_down_engine = None

def get_drill_down_engine() -> DrillDownQuestionEngine:
    """Get singleton drill-down engine instance"""
    global _drill_down_engine
    if _drill_down_engine is None:
        _drill_down_engine = DrillDownQuestionEngine()
    return _drill_down_engine
