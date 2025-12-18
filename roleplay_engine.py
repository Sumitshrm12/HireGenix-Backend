"""
Roleplay Simulation Engine
Simulates real-world workplace scenarios to assess soft skills, conflict resolution, and practical problem-solving.
Uses LangGraph to maintain state and persona consistency (e.g., an "Angry Customer" stays angry until appeased).
"""

import os
import json
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

class ScenarioType(str, Enum):
    CONFLICT_RESOLUTION = "conflict_resolution"
    CRISIS_MANAGEMENT = "crisis_management"
    STAKEHOLDER_NEGOTIATION = "stakeholder_negotiation"
    MENTORSHIP = "mentorship"

@dataclass
class RoleplayState:
    """State for the roleplay simulation"""
    scenario_id: str
    candidate_name: str
    role: str # e.g., "Product Manager"
    
    # Simulation Context
    scenario_description: str
    ai_persona: str # e.g., "Angry Customer"
    ai_mood: str # e.g., "Furious"
    
    # Conversation
    history: List[Dict[str, str]] = field(default_factory=list)
    last_candidate_message: str = ""
    last_ai_response: str = ""
    
    # Assessment
    turns_count: int = 0
    max_turns: int = 5
    objectives_met: List[str] = field(default_factory=list)
    soft_skills_score: float = 0.0
    feedback: str = ""
    status: str = "active" # active, completed, failed

class RoleplayEngine:
    """
    Orchestrates interactive roleplay scenarios.
    """
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.7
        )
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        workflow = StateGraph(RoleplayState)
        
        workflow.add_node("process_turn", self._process_turn)
        workflow.add_node("evaluate_state", self._evaluate_state)
        workflow.add_node("generate_response", self._generate_response)
        
        workflow.set_entry_point("process_turn")
        
        workflow.add_edge("process_turn", "evaluate_state")
        
        workflow.add_conditional_edges(
            "evaluate_state",
            self._check_completion,
            {
                "continue": "generate_response",
                "complete": END
            }
        )
        
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()

    async def start_simulation(self, scenario_type: ScenarioType, candidate_name: str, role: str) -> RoleplayState:
        """Initializes a new simulation scenario."""
        
        scenarios = {
            ScenarioType.CONFLICT_RESOLUTION: {
                "description": "You are a Team Lead. A senior engineer (AI) is refusing to adopt a new technology stack that the rest of the team agreed on.",
                "ai_persona": "Stubborn Senior Engineer",
                "ai_mood": "Defensive and skeptical"
            },
            ScenarioType.CRISIS_MANAGEMENT: {
                "description": "You are a DevOps Engineer. The production database just went down during Black Friday. The CTO (AI) is panicking and asking for updates every 30 seconds.",
                "ai_persona": "Panicked CTO",
                "ai_mood": "High anxiety and demanding"
            },
            ScenarioType.STAKEHOLDER_NEGOTIATION: {
                "description": "You are a Product Manager. A key client (AI) is demanding a feature that is not on the roadmap and will delay the release.",
                "ai_persona": "Demanding Client",
                "ai_mood": "Impatient and entitled"
            }
        }
        
        config = scenarios.get(scenario_type, scenarios[ScenarioType.CONFLICT_RESOLUTION])
        
        state = RoleplayState(
            scenario_id=scenario_type.value,
            candidate_name=candidate_name,
            role=role,
            scenario_description=config["description"],
            ai_persona=config["ai_persona"],
            ai_mood=config["ai_mood"]
        )
        
        # Initial AI message
        initial_msg = await self._generate_initial_message(state)
        state.history.append({"role": "ai", "content": initial_msg})
        state.last_ai_response = initial_msg
        
        return state

    async def process_message(self, state: RoleplayState, message: str) -> RoleplayState:
        """Processes a candidate's message and advances the simulation."""
        state.last_candidate_message = message
        state.history.append({"role": "candidate", "content": message})
        state.turns_count += 1
        
        return await self.workflow.ainvoke(state)

    async def _process_turn(self, state: RoleplayState) -> RoleplayState:
        """Placeholder for any pre-processing logic."""
        return state

    async def _evaluate_state(self, state: RoleplayState) -> RoleplayState:
        """Evaluates the candidate's move and updates AI mood/score."""
        prompt = f"""
        Scenario: {state.scenario_description}
        AI Persona: {state.ai_persona} (Current Mood: {state.ai_mood})
        Candidate Role: {state.role}
        
        Candidate's Latest Message: "{state.last_candidate_message}"
        
        Analyze the candidate's response:
        1. Did they de-escalate or escalate the situation?
        2. Did they show empathy?
        3. Did they propose a solution?
        
        Update the AI's mood (e.g., from "Angry" to "Listening" or "Furious").
        Rate the response (0-10).
        
        Return JSON: {{ "new_mood": "...", "score": 8.5, "feedback": "..." }}
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content.strip().replace("```json", "").replace("```", "")
            data = json.loads(content)
            
            state.ai_mood = data.get("new_mood", state.ai_mood)
            state.soft_skills_score = (state.soft_skills_score * (state.turns_count - 1) + data.get("score", 5)) / state.turns_count
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            
        return state

    def _check_completion(self, state: RoleplayState) -> str:
        if state.turns_count >= state.max_turns:
            state.status = "completed"
            return "complete"
        if "resolved" in state.ai_mood.lower() or "happy" in state.ai_mood.lower():
            state.status = "completed"
            return "complete"
        return "continue"

    async def _generate_response(self, state: RoleplayState) -> RoleplayState:
        """Generates the AI persona's next response."""
        prompt = f"""
        You are roleplaying as: {state.ai_persona}
        Current Mood: {state.ai_mood}
        Scenario: {state.scenario_description}
        
        Conversation History:
        {json.dumps(state.history[-3:])}
        
        Respond to the candidate naturally. Stay in character.
        If you are angry, be short. If you are happy, be cooperative.
        Keep it under 50 words.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        ai_msg = response.content.strip()
        
        state.last_ai_response = ai_msg
        state.history.append({"role": "ai", "content": ai_msg})
        
        return state

    async def _generate_initial_message(self, state: RoleplayState) -> str:
        prompt = f"""
        You are roleplaying as: {state.ai_persona}
        Mood: {state.ai_mood}
        Scenario: {state.scenario_description}
        
        Start the conversation with a complaint or urgent request relevant to the scenario.
        Keep it under 30 words.
        """
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

# Singleton
_roleplay_engine = None

def get_roleplay_engine() -> RoleplayEngine:
    global _roleplay_engine
    if _roleplay_engine is None:
        _roleplay_engine = RoleplayEngine()
    return _roleplay_engine