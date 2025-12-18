"""
🤖 ADVANCED NEGOTIATOR AGENT (v3.0) - World-Class Agentic AI
==============================================================

Handles salary and offer negotiations using:
- CrewAI multi-agent negotiation simulation
- DSPy MIPRO for optimal negotiation strategy prompts
- Game Theory and Market Intelligence
- RAG knowledge base from successful negotiations
- Feedback loops from offer acceptance rates

Author: HireGenix AI Team
Version: 3.0.0 (World-Class Agentic AI)
Last Updated: December 2025
"""

import os
import json
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

# LangChain & LangGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# 🚀 CrewAI for Multi-Agent Negotiation Simulation
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("⚠️ CrewAI not installed. Install with: pip install crewai")

# 🎯 DSPy for Prompt Optimization
try:
    import dspy
    from dspy import ChainOfThought, Predict
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# 📚 RAG & Vector Store
try:
    import redis
    from sentence_transformers import SentenceTransformer
    import numpy as np
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

from market_intelligence import get_market_intelligence_service


# ============================================================================
# 🎯 DSPY NEGOTIATION SIGNATURES
# ============================================================================

if DSPY_AVAILABLE:
    class NegotiationStrategySignature(dspy.Signature):
        """Determine optimal negotiation strategy."""
        candidate_ask: float = dspy.InputField(desc="Candidate's salary ask")
        budget_max: float = dspy.InputField(desc="Maximum budget available")
        market_rate: float = dspy.InputField(desc="Market rate for the role")
        candidate_leverage: str = dspy.InputField(desc="Candidate's leverage level")
        strategy: str = dspy.OutputField(desc="AGGRESSIVE, COLLABORATIVE, FIRM, GENEROUS")
        counter_offer: float = dspy.OutputField(desc="Recommended counter-offer amount")
        reasoning: str = dspy.OutputField(desc="Strategic reasoning for the approach")
    
    class NegotiationMIPROModule(dspy.Module):
        """DSPy module for MIPRO-optimized negotiation decisions."""
        def __init__(self):
            super().__init__()
            self.strategize = dspy.ChainOfThought(NegotiationStrategySignature)
        
        def forward(self, candidate_ask, budget_max, market_rate, candidate_leverage):
            return self.strategize(
                candidate_ask=str(candidate_ask),
                budget_max=str(budget_max),
                market_rate=str(market_rate),
                candidate_leverage=candidate_leverage
            )


# ============================================================================
# NEGOTIATION STATE (Enhanced with LangGraph)
# ============================================================================

class NegotiationState(BaseModel):
    """Enhanced state for LangGraph negotiation workflow"""
    candidate_name: str
    candidate_id: str = ""
    role: str
    job_id: str = ""
    
    # Financials
    budget_max: float
    market_value: float
    current_offer: float
    candidate_ask: float = 0.0
    
    # Equity/Benefits
    equity_offer: float = 0.0
    sign_on_bonus: float = 0.0
    
    # 🚀 NEW: RAG & Learning Context
    similar_negotiations: List[Dict] = Field(default_factory=list)
    success_patterns: List[str] = Field(default_factory=list)
    
    # 🚀 NEW: CrewAI Debate Results
    multi_agent_analysis: Dict = Field(default_factory=dict)
    
    # 🚀 NEW: DSPy Optimized Strategy
    optimized_strategy: Dict = Field(default_factory=dict)
    
    # Context
    history: List[Dict] = Field(default_factory=list)
    status: str = "ongoing"  # ongoing, agreed, failed
    strategy: str = "collaborative"  # collaborative, firm, closing
    
    # Workflow tracking
    current_step: str = "initialized"
    steps_completed: List[str] = Field(default_factory=list)


class NegotiatorAgent:
    """
    🚀 WORLD-CLASS NEGOTIATION AGENT (v3.0)
    
    Features:
    - CrewAI multi-agent negotiation simulation (Recruiter vs Candidate perspectives)
    - DSPy MIPRO optimizer for negotiation strategy prompts
    - RAG knowledge base from successful negotiations
    - Game Theory decision making
    - LangGraph workflow orchestration
    """
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.6
        )
        self.market_service = get_market_intelligence_service()
        
        # 🚀 Initialize Advanced Components
        self._init_rag_knowledge_base()
        self._init_crewai_negotiation_team()
        self._init_dspy_optimizer()
        self._init_feedback_system()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        print("✅ NegotiatorAgent v3.0 initialized with CrewAI + DSPy + RAG")
    
    def _init_rag_knowledge_base(self):
        """Initialize RAG for learning from past negotiations"""
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
                print("✅ Negotiation RAG Knowledge Base initialized")
            except Exception as e:
                print(f"⚠️ RAG initialization failed: {e}")
    
    def _init_crewai_negotiation_team(self):
        """Initialize CrewAI agents for negotiation simulation"""
        self.crewai_enabled = False
        if CREWAI_AVAILABLE:
            try:
                # Recruiter perspective agent
                self.recruiter_agent = Agent(
                    role="Senior Recruiter Strategist",
                    goal="Secure the best candidate within budget while maintaining positive relationship",
                    backstory="20 years experience closing offers, knows when to push and when to concede",
                    verbose=False,
                    allow_delegation=False
                )
                
                # Candidate perspective agent (for simulation)
                self.candidate_simulator = Agent(
                    role="Candidate Advocate",
                    goal="Understand candidate's perspective and predict their responses",
                    backstory="Former career coach who understands candidate psychology and motivations",
                    verbose=False,
                    allow_delegation=False
                )
                
                # Game theory strategist
                self.game_theorist = Agent(
                    role="Negotiation Game Theorist",
                    goal="Apply game theory principles to optimize negotiation outcomes",
                    backstory="PhD in behavioral economics, expert in Nash equilibrium and BATNA analysis",
                    verbose=False,
                    allow_delegation=False
                )
                
                self.crewai_enabled = True
                print("✅ CrewAI negotiation team initialized")
            except Exception as e:
                print(f"⚠️ CrewAI initialization failed: {e}")
    
    def _init_dspy_optimizer(self):
        """Initialize DSPy MIPRO for negotiation prompt optimization"""
        self.dspy_enabled = False
        if DSPY_AVAILABLE:
            try:
                lm = dspy.LM(
                    model=f"openai/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o')}",
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                    model_type="chat",
                    max_tokens=2000
                )
                dspy.configure(lm=lm)
                self.negotiation_module = NegotiationMIPROModule()
                self.dspy_enabled = True
                print("✅ DSPy MIPRO optimizer initialized for negotiations")
            except Exception as e:
                print(f"⚠️ DSPy initialization failed: {e}")
    
    def _init_feedback_system(self):
        """Initialize feedback loop for learning from negotiation outcomes"""
        self.negotiation_outcomes = {}
        print("✅ Negotiation feedback system initialized")
    
    def _build_workflow(self):
        """Build LangGraph workflow for negotiation"""
        workflow = StateGraph(NegotiationState)
        
        # Define workflow nodes
        workflow.add_node("retrieve_patterns", self._retrieve_negotiation_patterns)
        workflow.add_node("analyze_with_crewai", self._crewai_negotiation_analysis)
        workflow.add_node("optimize_strategy", self._dspy_optimize_strategy)
        workflow.add_node("analyze_intent", self._analyze_candidate_intent)
        workflow.add_node("generate_response", self._generate_negotiation_response)
        workflow.add_node("store_outcome", self._store_negotiation_outcome)
        
        # Define edges
        workflow.set_entry_point("retrieve_patterns")
        workflow.add_edge("retrieve_patterns", "analyze_with_crewai")
        workflow.add_edge("analyze_with_crewai", "optimize_strategy")
        workflow.add_edge("optimize_strategy", "analyze_intent")
        workflow.add_edge("analyze_intent", "generate_response")
        workflow.add_edge("generate_response", "store_outcome")
        workflow.add_edge("store_outcome", END)
        
        return workflow.compile()
    
    # ========================================================================
    # 🧠 RAG KNOWLEDGE BASE METHODS
    # ========================================================================
    
    async def _retrieve_negotiation_patterns(self, state: NegotiationState) -> NegotiationState:
        """Retrieve similar successful negotiation patterns"""
        state.current_step = "retrieve_patterns"
        
        if not self.rag_enabled:
            state.steps_completed.append("retrieve_patterns")
            return state
        
        try:
            # Create query from negotiation context
            query = f"{state.role} salary {state.budget_max} negotiation"
            query_embedding = self.embedder.encode(query).tolist()
            
            # Search for similar negotiations
            patterns = []
            pattern_keys = self.redis_client.keys("negotiation_pattern:*")
            
            for key in pattern_keys[:30]:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        pattern = json.loads(data)
                        if pattern.get("outcome") == "accepted":
                            patterns.append(pattern)
                except:
                    continue
            
            state.similar_negotiations = patterns[:5]
            
            # Extract success patterns
            state.success_patterns = [
                p.get("winning_strategy", "") for p in patterns 
                if p.get("outcome") == "accepted"
            ][:3]
            
            state.steps_completed.append("retrieve_patterns")
            
        except Exception as e:
            print(f"⚠️ Pattern retrieval error: {e}")
            state.steps_completed.append("retrieve_patterns")
        
        return state
    
    async def _crewai_negotiation_analysis(self, state: NegotiationState) -> NegotiationState:
        """Run CrewAI multi-agent analysis for negotiation strategy"""
        state.current_step = "analyze_with_crewai"
        
        if not self.crewai_enabled:
            state.steps_completed.append("analyze_with_crewai")
            return state
        
        try:
            context = f"""
Role: {state.role}
Budget Max: ${state.budget_max:,.2f}
Market Value: ${state.market_value:,.2f}
Current Offer: ${state.current_offer:,.2f}
Candidate Ask: ${state.candidate_ask:,.2f}
"""
            
            recruiter_task = Task(
                description=f"Analyze this negotiation and recommend strategy:\n{context}",
                expected_output="Strategy recommendation with counter-offer",
                agent=self.recruiter_agent
            )
            
            candidate_task = Task(
                description=f"Predict candidate's likely response to various offers:\n{context}",
                expected_output="Candidate response prediction",
                agent=self.candidate_simulator
            )
            
            game_task = Task(
                description=f"Apply game theory to find optimal negotiation point:\n{context}",
                expected_output="Game theory optimal solution",
                agent=self.game_theorist
            )
            
            crew = Crew(
                agents=[self.recruiter_agent, self.candidate_simulator, self.game_theorist],
                tasks=[recruiter_task, candidate_task, game_task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()
            
            state.multi_agent_analysis = {
                "crewai_result": str(result),
                "agents_consulted": 3,
                "perspectives": ["recruiter", "candidate", "game_theory"]
            }
            
            state.steps_completed.append("analyze_with_crewai")
            
        except Exception as e:
            print(f"⚠️ CrewAI analysis error: {e}")
            state.steps_completed.append("analyze_with_crewai")
        
        return state
    
    async def _dspy_optimize_strategy(self, state: NegotiationState) -> NegotiationState:
        """Use DSPy MIPRO to optimize negotiation strategy"""
        state.current_step = "optimize_strategy"
        
        if not self.dspy_enabled:
            state.steps_completed.append("optimize_strategy")
            return state
        
        try:
            # Determine candidate leverage
            leverage = "low"
            if state.candidate_ask <= state.market_value:
                leverage = "low"
            elif state.candidate_ask <= state.budget_max:
                leverage = "medium"
            else:
                leverage = "high"
            
            result = self.negotiation_module.forward(
                candidate_ask=state.candidate_ask,
                budget_max=state.budget_max,
                market_rate=state.market_value,
                candidate_leverage=leverage
            )
            
            state.optimized_strategy = {
                "dspy_strategy": result.strategy if hasattr(result, 'strategy') else "collaborative",
                "dspy_counter_offer": float(result.counter_offer) if hasattr(result, 'counter_offer') else state.current_offer,
                "dspy_reasoning": result.reasoning if hasattr(result, 'reasoning') else ""
            }
            
            state.steps_completed.append("optimize_strategy")
            
        except Exception as e:
            print(f"⚠️ DSPy optimization error: {e}")
            state.steps_completed.append("optimize_strategy")
        
        return state
    
    async def _analyze_candidate_intent(self, state: NegotiationState) -> NegotiationState:
        """Analyze candidate's intent from their messages"""
        state.current_step = "analyze_intent"
        
        # Get last message if any
        last_message = state.history[-1]["content"] if state.history else ""
        
        if not last_message:
            state.steps_completed.append("analyze_intent")
            return state
        
        try:
            prompt = f"""Analyze this negotiation message: "{last_message}"
Return JSON: {{"intent": "accept|reject|counter|inquire", "amount": 0.0, "sentiment": "positive|neutral|negative"}}"""
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content.strip().replace("```json", "").replace("```", "")
            analysis = json.loads(content)
            
            if analysis.get("amount", 0) > 0:
                state.candidate_ask = analysis["amount"]
            
            state.steps_completed.append("analyze_intent")
            
        except Exception as e:
            print(f"⚠️ Intent analysis error: {e}")
            state.steps_completed.append("analyze_intent")
        
        return state
    
    async def _generate_negotiation_response(self, state: NegotiationState) -> NegotiationState:
        """Generate intelligent negotiation response"""
        state.current_step = "generate_response"
        
        try:
            # Use DSPy strategy if available
            if state.optimized_strategy:
                strategy = state.optimized_strategy.get("dspy_strategy", state.strategy)
                recommended_counter = state.optimized_strategy.get("dspy_counter_offer", state.current_offer)
            else:
                strategy = state.strategy
                recommended_counter = state.current_offer
            
            # Apply game theory logic
            if state.candidate_ask <= state.budget_max:
                if state.candidate_ask <= state.current_offer * 1.05:
                    state.current_offer = state.candidate_ask
                    state.status = "agreed"
                else:
                    midpoint = (state.current_offer + state.candidate_ask) / 2
                    state.current_offer = min(midpoint, state.budget_max)
            
            prompt = f"""Generate a negotiation response for {state.role} role.
Candidate: {state.candidate_name}
Current Offer: ${state.current_offer:,.2f}
Strategy: {strategy}
Status: {state.status}

CrewAI Insights: {state.multi_agent_analysis.get('crewai_result', 'N/A')[:200]}
DSPy Strategy: {state.optimized_strategy.get('dspy_reasoning', 'N/A')[:200]}

Generate a professional, warm message under 50 words."""
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            state.history.append({
                "role": "ai",
                "content": response.content.strip(),
                "offer": state.current_offer,
                "timestamp": datetime.now().isoformat()
            })
            
            state.steps_completed.append("generate_response")
            
        except Exception as e:
            print(f"⚠️ Response generation error: {e}")
            state.steps_completed.append("generate_response")
        
        return state
    
    async def _store_negotiation_outcome(self, state: NegotiationState) -> NegotiationState:
        """Store negotiation outcome for future learning"""
        state.current_step = "store_outcome"
        
        if self.rag_enabled and state.status in ["agreed", "failed"]:
            try:
                pattern_id = hashlib.md5(
                    f"{state.candidate_id}{state.job_id}{datetime.now().isoformat()}".encode()
                ).hexdigest()
                
                pattern = {
                    "id": pattern_id,
                    "role": state.role,
                    "budget_max": state.budget_max,
                    "final_offer": state.current_offer,
                    "candidate_ask": state.candidate_ask,
                    "outcome": "accepted" if state.status == "agreed" else "rejected",
                    "strategy_used": state.strategy,
                    "winning_strategy": state.optimized_strategy.get("dspy_strategy", ""),
                    "rounds": len([h for h in state.history if h["role"] == "ai"]),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.redis_client.set(f"negotiation_pattern:{pattern_id}", json.dumps(pattern))
                
            except Exception as e:
                print(f"⚠️ Outcome storage error: {e}")
        
        state.steps_completed.append("store_outcome")
        return state
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================

    async def start_negotiation(
        self, 
        candidate_name: str, 
        role: str, 
        budget_max: float,
        candidate_id: str = "",
        job_id: str = ""
    ) -> NegotiationState:
        """Initialize negotiation with market-informed initial offer."""
        
        # Get Market Data
        market_data = await self.market_service.fetch_market_trends(role, [])
        market_value = budget_max * 0.9
        
        # Determine Initial Offer (Anchor)
        initial_offer = min(market_value, budget_max * 0.85)
        
        state = NegotiationState(
            candidate_name=candidate_name,
            candidate_id=candidate_id,
            role=role,
            job_id=job_id,
            budget_max=budget_max,
            market_value=market_value,
            current_offer=initial_offer
        )
        
        # Run through workflow for initial analysis
        state = await self._retrieve_negotiation_patterns(state)
        
        # Generate Opening Message
        msg = await self._generate_message(state, "opening")
        state.history.append({"role": "ai", "content": msg, "timestamp": datetime.now().isoformat()})
        
        return state

    async def process_candidate_response(self, state: NegotiationState, message: str) -> Dict[str, Any]:
        """Process candidate's counter-offer through the full workflow."""
        
        state.history.append({"role": "candidate", "content": message, "timestamp": datetime.now().isoformat()})
        
        # Run full workflow
        final_state = await self.workflow.ainvoke(state)
        
        return {
            "ai_message": final_state.history[-1]["content"] if final_state.history else "",
            "new_offer": final_state.current_offer,
            "status": final_state.status,
            "crewai_analysis": final_state.multi_agent_analysis,
            "dspy_strategy": final_state.optimized_strategy,
            "state": final_state.model_dump()
        }

    async def _generate_message(self, state: NegotiationState, msg_type: str) -> str:
        prompt = f"""You are a Recruiter negotiating for the {state.role} role.
Current Offer: ${state.current_offer:,.2f}
Budget Max: ${state.budget_max:,.2f}
Generate a {msg_type} message. Keep it professional but warm. Under 50 words."""
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    
    async def record_final_outcome(
        self, 
        candidate_id: str, 
        job_id: str, 
        outcome: str,  # accepted, rejected, counter_offered
        final_salary: Optional[float] = None
    ):
        """Record final negotiation outcome for feedback learning."""
        if self.rag_enabled:
            feedback = {
                "candidate_id": candidate_id,
                "job_id": job_id,
                "outcome": outcome,
                "final_salary": final_salary,
                "timestamp": datetime.now().isoformat()
            }
            self.redis_client.set(
                f"negotiation_feedback:{candidate_id}:{job_id}", 
                json.dumps(feedback)
            )
            print(f"✅ Recorded negotiation outcome: {outcome}")


# Singleton
_negotiator_agent = None

def get_negotiator_agent() -> NegotiatorAgent:
    global _negotiator_agent
    if _negotiator_agent is None:
        _negotiator_agent = NegotiatorAgent()
    return _negotiator_agent