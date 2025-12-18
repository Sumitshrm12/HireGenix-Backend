"""
💻 LIVE CODING OBSERVER - Real-Time Code Analysis Agent
Monitors candidate's coding in real-time during technical interviews,
providing intelligent analysis of problem-solving approach, code quality,
and debugging strategies.

Features:
- Real-time code change tracking
- Solution approach analysis
- Bug detection before execution
- Code quality scoring
- Debugging behavior analysis
- Time management tracking
- Hint system for stuck candidates
- Multi-language support

Tech Stack:
- LangGraph for analysis workflow
- Tree-sitter for code parsing
- LangChain for intelligent analysis
- WebSocket for real-time updates
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import difflib
import logging

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodingPhase(str, Enum):
    """Phases of problem-solving"""
    READING = "reading"  # Reading the problem
    PLANNING = "planning"  # Thinking about approach
    INITIAL_CODING = "initial_coding"  # First implementation attempt
    ITERATING = "iterating"  # Refining solution
    DEBUGGING = "debugging"  # Fixing bugs
    TESTING = "testing"  # Testing solution
    OPTIMIZING = "optimizing"  # Performance improvements
    STUCK = "stuck"  # No progress for extended time


class CodeQuality(str, Enum):
    """Code quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"


class ProgrammingLanguage(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"


class LiveCodingState(BaseModel):
    """State for live coding observation"""
    # Session info
    session_id: str = ""
    candidate_id: str = ""
    problem_id: str = ""
    
    # Problem context
    problem_statement: str = ""
    expected_approach: List[str] = Field(default_factory=list)
    difficulty: str = "medium"
    time_limit_minutes: int = 45
    
    # Current code
    current_code: str = ""
    language: str = ProgrammingLanguage.PYTHON.value
    code_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Progress tracking
    current_phase: str = CodingPhase.READING.value
    phase_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    time_elapsed_seconds: int = 0
    
    # Analysis results
    approach_analysis: Dict[str, Any] = Field(default_factory=dict)
    detected_bugs: List[Dict[str, Any]] = Field(default_factory=list)
    code_quality: Dict[str, Any] = Field(default_factory=dict)
    
    # Behavioral observations
    typing_patterns: Dict[str, Any] = Field(default_factory=dict)
    delete_ratio: float = 0.0  # Ratio of deleted to written code
    refactor_count: int = 0
    
    # Problem-solving indicators
    uses_tests: bool = False
    reads_error_messages: bool = False
    breaks_down_problem: bool = False
    uses_helper_functions: bool = False
    
    # Intervention signals
    stuck_duration_seconds: int = 0
    should_provide_hint: bool = False
    hint_level: int = 0
    
    # Scoring
    problem_solving_score: float = 0.0
    code_quality_score: float = 0.0
    efficiency_score: float = 0.0
    overall_assessment: str = ""


class LiveCodingObserver:
    """
    Real-time code observation agent for technical interviews.
    Analyzes coding behavior, approach, and provides intelligent insights.
    """
    
    def __init__(self):
        logger.info("💻 Initializing Live Coding Observer...")
        
        # LLM for code analysis
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.2  # Lower temperature for code analysis
        )
        
        # Build observation workflow
        self.workflow = self._build_observation_workflow()
        
        # Language-specific patterns
        self._init_language_patterns()
        
        # Active sessions
        self.sessions: Dict[str, LiveCodingState] = {}
        
        logger.info("✅ Live Coding Observer initialized")
    
    def _init_language_patterns(self):
        """Initialize language-specific patterns for analysis"""
        self.language_patterns = {
            ProgrammingLanguage.PYTHON.value: {
                "function_def": r"def\s+\w+\s*\(",
                "class_def": r"class\s+\w+",
                "test_patterns": ["assert ", "pytest", "unittest", "def test_"],
                "common_bugs": ["off_by_one", "none_check", "indent_error"],
                "best_practices": ["type_hints", "docstrings", "list_comprehension"]
            },
            ProgrammingLanguage.JAVASCRIPT.value: {
                "function_def": r"function\s+\w+|const\s+\w+\s*=\s*\(|=>\s*{",
                "class_def": r"class\s+\w+",
                "test_patterns": ["describe(", "it(", "expect(", "test("],
                "common_bugs": ["undefined_check", "async_await", "equality_check"],
                "best_practices": ["const_let", "arrow_functions", "destructuring"]
            },
            ProgrammingLanguage.JAVA.value: {
                "function_def": r"(public|private|protected)\s+\w+\s+\w+\s*\(",
                "class_def": r"class\s+\w+",
                "test_patterns": ["@Test", "assertEquals", "assertTrue"],
                "common_bugs": ["null_pointer", "array_bounds", "type_mismatch"],
                "best_practices": ["encapsulation", "naming_conventions", "exception_handling"]
            }
        }
    
    def _build_observation_workflow(self) -> StateGraph:
        """Build LangGraph workflow for code observation"""
        workflow = StateGraph(LiveCodingState)
        
        # Define nodes
        workflow.add_node("detect_phase", self._detect_phase)
        workflow.add_node("analyze_code_changes", self._analyze_code_changes)
        workflow.add_node("detect_bugs", self._detect_bugs)
        workflow.add_node("analyze_approach", self._analyze_approach)
        workflow.add_node("assess_quality", self._assess_quality)
        workflow.add_node("check_intervention_needed", self._check_intervention_needed)
        workflow.add_node("generate_insights", self._generate_insights)
        
        # Define edges
        workflow.set_entry_point("detect_phase")
        workflow.add_edge("detect_phase", "analyze_code_changes")
        workflow.add_edge("analyze_code_changes", "detect_bugs")
        workflow.add_edge("detect_bugs", "analyze_approach")
        workflow.add_edge("analyze_approach", "assess_quality")
        workflow.add_edge("assess_quality", "check_intervention_needed")
        workflow.add_edge("check_intervention_needed", "generate_insights")
        workflow.add_edge("generate_insights", END)
        
        return workflow.compile()
    
    async def _detect_phase(self, state: LiveCodingState) -> LiveCodingState:
        """Detect current problem-solving phase"""
        try:
            code = state.current_code
            history = state.code_history
            
            # No code yet
            if not code or len(code.strip()) < 10:
                if state.time_elapsed_seconds < 60:
                    state.current_phase = CodingPhase.READING.value
                elif state.time_elapsed_seconds < 180:
                    state.current_phase = CodingPhase.PLANNING.value
                else:
                    state.current_phase = CodingPhase.STUCK.value
                    state.stuck_duration_seconds = state.time_elapsed_seconds - 180
                return state
            
            # Check for testing patterns
            lang_patterns = self.language_patterns.get(state.language, {})
            test_patterns = lang_patterns.get("test_patterns", [])
            has_tests = any(pattern in code for pattern in test_patterns)
            
            if has_tests:
                state.current_phase = CodingPhase.TESTING.value
                state.uses_tests = True
                return state
            
            # Analyze recent changes
            if len(history) >= 2:
                recent_changes = self._calculate_changes(
                    history[-2].get("code", ""),
                    code
                )
                
                # Lots of deletions = debugging or refactoring
                if recent_changes.get("deletions", 0) > recent_changes.get("additions", 0):
                    state.current_phase = CodingPhase.DEBUGGING.value
                    return state
                
                # Small focused changes = iterating
                if recent_changes.get("additions", 0) < 20:
                    state.current_phase = CodingPhase.ITERATING.value
                    return state
            
            # Check for optimization patterns
            optimization_keywords = ["optimize", "faster", "O(", "complexity", "cache"]
            if any(kw in code.lower() for kw in optimization_keywords):
                state.current_phase = CodingPhase.OPTIMIZING.value
                return state
            
            # Default to initial coding if writing code
            state.current_phase = CodingPhase.INITIAL_CODING.value
            
            # Track phase transition
            state.phase_timeline.append({
                "phase": state.current_phase,
                "timestamp": datetime.now().isoformat(),
                "time_elapsed": state.time_elapsed_seconds
            })
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Phase detection error: {e}")
            return state
    
    def _calculate_changes(self, old_code: str, new_code: str) -> Dict[str, int]:
        """Calculate code changes between versions"""
        old_lines = old_code.split('\n')
        new_lines = new_code.split('\n')
        
        diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=''))
        
        additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        
        return {
            "additions": additions,
            "deletions": deletions,
            "net_change": additions - deletions
        }
    
    async def _analyze_code_changes(self, state: LiveCodingState) -> LiveCodingState:
        """Analyze code changes over time"""
        try:
            if len(state.code_history) < 2:
                return state
            
            # Calculate overall typing patterns
            total_additions = 0
            total_deletions = 0
            
            for i in range(1, len(state.code_history)):
                changes = self._calculate_changes(
                    state.code_history[i-1].get("code", ""),
                    state.code_history[i].get("code", "")
                )
                total_additions += changes["additions"]
                total_deletions += changes["deletions"]
            
            # Calculate delete ratio
            if total_additions > 0:
                state.delete_ratio = total_deletions / total_additions
            
            # Count refactors (significant restructuring)
            refactor_count = 0
            for i in range(1, len(state.code_history)):
                changes = self._calculate_changes(
                    state.code_history[i-1].get("code", ""),
                    state.code_history[i].get("code", "")
                )
                if changes["deletions"] > 10 and changes["additions"] > 10:
                    refactor_count += 1
            
            state.refactor_count = refactor_count
            
            # Typing patterns
            state.typing_patterns = {
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "delete_ratio": state.delete_ratio,
                "refactor_count": refactor_count,
                "average_change_size": total_additions / len(state.code_history)
            }
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Code change analysis error: {e}")
            return state
    
    async def _detect_bugs(self, state: LiveCodingState) -> LiveCodingState:
        """Detect potential bugs in the code"""
        try:
            if not state.current_code:
                return state
            
            prompt = f"""
            Analyze this code for potential bugs. Do NOT execute the code.
            
            LANGUAGE: {state.language}
            PROBLEM: {state.problem_statement[:500]}
            
            CODE:
            ```{state.language}
            {state.current_code}
            ```
            
            Identify:
            1. Syntax errors
            2. Logic errors (off-by-one, null checks, etc.)
            3. Edge case handling issues
            4. Performance issues (if obvious)
            
            Return JSON:
            {{
                "bugs": [
                    {{
                        "type": "syntax|logic|edge_case|performance",
                        "description": "Brief description",
                        "line_hint": "approximate line or section",
                        "severity": "critical|major|minor",
                        "fix_hint": "What to look for"
                    }}
                ],
                "has_critical_bugs": true|false
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.detected_bugs = result.get("bugs", [])
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Bug detection error: {e}")
            return state
    
    async def _analyze_approach(self, state: LiveCodingState) -> LiveCodingState:
        """Analyze the candidate's problem-solving approach"""
        try:
            if not state.current_code or len(state.current_code) < 50:
                return state
            
            prompt = f"""
            Analyze the problem-solving approach in this code.
            
            PROBLEM: {state.problem_statement[:500]}
            
            EXPECTED APPROACHES:
            {json.dumps(state.expected_approach)}
            
            CODE:
            ```{state.language}
            {state.current_code}
            ```
            
            PHASE HISTORY:
            {json.dumps(state.phase_timeline[-5:])}
            
            Analyze:
            1. What approach is the candidate taking?
            2. Is it one of the expected approaches?
            3. Is the approach likely to work?
            4. Are they breaking down the problem well?
            5. Using helper functions appropriately?
            
            Return JSON:
            {{
                "detected_approach": "Description of their approach",
                "matches_expected": true|false,
                "approach_viability": "will_work|may_work|unlikely_to_work",
                "breaks_down_problem": true|false,
                "uses_helper_functions": true|false,
                "approach_score": 0.75,
                "observations": ["observation1", "observation2"]
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.approach_analysis = result
            state.breaks_down_problem = result.get("breaks_down_problem", False)
            state.uses_helper_functions = result.get("uses_helper_functions", False)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Approach analysis error: {e}")
            return state
    
    async def _assess_quality(self, state: LiveCodingState) -> LiveCodingState:
        """Assess code quality"""
        try:
            if not state.current_code or len(state.current_code) < 50:
                return state
            
            prompt = f"""
            Assess the quality of this code.
            
            LANGUAGE: {state.language}
            
            CODE:
            ```{state.language}
            {state.current_code}
            ```
            
            Evaluate:
            1. Readability (naming, structure, comments)
            2. Correctness (based on visible logic)
            3. Efficiency (algorithm choice, complexity)
            4. Best practices for {state.language}
            5. Edge case handling
            
            Return JSON:
            {{
                "readability_score": 0.8,
                "correctness_score": 0.75,
                "efficiency_score": 0.7,
                "best_practices_score": 0.65,
                "edge_case_score": 0.5,
                "overall_quality": "excellent|good|acceptable|needs_improvement|poor",
                "strengths": ["strength1"],
                "improvements_needed": ["improvement1"]
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.code_quality = result
            
            # Calculate scores
            state.code_quality_score = (
                result.get("readability_score", 0.5) * 0.2 +
                result.get("correctness_score", 0.5) * 0.3 +
                result.get("efficiency_score", 0.5) * 0.2 +
                result.get("best_practices_score", 0.5) * 0.15 +
                result.get("edge_case_score", 0.5) * 0.15
            )
            
            state.efficiency_score = result.get("efficiency_score", 0.5)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Quality assessment error: {e}")
            return state
    
    async def _check_intervention_needed(self, state: LiveCodingState) -> LiveCodingState:
        """Check if candidate needs a hint or intervention"""
        try:
            # Check if stuck
            if state.current_phase == CodingPhase.STUCK.value:
                if state.stuck_duration_seconds > 120:  # 2 minutes stuck
                    state.should_provide_hint = True
                    state.hint_level = min(3, state.hint_level + 1)
            
            # Check time pressure
            time_used_ratio = state.time_elapsed_seconds / (state.time_limit_minutes * 60)
            
            # At 80% time with no working solution
            if time_used_ratio > 0.8:
                has_critical_bugs = any(
                    bug.get("severity") == "critical" 
                    for bug in state.detected_bugs
                )
                if has_critical_bugs:
                    state.should_provide_hint = True
                    state.hint_level = 2
            
            # Very high delete ratio indicates struggle
            if state.delete_ratio > 2.0 and state.time_elapsed_seconds > 600:
                state.should_provide_hint = True
                state.hint_level = 1
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Intervention check error: {e}")
            return state
    
    async def _generate_insights(self, state: LiveCodingState) -> LiveCodingState:
        """Generate final insights for interviewer"""
        try:
            # Calculate problem-solving score
            approach_score = state.approach_analysis.get("approach_score", 0.5)
            methodology_bonus = 0.1 if state.breaks_down_problem else 0
            helper_bonus = 0.05 if state.uses_helper_functions else 0
            testing_bonus = 0.1 if state.uses_tests else 0
            
            state.problem_solving_score = min(1.0, approach_score + methodology_bonus + helper_bonus + testing_bonus)
            
            # Generate overall assessment
            prompt = f"""
            Generate a brief assessment of this coding interview session.
            
            PROBLEM: {state.problem_statement[:300]}
            
            METRICS:
            - Time Used: {state.time_elapsed_seconds}s of {state.time_limit_minutes * 60}s
            - Problem-Solving Score: {state.problem_solving_score:.2f}
            - Code Quality Score: {state.code_quality_score:.2f}
            - Efficiency Score: {state.efficiency_score:.2f}
            - Delete Ratio: {state.delete_ratio:.2f}
            - Refactor Count: {state.refactor_count}
            
            APPROACH: {json.dumps(state.approach_analysis)}
            
            BUGS FOUND: {len(state.detected_bugs)}
            
            PHASE HISTORY: {json.dumps(state.phase_timeline[-5:])}
            
            Generate a 2-3 sentence assessment for the interviewer.
            
            Return JSON:
            {{
                "assessment": "Brief overall assessment",
                "recommendation": "proceed|discuss|concern",
                "key_strengths": ["strength1"],
                "areas_to_explore": ["topic to ask about"],
                "follow_up_questions": ["question1"]
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            state.overall_assessment = result.get("assessment", "")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Insight generation error: {e}")
            state.overall_assessment = f"Coding session analyzed. Quality: {state.code_quality_score:.0%}"
            return state
    
    async def start_session(
        self,
        session_id: str,
        candidate_id: str,
        problem_id: str,
        problem_statement: str,
        expected_approaches: List[str],
        difficulty: str = "medium",
        time_limit_minutes: int = 45,
        language: str = "python"
    ) -> Dict[str, Any]:
        """Start a new coding observation session"""
        try:
            state = LiveCodingState(
                session_id=session_id,
                candidate_id=candidate_id,
                problem_id=problem_id,
                problem_statement=problem_statement,
                expected_approach=expected_approaches,
                difficulty=difficulty,
                time_limit_minutes=time_limit_minutes,
                language=language
            )
            
            self.sessions[session_id] = state
            
            return {
                "success": True,
                "session_id": session_id,
                "message": "Coding session started"
            }
            
        except Exception as e:
            logger.error(f"❌ Session start error: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_code(
        self,
        session_id: str,
        code: str,
        time_elapsed_seconds: int
    ) -> Dict[str, Any]:
        """Update code and run analysis"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            
            # Store in history
            state.code_history.append({
                "code": code,
                "timestamp": datetime.now().isoformat(),
                "time_elapsed": time_elapsed_seconds
            })
            
            state.current_code = code
            state.time_elapsed_seconds = time_elapsed_seconds
            
            # Run analysis workflow
            final_state = await self.workflow.ainvoke(state)
            
            # Update stored state
            self.sessions[session_id] = final_state
            
            return {
                "success": True,
                "current_phase": final_state.current_phase,
                "detected_bugs": final_state.detected_bugs,
                "approach_analysis": final_state.approach_analysis,
                "code_quality": final_state.code_quality,
                "should_provide_hint": final_state.should_provide_hint,
                "hint_level": final_state.hint_level,
                "problem_solving_score": final_state.problem_solving_score,
                "code_quality_score": final_state.code_quality_score
            }
            
        except Exception as e:
            logger.error(f"❌ Code update error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_hint(
        self,
        session_id: str,
        hint_level: int = 1
    ) -> Dict[str, Any]:
        """Generate a hint for the candidate"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            
            hints = {
                1: "Think about the edge cases. What happens with empty input?",
                2: f"Consider using a {state.expected_approach[0] if state.expected_approach else 'different approach'}",
                3: f"The optimal solution typically uses {state.expected_approach[0] if state.expected_approach else 'a specific technique'}"
            }
            
            prompt = f"""
            Generate a helpful hint for a candidate stuck on this problem.
            
            PROBLEM: {state.problem_statement}
            EXPECTED APPROACHES: {state.expected_approach}
            
            CURRENT CODE:
            ```{state.language}
            {state.current_code[:500]}
            ```
            
            HINT LEVEL: {hint_level} (1=subtle, 2=moderate, 3=direct)
            
            Generate a hint appropriate for level {hint_level}.
            - Level 1: Point them in the right direction without giving away the solution
            - Level 2: Mention the technique or data structure to consider
            - Level 3: Give a more direct hint about the approach
            
            Return JSON:
            {{
                "hint": "The hint text",
                "concept_to_consider": "A concept they should think about"
            }}
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            result = json.loads(response.content.replace("```json", "").replace("```", ""))
            
            return {
                "success": True,
                "hint": result.get("hint", hints.get(hint_level, "Think about the problem differently")),
                "hint_level": hint_level
            }
            
        except Exception as e:
            logger.error(f"❌ Hint generation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def end_session(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """End session and get final report"""
        try:
            if session_id not in self.sessions:
                return {"success": False, "error": "Session not found"}
            
            state = self.sessions[session_id]
            
            # Run final analysis
            final_state = await self.workflow.ainvoke(state)
            
            # Generate comprehensive report
            report = {
                "success": True,
                "session_id": session_id,
                "candidate_id": state.candidate_id,
                "problem_id": state.problem_id,
                "time_used_seconds": state.time_elapsed_seconds,
                "time_limit_seconds": state.time_limit_minutes * 60,
                "completion_ratio": state.time_elapsed_seconds / (state.time_limit_minutes * 60),
                "scores": {
                    "problem_solving": final_state.problem_solving_score,
                    "code_quality": final_state.code_quality_score,
                    "efficiency": final_state.efficiency_score
                },
                "approach": final_state.approach_analysis,
                "code_quality_details": final_state.code_quality,
                "bugs_detected": final_state.detected_bugs,
                "behavioral_metrics": {
                    "delete_ratio": final_state.delete_ratio,
                    "refactor_count": final_state.refactor_count,
                    "uses_tests": final_state.uses_tests,
                    "breaks_down_problem": final_state.breaks_down_problem,
                    "uses_helper_functions": final_state.uses_helper_functions
                },
                "phase_timeline": final_state.phase_timeline,
                "overall_assessment": final_state.overall_assessment,
                "final_code": state.current_code
            }
            
            # Clean up session
            del self.sessions[session_id]
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Session end error: {e}")
            return {"success": False, "error": str(e)}


# Singleton instance
_coding_observer = None

def get_coding_observer() -> LiveCodingObserver:
    """Get singleton coding observer instance"""
    global _coding_observer
    if _coding_observer is None:
        _coding_observer = LiveCodingObserver()
    return _coding_observer
