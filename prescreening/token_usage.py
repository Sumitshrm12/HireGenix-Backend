# prescreening/token_usage.py - Token Usage Tracking for Pre-screening
import time
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

class TokenUsageTracker:
    """Track token usage for pre-screening operations"""
    
    def __init__(self):
        self.usage_log = []
        self.current_session = {}
    
    def start_tracking(self, operation: str, context: Dict[str, Any] = None):
        """Start tracking tokens for an operation"""
        session_id = f"{operation}_{int(time.time())}"
        self.current_session[session_id] = {
            "operation": operation,
            "start_time": time.time(),
            "context": context or {},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        return session_id
    
    def add_usage(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4o-mini"
    ):
        """Add token usage to tracking session"""
        if session_id in self.current_session:
            session = self.current_session[session_id]
            session["prompt_tokens"] += prompt_tokens
            session["completion_tokens"] += completion_tokens
            session["total_tokens"] += prompt_tokens + completion_tokens
            session["model"] = model
    
    def end_tracking(self, session_id: str) -> Dict[str, Any]:
        """End tracking session and return usage summary"""
        if session_id not in self.current_session:
            return {}
        
        session = self.current_session[session_id]
        session["end_time"] = time.time()
        session["duration_seconds"] = session["end_time"] - session["start_time"]
        
        # Move to log
        self.usage_log.append(session.copy())
        del self.current_session[session_id]
        
        return session
    
    def get_usage_summary(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Get usage summary for operations"""
        relevant_logs = self.usage_log
        if operation_type:
            relevant_logs = [log for log in self.usage_log if log["operation"] == operation_type]
        
        if not relevant_logs:
            return {
                "total_operations": 0,
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "average_tokens_per_operation": 0,
                "total_duration": 0,
                "average_duration": 0
            }
        
        total_tokens = sum(log["total_tokens"] for log in relevant_logs)
        total_prompt = sum(log["prompt_tokens"] for log in relevant_logs)
        total_completion = sum(log["completion_tokens"] for log in relevant_logs)
        total_duration = sum(log["duration_seconds"] for log in relevant_logs)
        
        return {
            "total_operations": len(relevant_logs),
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "average_tokens_per_operation": total_tokens / len(relevant_logs),
            "total_duration": total_duration,
            "average_duration": total_duration / len(relevant_logs)
        }

# Global tracker instance
_global_tracker = TokenUsageTracker()

def get_token_tracker() -> TokenUsageTracker:
    """Get global token tracker instance"""
    return _global_tracker

class TokenTrackingContext:
    """Context manager for token tracking"""
    
    def __init__(self, operation: str, context: Dict[str, Any] = None):
        self.operation = operation
        self.context = context or {}
        self.session_id = None
        self.tracker = get_token_tracker()
    
    def __enter__(self):
        self.session_id = self.tracker.start_tracking(self.operation, self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session_id:
            return self.tracker.end_tracking(self.session_id)
    
    def add_usage(self, prompt_tokens: int, completion_tokens: int, model: str = "gpt-4o-mini"):
        """Add token usage to this context"""
        if self.session_id:
            self.tracker.add_usage(self.session_id, prompt_tokens, completion_tokens, model)

async def track_openai_usage(
    operation: str,
    openai_call,
    context: Dict[str, Any] = None
) -> Any:
    """
    Async context manager for tracking OpenAI API usage
    """
    tracker = get_token_tracker()
    session_id = tracker.start_tracking(operation, context)
    
    try:
        # Execute the OpenAI call
        result = await openai_call
        
        # Extract usage information if available
        if hasattr(result, 'usage') and result.usage:
            tracker.add_usage(
                session_id,
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
                getattr(result, 'model', 'gpt-4o-mini')
            )
        
        return result
        
    except Exception as e:
        # Still track the session even if it failed
        tracker.add_usage(session_id, 0, 0, "error")
        raise
    finally:
        tracker.end_tracking(session_id)

# Usage statistics for pre-screening operations
OPERATION_COSTS = {
    "resume_matching": {
        "avg_prompt_tokens": 1500,
        "avg_completion_tokens": 300,
        "cost_per_1k_prompt": 0.00015,  # GPT-4o-mini
        "cost_per_1k_completion": 0.0006
    },
    "mcq_generation": {
        "avg_prompt_tokens": 800,
        "avg_completion_tokens": 1200,
        "cost_per_1k_prompt": 0.00015,
        "cost_per_1k_completion": 0.0006
    },
    "ai_rationale": {
        "avg_prompt_tokens": 400,
        "avg_completion_tokens": 150,
        "cost_per_1k_prompt": 0.00015,
        "cost_per_1k_completion": 0.0006
    }
}

def calculate_operation_cost(operation: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost for an operation"""
    if operation not in OPERATION_COSTS:
        # Default to GPT-4o-mini pricing
        cost_config = {
            "cost_per_1k_prompt": 0.00015,
            "cost_per_1k_completion": 0.0006
        }
    else:
        cost_config = OPERATION_COSTS[operation]
    
    prompt_cost = (prompt_tokens / 1000) * cost_config["cost_per_1k_prompt"]
    completion_cost = (completion_tokens / 1000) * cost_config["cost_per_1k_completion"]
    
    return prompt_cost + completion_cost