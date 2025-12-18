# utils/token_tracker.py
from typing import Dict, Any, Optional
from langchain_core.callbacks import BaseCallbackHandler
from threading import Lock

class TokenTracker:
    """
    Thread-safe singleton class to track token usage across all API calls
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TokenTracker, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self._lock = Lock()
    
    def reset_totals(self):
        """Reset all token counters"""
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_tokens = 0
    
    def update_usage(self, prompt_tokens: int, completion_tokens: int):
        """Update token counts in a thread-safe manner"""
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += (prompt_tokens + completion_tokens)
    
    def get_usage(self) -> Dict[str, Dict[str, int]]:
        """Get current token usage statistics"""
        with self._lock:
            return {
                "cumulative": {
                    "prompt_tokens": self.total_prompt_tokens,
                    "completion_tokens": self.total_completion_tokens,
                    "total_tokens": self.total_tokens
                }
            }

class TokenTrackingCallback(BaseCallbackHandler):
    """
    Callback handler for tracking token usage in individual LLM calls
    """
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.token_tracker = TokenTracker()
    
    def on_llm_end(self, response, **kwargs):
        """Capture token usage when LLM call finishes"""
        if hasattr(response, 'llm_output') and response.llm_output is not None:
            usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens = usage.get('prompt_tokens', 0)
            self.completion_tokens = usage.get('completion_tokens', 0)
            self.total_tokens = usage.get('total_tokens', 0)
            self.token_tracker.update_usage(self.prompt_tokens, self.completion_tokens)
    
    def get_usage(self) -> Dict[str, int]:
        """Get token usage for this specific call"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }

def get_token_tracker() -> TokenTrackingCallback:
    """Factory function to create a new token tracker callback"""
    return TokenTrackingCallback()

def get_token_usage_stats() -> Dict[str, Dict[str, int]]:
    """Get complete token usage statistics"""
    return TokenTracker().get_usage()

def reset_token_tracking() -> None:
    """Reset all token counters"""
    TokenTracker().reset_totals()