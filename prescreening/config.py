# prescreening/config.py - Configuration for Pre-screening Module
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class PreScreeningConfig:
    """Configuration for pre-screening system"""
    
    # Azure OpenAI Configuration
    azure_openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_deployment_name: str = os.getenv("AZURE_DEPLOYMENT_NAME", "")
    azure_api_version: str = os.getenv("AZURE_API_VERSION", "2024-02-01")
    
    # Text Embedding Configuration  
    text_embedding_endpoint: str = os.getenv("TEXT_EMBEDDING_ENDPOINT", "")
    text_embedding_model: str = os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-ada-002")
    text_embedding_api_version: str = os.getenv("TEXT_EMBEDDING_API_VERSION", "2024-02-01")
    
    # Score Thresholds
    excellent_threshold: float = 80.0
    good_threshold: float = 70.0
    potential_threshold: float = 60.0
    
    # Weights for Hybrid Scoring
    embedding_weight: float = 0.6  # 60%
    keyword_weight: float = 0.25   # 25%
    experience_weight: float = 0.15 # 15%
    
    # MCQ Configuration
    default_mcq_count: int = 10
    mcq_time_limit_minutes: int = 30
    mcq_passing_score: float = 70.0
    
    # Proctoring Configuration
    proctoring_enabled: bool = True
    max_tab_switches: int = 3
    max_face_detection_fails: int = 5
    
    # Processing Limits
    max_resume_length: int = 50000  # characters
    max_job_description_length: int = 10000  # characters
    processing_timeout_seconds: int = 30
    
    def get_score_weights(self) -> Dict[str, float]:
        """Get scoring weights as dictionary"""
        return {
            "embedding": self.embedding_weight,
            "keyword": self.keyword_weight,
            "experience": self.experience_weight
        }
    
    def get_score_thresholds(self) -> Dict[str, float]:
        """Get score thresholds as dictionary"""
        return {
            "excellent": self.excellent_threshold,
            "good": self.good_threshold,
            "potential": self.potential_threshold
        }
    
    def get_mcq_config(self) -> Dict[str, Any]:
        """Get MCQ configuration as dictionary"""
        return {
            "default_count": self.default_mcq_count,
            "time_limit_minutes": self.mcq_time_limit_minutes,
            "passing_score": self.mcq_passing_score
        }
    
    def validate_config(self) -> bool:
        """Validate that required configuration is present"""
        required_fields = [
            "azure_openai_api_key",
            "azure_openai_endpoint", 
            "azure_deployment_name",
            "text_embedding_endpoint",
            "text_embedding_model"
        ]
        
        for field in required_fields:
            if not getattr(self, field):
                print(f"Missing required configuration: {field}")
                return False
        
        # Validate weights sum to 1.0
        total_weight = self.embedding_weight + self.keyword_weight + self.experience_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            print(f"Score weights must sum to 1.0, got {total_weight}")
            return False
        
        return True

# Global configuration instance
settings = PreScreeningConfig()

# Validate configuration on import
if not settings.validate_config():
    print("Warning: Pre-screening configuration validation failed")