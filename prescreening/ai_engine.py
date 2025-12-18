# prescreening/ai_engine.py - AI Video Pre-screening Engine
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import base64
import cv2
import numpy as np
from dataclasses import dataclass

# Azure OpenAI direct import (better for gpt-4.1-mini compatibility)
from openai import AsyncAzureOpenAI

# LangChain imports (kept for compatibility)
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from .models import (
    PreScreeningSession, MCQQuestion, ProctoringEvent,
    PreScreeningStatus, ScoreBucket
)
from .database import get_database
from .token_usage import get_token_tracker, TokenTrackingContext
from .config import settings
import os

@dataclass
class VideoAnalysisResult:
    """Result of video analysis"""
    confidence_score: float
    engagement_score: float
    clarity_score: float
    professionalism_score: float
    red_flags: List[str]
    positive_indicators: List[str]
    transcript: str
    analysis_notes: str

@dataclass
class MCQEvaluation:
    """MCQ evaluation result"""
    question_id: str
    selected_answer: str
    is_correct: bool
    time_taken_seconds: int
    confidence_level: float
    explanation: str

class VideoAnalysisEngine:
    """AI-powered video analysis for pre-screening"""
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-5.2.chat",
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.1  # Low temperature for consistent analysis
        )
    
    async def analyze_video_response(
        self,
        video_base64: str,
        question: MCQQuestion,
        audio_transcript: str
    ) -> VideoAnalysisResult:
        """Analyze candidate's video response to MCQ question"""
        
        with TokenTrackingContext("video_analysis") as tracker:
            try:
                # Create analysis prompt
                analysis_prompt = self._create_video_analysis_prompt(
                    question, audio_transcript
                )
                
                # Send to GPT-4V (if available) or GPT-4 with transcript
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are an expert HR analyst evaluating candidate video responses for technical pre-screening."),
                    HumanMessage(content=analysis_prompt)
                ])
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    tracker.add_usage(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                
                # Parse the response
                analysis_data = self._parse_analysis_response(response.content)
                
                return VideoAnalysisResult(
                    confidence_score=analysis_data.get('confidence_score', 0.0),
                    engagement_score=analysis_data.get('engagement_score', 0.0),
                    clarity_score=analysis_data.get('clarity_score', 0.0),
                    professionalism_score=analysis_data.get('professionalism_score', 0.0),
                    red_flags=analysis_data.get('red_flags', []),
                    positive_indicators=analysis_data.get('positive_indicators', []),
                    transcript=audio_transcript,
                    analysis_notes=analysis_data.get('analysis_notes', '')
                )
                
            except Exception as e:
                print(f"Error in video analysis: {e}")
                # Return default analysis
                return VideoAnalysisResult(
                    confidence_score=50.0,
                    engagement_score=50.0,
                    clarity_score=50.0,
                    professionalism_score=50.0,
                    red_flags=['Analysis failed - technical error'],
                    positive_indicators=[],
                    transcript=audio_transcript,
                    analysis_notes=f'Technical analysis error: {str(e)}'
                )
    
    def _create_video_analysis_prompt(
        self,
        question: MCQQuestion,
        transcript: str
    ) -> str:
        """Create detailed analysis prompt"""
        return f"""
        Analyze this candidate's video response to a technical screening question.
        
        QUESTION:
        {question.question_text}
        
        CORRECT ANSWER: {question.correct_answer}
        OPTIONS: {json.dumps(question.options)}
        
        CANDIDATE TRANSCRIPT:
        "{transcript}"
        
        Evaluate the candidate on these dimensions (score 0-100):
        
        1. CONFIDENCE_SCORE: How confident and self-assured does the candidate sound?
        2. ENGAGEMENT_SCORE: How engaged and enthusiastic is the candidate?
        3. CLARITY_SCORE: How clear and articulate is their communication?
        4. PROFESSIONALISM_SCORE: How professional is their demeanor and language?
        
        Also identify:
        - RED_FLAGS: Any concerning indicators (hesitation, confusion, unprofessional language)
        - POSITIVE_INDICATORS: Strong points (clear reasoning, technical knowledge, good communication)
        - ANALYSIS_NOTES: Overall assessment and recommendations
        
        Respond in JSON format:
        {{
            "confidence_score": <0-100>,
            "engagement_score": <0-100>,
            "clarity_score": <0-100>,
            "professionalism_score": <0-100>,
            "red_flags": ["flag1", "flag2"],
            "positive_indicators": ["positive1", "positive2"],
            "analysis_notes": "Detailed analysis and recommendations"
        }}
        """
    
    def _parse_analysis_response(self, response_content: str) -> Dict[str, Any]:
        """Parse AI analysis response"""
        try:
            # Extract JSON from response
            if '```json' in response_content:
                json_start = response_content.find('```json') + 7
                json_end = response_content.find('```', json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content.strip()
            
            return json.loads(json_content)
            
        except Exception as e:
            print(f"Error parsing analysis response: {e}")
            return {
                'confidence_score': 50.0,
                'engagement_score': 50.0,
                'clarity_score': 50.0,
                'professionalism_score': 50.0,
                'red_flags': ['Failed to parse analysis'],
                'positive_indicators': [],
                'analysis_notes': 'Analysis parsing failed'
            }

class RealTimeMCQGenerator:
    """Generate MCQ questions in real-time based on candidate performance"""
    
    def __init__(self):
        # Use direct Azure OpenAI client for GPT-4.1
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2025-01-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4", "gpt-4.1")
    
    async def generate_adaptive_question(
        self,
        job_requirements: List[str],
        previous_performance: List[MCQEvaluation],
        difficulty_adjustment: str = "maintain"
    ) -> MCQQuestion:
        """Generate next question based on candidate's performance"""
        
        with TokenTrackingContext("adaptive_mcq_generation") as tracker:
            try:
                # Analyze previous performance
                performance_summary = self._analyze_performance(previous_performance)
                
                # Create adaptive prompt
                prompt = self._create_adaptive_prompt(
                    job_requirements,
                    performance_summary,
                    difficulty_adjustment
                )
                
                # Use direct API call with max_completion_tokens for gpt-4.1-mini
                response = await self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": "You are an expert technical interviewer creating adaptive MCQ questions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,  # GPT-4.1 uses max_tokens
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                # Track tokens
                if response.usage:
                    tracker.add_usage(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )
                
                # Parse question
                question_data = self._parse_question_response(response.choices[0].message.content)
                
                return MCQQuestion(
                    id=f"adaptive_mcq_{int(time.time())}",
                    question_text=question_data['question'],
                    options=question_data['options'],
                    correct_answer=question_data['correct_answer'],
                    explanation=question_data['explanation'],
                    category=question_data.get('category', 'Adaptive'),
                    difficulty_level=question_data.get('difficulty', 'intermediate'),
                    created_at=datetime.now()
                )
                
            except Exception as e:
                print(f"Error generating adaptive question: {e}")
                # Return fallback question
                return self._get_fallback_question()
    
    def _analyze_performance(self, evaluations: List[MCQEvaluation]) -> Dict[str, Any]:
        """Analyze candidate's performance pattern"""
        if not evaluations:
            return {
                'accuracy': 0.0,
                'avg_time': 0.0,
                'confidence': 0.0,
                'weak_areas': [],
                'strong_areas': []
            }
        
        correct_count = sum(1 for eval in evaluations if eval.is_correct)
        accuracy = correct_count / len(evaluations) * 100
        
        avg_time = sum(eval.time_taken_seconds for eval in evaluations) / len(evaluations)
        avg_confidence = sum(eval.confidence_level for eval in evaluations) / len(evaluations)
        
        # Analyze patterns
        incorrect_questions = [eval for eval in evaluations if not eval.is_correct]
        slow_questions = [eval for eval in evaluations if eval.time_taken_seconds > avg_time * 1.5]
        
        return {
            'accuracy': accuracy,
            'avg_time': avg_time,
            'confidence': avg_confidence,
            'total_questions': len(evaluations),
            'correct_count': correct_count,
            'needs_difficulty_adjustment': accuracy > 80 or accuracy < 40
        }
    
    def _create_adaptive_prompt(
        self,
        job_requirements: List[str],
        performance: Dict[str, Any],
        difficulty_adjustment: str
    ) -> str:
        """Create prompt for adaptive question generation"""
        
        difficulty_instruction = {
            "increase": "Make the question more challenging to better assess the candidate's limits.",
            "decrease": "Make the question easier to build candidate confidence and assess foundational knowledge.",
            "maintain": "Keep the difficulty level appropriate for the current performance level."
        }.get(difficulty_adjustment, "maintain")
        
        return f"""
        Generate an adaptive MCQ question for technical pre-screening.
        
        JOB REQUIREMENTS: {", ".join(job_requirements[:5])}
        
        CANDIDATE PERFORMANCE SO FAR:
        - Accuracy: {performance.get('accuracy', 0):.1f}%
        - Average time per question: {performance.get('avg_time', 0):.1f} seconds
        - Questions answered: {performance.get('total_questions', 0)}
        
        DIFFICULTY ADJUSTMENT: {difficulty_instruction}
        
        Generate a question that:
        1. Tests relevant skills for the job requirements
        2. Adapts to the candidate's demonstrated ability level
        3. Is appropriate for video response (can be answered verbally)
        4. Has clear, unambiguous options
        
        Respond in JSON format:
        {{
            "question": "Question text here",
            "options": {{
                "A": "Option A text",
                "B": "Option B text", 
                "C": "Option C text",
                "D": "Option D text"
            }},
            "correct_answer": "A|B|C|D",
            "explanation": "Brief explanation of correct answer",
            "category": "Skill category",
            "difficulty": "easy|intermediate|hard"
        }}
        """
    
    def _parse_question_response(self, response_content: str) -> Dict[str, Any]:
        """Parse question generation response"""
        try:
            if '```json' in response_content:
                json_start = response_content.find('```json') + 7
                json_end = response_content.find('```', json_start)
                json_content = response_content[json_start:json_end].strip()
            else:
                json_content = response_content.strip()
            
            return json.loads(json_content)
            
        except Exception as e:
            print(f"Error parsing question response: {e}")
            return self._get_fallback_question_data()
    
    def _get_fallback_question(self) -> MCQQuestion:
        """Get fallback question if generation fails"""
        return MCQQuestion(
            id=f"fallback_mcq_{int(time.time())}",
            question_text="What is the primary purpose of version control systems in software development?",
            options={
                "A": "To compile code faster",
                "B": "To track changes and enable collaboration",
                "C": "To run automated tests",
                "D": "To deploy applications"
            },
            correct_answer="B",
            explanation="Version control systems primarily track code changes and enable team collaboration.",
            category="Software Development",
            difficulty_level="intermediate",
            created_at=datetime.now()
        )
    
    def _get_fallback_question_data(self) -> Dict[str, Any]:
        """Get fallback question data"""
        return {
            "question": "What is the primary purpose of version control systems in software development?",
            "options": {
                "A": "To compile code faster",
                "B": "To track changes and enable collaboration",
                "C": "To run automated tests",
                "D": "To deploy applications"
            },
            "correct_answer": "B",
            "explanation": "Version control systems primarily track code changes and enable team collaboration.",
            "category": "Software Development",
            "difficulty": "intermediate"
        }

class PreScreeningEvaluationPipeline:
    """Complete evaluation pipeline for AI pre-screening"""
    
    def __init__(self):
        self.video_analyzer = VideoAnalysisEngine()
        self.mcq_generator = RealTimeMCQGenerator()
    
    async def evaluate_mcq_response(
        self,
        session_id: str,
        question: MCQQuestion,
        selected_answer: str,
        video_base64: str,
        audio_transcript: str,
        time_taken: int
    ) -> MCQEvaluation:
        """Evaluate a single MCQ response with video analysis"""
        
        try:
            # Basic correctness check
            is_correct = selected_answer.upper() == question.correct_answer.upper()
            
            # Analyze video response
            video_analysis = await self.video_analyzer.analyze_video_response(
                video_base64, question, audio_transcript
            )
            
            # Calculate confidence based on video analysis
            confidence_level = (
                video_analysis.confidence_score * 0.4 +
                video_analysis.clarity_score * 0.3 +
                video_analysis.engagement_score * 0.2 +
                video_analysis.professionalism_score * 0.1
            )
            
            # Create evaluation
            evaluation = MCQEvaluation(
                question_id=question.id,
                selected_answer=selected_answer,
                is_correct=is_correct,
                time_taken_seconds=time_taken,
                confidence_level=confidence_level,
                explanation=f"Video analysis: {video_analysis.analysis_notes}"
            )
            
            # Store evaluation in database
            await self._store_evaluation(session_id, evaluation, video_analysis)
            
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating MCQ response: {e}")
            # Return basic evaluation
            return MCQEvaluation(
                question_id=question.id,
                selected_answer=selected_answer,
                is_correct=selected_answer.upper() == question.correct_answer.upper(),
                time_taken_seconds=time_taken,
                confidence_level=50.0,
                explanation=f"Evaluation error: {str(e)}"
            )
    
    async def generate_next_question(
        self,
        session_id: str,
        job_requirements: List[str],
        previous_evaluations: List[MCQEvaluation]
    ) -> MCQQuestion:
        """Generate the next adaptive question"""
        
        # Determine difficulty adjustment
        if len(previous_evaluations) >= 2:
            recent_accuracy = sum(
                1 for eval in previous_evaluations[-3:] if eval.is_correct
            ) / min(3, len(previous_evaluations))
            
            if recent_accuracy >= 0.8:
                difficulty_adjustment = "increase"
            elif recent_accuracy <= 0.4:
                difficulty_adjustment = "decrease"
            else:
                difficulty_adjustment = "maintain"
        else:
            difficulty_adjustment = "maintain"
        
        # Generate adaptive question
        question = await self.mcq_generator.generate_adaptive_question(
            job_requirements, previous_evaluations, difficulty_adjustment
        )
        
        # Store question in database
        await self._store_question(session_id, question)
        
        return question
    
    async def calculate_final_score(
        self,
        evaluations: List[MCQEvaluation],
        video_analyses: List[VideoAnalysisResult]
    ) -> Dict[str, Any]:
        """Calculate final pre-screening score"""
        
        if not evaluations:
            return {
                'overall_score': 0.0,
                'accuracy_score': 0.0,
                'confidence_score': 0.0,
                'engagement_score': 0.0,
                'professionalism_score': 0.0,
                'bucket': ScoreBucket.NOT_ELIGIBLE.value,
                'recommendations': ['No responses evaluated']
            }
        
        # Calculate component scores
        accuracy_score = (sum(1 for eval in evaluations if eval.is_correct) / len(evaluations)) * 100
        
        avg_confidence = sum(eval.confidence_level for eval in evaluations) / len(evaluations)
        avg_engagement = sum(analysis.engagement_score for analysis in video_analyses) / len(video_analyses) if video_analyses else 50.0
        avg_professionalism = sum(analysis.professionalism_score for analysis in video_analyses) / len(video_analyses) if video_analyses else 50.0
        
        # Weighted final score
        overall_score = (
            accuracy_score * 0.5 +        # 50% accuracy
            avg_confidence * 0.2 +        # 20% confidence
            avg_engagement * 0.2 +        # 20% engagement  
            avg_professionalism * 0.1     # 10% professionalism
        )
        
        # Determine bucket
        if overall_score >= 80:
            bucket = ScoreBucket.EXCELLENT
        elif overall_score >= 70:
            bucket = ScoreBucket.GOOD
        elif overall_score >= 60:
            bucket = ScoreBucket.POTENTIAL
        else:
            bucket = ScoreBucket.NOT_ELIGIBLE
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            accuracy_score, avg_confidence, avg_engagement, avg_professionalism
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'accuracy_score': round(accuracy_score, 2),
            'confidence_score': round(avg_confidence, 2),
            'engagement_score': round(avg_engagement, 2),
            'professionalism_score': round(avg_professionalism, 2),
            'bucket': bucket.value,
            'total_questions': len(evaluations),
            'correct_answers': sum(1 for eval in evaluations if eval.is_correct),
            'recommendations': recommendations
        }
    
    def _generate_recommendations(
        self,
        accuracy: float,
        confidence: float,
        engagement: float,
        professionalism: float
    ) -> List[str]:
        """Generate recommendations based on scores"""
        recommendations = []
        
        if accuracy >= 80:
            recommendations.append("Strong technical knowledge demonstrated")
        elif accuracy < 60:
            recommendations.append("Technical knowledge needs improvement")
        
        if confidence >= 80:
            recommendations.append("Confident and self-assured responses")
        elif confidence < 60:
            recommendations.append("Work on building confidence in responses")
        
        if engagement >= 80:
            recommendations.append("Highly engaged and enthusiastic")
        elif engagement < 60:
            recommendations.append("Could show more engagement and enthusiasm")
        
        if professionalism >= 80:
            recommendations.append("Professional communication style")
        elif professionalism < 60:
            recommendations.append("Professional communication could be improved")
        
        return recommendations if recommendations else ["Overall performance assessed"]
    
    async def _store_evaluation(
        self,
        session_id: str,
        evaluation: MCQEvaluation,
        video_analysis: VideoAnalysisResult
    ):
        """Store evaluation and video analysis in database"""
        try:
            db = await get_database()
            # In a real implementation, you'd have specific tables for these
            # For now, we'll store as JSON in the session or proctoring events
            await db.record_proctoring_event(ProctoringEvent(
                id=f"eval_{int(time.time())}",
                session_id=session_id,
                candidate_id="",  # Would be filled from session
                event_type="MCQ_EVALUATION",
                event_data={
                    "question_id": evaluation.question_id,
                    "selected_answer": evaluation.selected_answer,
                    "is_correct": evaluation.is_correct,
                    "confidence_level": evaluation.confidence_level,
                    "video_analysis": {
                        "confidence_score": video_analysis.confidence_score,
                        "engagement_score": video_analysis.engagement_score,
                        "clarity_score": video_analysis.clarity_score,
                        "professionalism_score": video_analysis.professionalism_score
                    }
                },
                severity="INFO",
                detected_at=datetime.now()
            ))
        except Exception as e:
            print(f"Error storing evaluation: {e}")
    
    async def _store_question(self, session_id: str, question: MCQQuestion):
        """Store generated question"""
        try:
            db = await get_database()
            # Store question generation event
            await db.record_proctoring_event(ProctoringEvent(
                id=f"qgen_{int(time.time())}",
                session_id=session_id,
                candidate_id="",
                event_type="QUESTION_GENERATED",
                event_data={
                    "question_id": question.id,
                    "category": question.category,
                    "difficulty": question.difficulty_level
                },
                severity="INFO",
                detected_at=datetime.now()
            ))
        except Exception as e:
            print(f"Error storing question: {e}")

# Factory functions
def create_video_analysis_engine() -> VideoAnalysisEngine:
    """Create video analysis engine instance"""
    return VideoAnalysisEngine()

def create_mcq_generator() -> RealTimeMCQGenerator:
    """Create MCQ generator instance"""
    return RealTimeMCQGenerator()

def create_evaluation_pipeline() -> PreScreeningEvaluationPipeline:
    """Create complete evaluation pipeline"""
    return PreScreeningEvaluationPipeline()
