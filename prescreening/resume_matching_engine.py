# prescreening/resume_matching_engine.py - Enhanced Resume Matching for Pre-screening
import os
import re
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import httpx

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# Local imports
from .models import ResumeMatchingResult, ScoreBucket
from .token_usage import get_token_tracker, track_openai_usage
from .config import settings

class KeywordAnalyzer:
    """Analyzes keyword matches between resume and job requirements"""
    
    def __init__(self):
        # Technical skills patterns
        self.tech_patterns = {
            'languages': r'\b(python|javascript|java|c\+\+|c#|php|ruby|go|rust|swift|kotlin|typescript|scala|r|matlab|sql)\b',
            'frameworks': r'\b(react|angular|vue|django|flask|express|spring|rails|laravel|nextjs|nodejs|dotnet)\b',
            'databases': r'\b(mysql|postgresql|mongodb|redis|elasticsearch|cassandra|oracle|sqlite|dynamodb)\b',
            'cloud': r'\b(aws|azure|gcp|docker|kubernetes|terraform|jenkins|gitlab|github actions)\b',
            'tools': r'\b(git|jira|confluence|slack|figma|photoshop|illustrator|sketch|postman|swagger)\b'
        }
        
        # Soft skills patterns
        self.soft_skills_patterns = r'\b(leadership|communication|teamwork|problem.solving|analytical|creative|organized|detail.oriented|adaptable|collaborative)\b'
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical and soft skills from text"""
        text_lower = text.lower()
        extracted_skills = {}
        
        # Extract technical skills by category
        for category, pattern in self.tech_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            extracted_skills[category] = list(set(matches))
        
        # Extract soft skills
        soft_matches = re.findall(self.soft_skills_patterns, text_lower, re.IGNORECASE)
        extracted_skills['soft_skills'] = [skill.replace('.', ' ') for skill in set(soft_matches)]
        
        return extracted_skills
    
    def calculate_keyword_score(self, resume_text: str, job_requirements: List[str]) -> Dict[str, Any]:
        """Calculate keyword matching score between resume and job"""
        resume_skills = self.extract_skills(resume_text)
        job_skills = self.extract_skills(" ".join(job_requirements))
        
        # Flatten extracted skills for comparison
        resume_all_skills = []
        job_all_skills = []
        
        for category_skills in resume_skills.values():
            resume_all_skills.extend(category_skills)
        
        for category_skills in job_skills.values():
            job_all_skills.extend(category_skills)
        
        # Calculate matches
        matched_skills = set(resume_all_skills) & set(job_all_skills)
        missing_skills = set(job_all_skills) - set(resume_all_skills)
        
        # Calculate score (0-100)
        if not job_all_skills:
            score = 50  # Neutral score if no specific skills mentioned
        else:
            score = (len(matched_skills) / len(job_all_skills)) * 100
        
        return {
            'score': min(100, score),
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'resume_skills': resume_skills,
            'job_skills': job_skills
        }

class ExperienceMatcher:
    """Matches experience level between resume and job requirements"""
    
    def __init__(self):
        self.experience_keywords = {
            'junior': ['junior', 'entry-level', 'graduate', 'fresher', '0-2 years'],
            'mid': ['mid-level', 'intermediate', '2-5 years', '3-6 years'],
            'senior': ['senior', 'experienced', '5+ years', '7+ years', 'lead'],
            'expert': ['expert', 'principal', 'architect', '10+ years', 'director']
        }
    
    def extract_years_of_experience(self, text: str) -> int:
        """Extract years of experience from text"""
        # Look for patterns like "5 years", "3+ years", "2-4 years"
        year_patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'(\d+)\+?\s*years?\s*experience',
            r'(\d+)\s*\+\s*years',
            r'(\d+)-(\d+)\s*years'
        ]
        
        max_years = 0
        text_lower = text.lower()
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle range patterns like "2-4 years"
                    years = max([int(x) for x in match if x.isdigit()])
                else:
                    years = int(match)
                max_years = max(max_years, years)
        
        return max_years
    
    def calculate_experience_score(self, resume_text: str, job_requirements: str) -> Dict[str, Any]:
        """Calculate experience matching score"""
        resume_years = self.extract_years_of_experience(resume_text)
        required_years = self.extract_years_of_experience(job_requirements)
        
        if required_years == 0:
            # If no specific experience mentioned, give neutral score
            score = 70
        elif resume_years >= required_years:
            # Meets or exceeds requirement
            score = 100
        elif resume_years >= (required_years * 0.7):
            # Close to requirement (70% match)
            score = 80
        else:
            # Below requirement
            score = max(20, (resume_years / required_years) * 100) if required_years > 0 else 50
        
        return {
            'score': min(100, score),
            'resume_years': resume_years,
            'required_years': required_years,
            'meets_requirement': resume_years >= required_years
        }

class HybridResumeMatchingEngine:
    """Main engine that combines embedding, keyword, and experience matching"""
    
    def __init__(self):
        self.embeddings_model = AzureOpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("TEXT_EMBEDDING_ENDPOINT"),
            deployment=os.getenv("TEXT_EMBEDDING_MODEL"),
            openai_api_version=os.getenv("TEXT_EMBEDDING_API_VERSION")
        )
        self.keyword_analyzer = KeywordAnalyzer()
        self.experience_matcher = ExperienceMatcher()
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-5.2.chat",
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.0
        )
    
    async def compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using embeddings"""
        try:
            embeddings = await asyncio.to_thread(
                self.embeddings_model.embed_documents, [text1, text2]
            )
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return max(0, min(100, similarity * 100))  # Scale to 0-100
            
        except Exception as e:
            print(f"Error computing embedding similarity: {e}")
            return 50  # Return neutral score on error
    
    def compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Compute TF-IDF based similarity as fallback"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return max(0, min(100, similarity * 100))
        except Exception as e:
            print(f"Error computing TF-IDF similarity: {e}")
            return 50
    
    async def generate_ai_rationale(
        self, 
        overall_score: float,
        embedding_score: float,
        keyword_score: float,
        experience_score: float,
        matched_keywords: List[str],
        missing_keywords: List[str]
    ) -> str:
        """Generate AI explanation for the matching score"""
        
        token_tracker = get_token_tracker()
        
        prompt = f"""
        Analyze this resume-job matching result and provide a concise rationale (2-3 sentences max):
        
        Overall Score: {overall_score:.1f}%
        - Semantic Similarity: {embedding_score:.1f}%
        - Keyword Match: {keyword_score:.1f}%
        - Experience Match: {experience_score:.1f}%
        
        Matched Skills: {', '.join(matched_keywords[:10]) if matched_keywords else 'None'}
        Missing Skills: {', '.join(missing_keywords[:5]) if missing_keywords else 'None'}
        
        Provide a brief explanation focusing on the strongest and weakest areas.
        """
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"Error generating AI rationale: {e}")
            return f"Score based on {overall_score:.1f}% overall match considering skills alignment and experience level."
    
    def classify_score_bucket(self, score: float) -> ScoreBucket:
        """Classify score into predefined buckets"""
        if score >= 80:
            return ScoreBucket.EXCELLENT
        elif score >= 70:
            return ScoreBucket.GOOD
        elif score >= 60:
            return ScoreBucket.POTENTIAL
        else:
            return ScoreBucket.NOT_ELIGIBLE
    
    async def analyze_resume_match(
        self,
        candidate_id: str,
        job_id: str,
        resume_text: str,
        job_description: str,
        job_requirements: Optional[List[str]] = None
    ) -> ResumeMatchingResult:
        """
        Main method to analyze resume-job match using hybrid approach
        """
        start_time = time.time()
        
        try:
            # Prepare job requirements
            if not job_requirements:
                job_requirements = [job_description]
            
            job_text = f"{job_description}\n" + "\n".join(job_requirements)
            
            # 1. Compute embedding similarity (60% weight)
            embedding_score = await self.compute_embedding_similarity(resume_text, job_text)
            
            # 2. Compute keyword matching (25% weight)
            keyword_result = self.keyword_analyzer.calculate_keyword_score(resume_text, job_requirements)
            keyword_score = keyword_result['score']
            
            # 3. Compute experience matching (15% weight)
            experience_result = self.experience_matcher.calculate_experience_score(resume_text, job_text)
            experience_score = experience_result['score']
            
            # 4. Calculate final hybrid score
            final_score = (
                0.6 * embedding_score +
                0.25 * keyword_score +
                0.15 * experience_score
            )
            
            # 5. Classify bucket
            bucket = self.classify_score_bucket(final_score)
            
            # 6. Generate AI rationale
            rationale = await self.generate_ai_rationale(
                final_score,
                embedding_score,
                keyword_score,
                experience_score,
                keyword_result['matched_skills'],
                keyword_result['missing_skills']
            )
            
            processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            
            return ResumeMatchingResult(
                id=f"rm_{int(time.time())}_{candidate_id[:8]}",
                candidate_id=candidate_id,
                job_id=job_id,
                overall_score=round(final_score, 2),
                embedding_score=round(embedding_score, 2),
                keyword_score=round(keyword_score, 2),
                experience_score=round(experience_score, 2),
                matched_keywords=keyword_result['matched_skills'],
                missing_keywords=keyword_result['missing_skills'],
                score_rationale=rationale,
                bucket=bucket,
                processing_time_ms=processing_time,
                created_at=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in resume matching analysis: {e}")
            # Return error result
            return ResumeMatchingResult(
                id=f"rm_error_{int(time.time())}",
                candidate_id=candidate_id,
                job_id=job_id,
                overall_score=0.0,
                embedding_score=0.0,
                keyword_score=0.0,
                experience_score=0.0,
                matched_keywords=[],
                missing_keywords=[],
                score_rationale=f"Error in analysis: {str(e)}",
                bucket=ScoreBucket.NOT_ELIGIBLE,
                processing_time_ms=int((time.time() - start_time) * 1000),
                created_at=datetime.now()
            )

class PreScreeningDecisionEngine:
    """Determines next steps based on resume matching score"""
    
    def __init__(self):
        self.thresholds = {
            'excellent': 80.0,
            'good': 70.0,
            'potential': 60.0
        }
    
    def make_decision(self, score: float, bucket: ScoreBucket) -> Dict[str, Any]:
        """Make decision on next steps for candidate"""
        
        if bucket == ScoreBucket.EXCELLENT:
            return {
                'decision': 'auto_proceed',
                'next_step': 'prescreening',
                'requires_human_review': False,
                'priority': 'high',
                'action': 'Schedule pre-screening test automatically',
                'notification_type': 'immediate'
            }
        elif bucket == ScoreBucket.GOOD:
            return {
                'decision': 'human_review_recommended',
                'next_step': 'human_review',
                'requires_human_review': True,
                'priority': 'medium',
                'action': 'Recommend for pre-screening with human confirmation',
                'notification_type': 'reviewer_queue'
            }
        elif bucket == ScoreBucket.POTENTIAL:
            return {
                'decision': 'manual_review_required',
                'next_step': 'human_review',
                'requires_human_review': True,
                'priority': 'low',
                'action': 'Manual review required before proceeding',
                'notification_type': 'reviewer_queue'
            }
        else:  # NOT_ELIGIBLE
            return {
                'decision': 'auto_reject',
                'next_step': 'rejected',
                'requires_human_review': False,
                'priority': 'low',
                'action': 'Automatically rejected - score too low',
                'notification_type': 'none'
            }

# Factory function for easy instantiation
def create_resume_matching_engine() -> HybridResumeMatchingEngine:
    """Create and return a configured resume matching engine instance"""
    return HybridResumeMatchingEngine()

def create_decision_engine() -> PreScreeningDecisionEngine:
    """Create and return a decision engine instance"""
    return PreScreeningDecisionEngine()
