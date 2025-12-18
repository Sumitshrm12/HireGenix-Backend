"""
Context Service - The "Brain" of the Human-like Agentic AI
Aggregates Candidate, Job, Company, and Market context into a unified knowledge graph.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Local imports
from resume_parser import ResumeParser
from crawl4ai_service import AgenticCompanyIntelligenceOrchestrator, AgenticCompanyResearchRequest, CrawlStrategy
from market_intelligence import get_market_intelligence_service

# LangChain / AI imports
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

@dataclass
class InterviewContext:
    """Unified context object for the interview agent"""
    candidate_profile: Dict[str, Any]
    job_requirements: Dict[str, Any]
    company_intelligence: Dict[str, Any]
    market_trends: List[str]
    generated_at: str

class ContextService:
    """
    Service to build the 'Context Engine' for human-like interviews.
    """
    
    def __init__(self):
        self.resume_parser = ResumeParser()
        self.company_researcher = AgenticCompanyIntelligenceOrchestrator()
        self.market_service = get_market_intelligence_service()
        
        # Initialize Azure OpenAI for synthesis
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=0.7
        )

    async def build_interview_context(
        self,
        candidate_id: str,
        job_title: str,
        job_description: str,
        company_name: str,
        resume_file_bytes: Optional[bytes] = None,
        resume_filename: Optional[str] = None,
        company_website: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive context object for the interview session.
        """
        print(f"Building Interview Context for {candidate_id} @ {company_name}")
        
        # 1. Parse Resume (Candidate Context)
        candidate_profile = {}
        if resume_file_bytes and resume_filename:
            try:
                print("Parsing Resume...")
                candidate_profile = self.resume_parser.parse(resume_file_bytes, resume_filename)
                # Enhance with AI extraction if needed (simplified here)
            except Exception as e:
                print(f"Resume parsing failed: {e}")
                candidate_profile = {"error": "Failed to parse resume"}

        # 2. Fetch Company Intelligence (Company Context)
        company_context = {}
        try:
            print(f"Fetching Company Intelligence for {company_name}...")
            # Check cache or DB first (omitted for brevity)
            
            research_request = AgenticCompanyResearchRequest(
                company_name=company_name,
                website=company_website,
                research_depth=CrawlStrategy.FAST, # Fast for real-time context
                max_sources=5,
                include_competitors=False,
                include_news=True,
                include_social=False,
                parallel_agents=3
            )
            
            research_result = await self.company_researcher.orchestrate_research(research_request)
            if research_result.success:
                company_context = research_result.intelligence_data
            else:
                print("Company research returned unsuccessful status")
                
        except Exception as e:
            print(f"Company research failed: {e}")
            company_context = {"name": company_name, "note": "Intelligence unavailable"}

        # 3. Fetch Market Trends (Market Context)
        # Uses Real-Time Market Intelligence Service
        tech_stack = company_context.get('technology', {}).get('tech_stack', [])
        market_data = await self.market_service.fetch_market_trends(job_title, tech_stack)
        
        # Convert structured data to list of strings for agent compatibility
        market_trends = []
        if "emerging_technologies" in market_data:
            market_trends.append(f"Emerging Tech: {', '.join(market_data['emerging_technologies'])}")
        if "salary_benchmark" in market_data:
            market_trends.append(f"Salary Trends: {market_data['salary_benchmark']}")
        if "key_challenges" in market_data:
            market_trends.extend([f"Challenge: {c}" for c in market_data['key_challenges']])
        if "interview_topics" in market_data:
            market_trends.extend([f"Hot Topic: {t}" for t in market_data['interview_topics']])

        # 4. Synthesize Job Requirements
        job_reqs = {
            "title": job_title,
            "description": job_description,
            "key_skills": self._extract_skills_from_jd(job_description)
        }

        # Construct final context
        context = InterviewContext(
            candidate_profile=candidate_profile,
            job_requirements=job_reqs,
            company_intelligence=company_context,
            market_trends=market_trends,
            generated_at=datetime.now().isoformat()
        )
        
        return context.__dict__

    # _fetch_market_trends removed as it is replaced by MarketIntelligenceService

    def _extract_skills_from_jd(self, jd: str) -> List[str]:
        """Simple heuristic extraction, ideally replaced by AI extraction"""
        # Placeholder for actual extraction logic
        common_skills = ["Python", "JavaScript", "React", "Node.js", "SQL", "AWS", "Docker", "Kubernetes", "Java", "C++"]
        found_skills = [skill for skill in common_skills if skill.lower() in jd.lower()]
        return found_skills if found_skills else ["General Technical Skills"]

# Singleton
_context_service = None

def get_context_service() -> ContextService:
    global _context_service
    if _context_service is None:
        _context_service = ContextService()
    return _context_service
