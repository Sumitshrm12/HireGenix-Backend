"""
Advanced Agentic AI Company Intelligence Service - v3.1 ENTERPRISE
Multi-Agent Autonomous System using HTTP-based Crawling for Deep Company Research

ARCHITECTURE:
- DiscoveryAgent: Intelligent URL discovery with dynamic prioritization
- CrawlStrategyAgent: Adaptive crawling strategies per source type
- ExtractionAgent: HTTP-based content extraction (no browser required)
- AnalysisAgent: AI-powered data extraction and structuring  
- CrossReferenceAgent: Data validation across multiple sources
- EnrichmentAgent: Additional intelligence gathering via Tavily
- ValidationAgent: Quality scoring and confidence assessment
- OrchestratorAgent: Coordinates all agents with autonomous decision-making

FEATURES:
- Lightweight HTTP-based crawling (no Playwright/browser needed)
- Tavily API integration for JavaScript-heavy sites
- Autonomous multi-step research with self-optimization
- Intelligent URL discovery and prioritization
- Parallel processing with intelligent batching
- Automatic retry with exponential backoff
- Cross-source data correlation and validation
- Real-time confidence scoring
- Comprehensive product/service extraction
- Memory and caching layer

Author: HireGenix AI Team
Version: 3.1.0 ENTERPRISE (No Browser Edition)
Last Updated: July 2025
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import os
import hashlib
import aiohttp
from bs4 import BeautifulSoup
from enum import Enum
from dataclasses import dataclass, field
import logging
import re
import base64
from fastapi.responses import StreamingResponse
from playwright.async_api import async_playwright, Page

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# ============================================================================
# AGENTIC AI CONFIGURATION
# ============================================================================

class AgentRole(str, Enum):
    DISCOVERY = "discovery_agent"
    STRATEGY = "strategy_agent"
    EXTRACTION = "extraction_agent"
    ANALYSIS = "analysis_agent"
    CROSS_REFERENCE = "cross_reference_agent"
    ENRICHMENT = "enrichment_agent"
    VALIDATION = "validation_agent"
    VISION = "vision_agent"
    BROWSER = "browser_agent"
    ORCHESTRATOR = "orchestrator"

class CrawlStrategy(str, Enum):
    FAST = "fast"          # Quick HTTP crawl (10s timeout)
    STANDARD = "standard"  # Standard HTTP crawl (20s timeout)
    DEEP = "deep"          # Deep multi-page crawl (30s timeout)
    INTELLIGENT = "intelligent"  # AI-driven adaptive crawling
    STEALTH = "stealth"    # Use Tavily API for protected sites
    BROWSER = "browser"    # Use Headless Browser (Playwright)

class SourceType(str, Enum):
    OFFICIAL_WEBSITE = "official_website"
    LINKEDIN = "linkedin"
    CRUNCHBASE = "crunchbase"
    GLASSDOOR = "glassdoor"
    NEWS = "news"
    GITHUB = "github"
    BLOG = "blog"
    OTHER = "other"

class DataQuality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXCELLENT = "excellent"

@dataclass
class AgentReport:
    """Report from an individual agent"""
    agent: AgentRole
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

@dataclass
class CrawlTask:
    """A crawl task to be executed"""
    id: str
    url: str
    source_type: SourceType
    priority: int
    strategy: CrawlStrategy
    status: str = "pending"
    retries: int = 0
    max_retries: int = 3
    result: Optional[Dict[str, Any]] = None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CrawlRequest(BaseModel):
    url: HttpUrl
    headless: bool = True
    javascript: bool = True
    timeout: int = 30000
    wait_for: str = "body"
    user_agent: Optional[str] = None

class BatchCrawlRequest(BaseModel):
    urls: List[HttpUrl]
    headless: bool = True
    javascript: bool = True
    timeout: int = 30000
    concurrency: int = 3

class CrawlResponse(BaseModel):
    success: bool
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    markdown: Optional[str] = None
    html: Optional[str] = None
    links: List[str] = []
    images: List[str] = []
    metadata: Dict[str, Any] = {}
    crawled_at: datetime
    duration_ms: int
    error: Optional[str] = None

class AgenticCompanyResearchRequest(BaseModel):
    company_name: str = Field(..., description="Company name to research")
    website: Optional[HttpUrl] = Field(None, description="Company website URL")
    industry: Optional[str] = Field(None, description="Industry sector")
    research_depth: CrawlStrategy = Field(CrawlStrategy.INTELLIGENT, description="Research depth level")
    max_sources: int = Field(30, description="Maximum sources to crawl", ge=5, le=100)
    include_competitors: bool = Field(True, description="Include competitor analysis")
    include_news: bool = Field(True, description="Include recent news")
    include_social: bool = Field(True, description="Include social media analysis")
    use_browser: bool = Field(False, description="Use headless browser for crawling (slower but more accurate)")
    analyze_branding: bool = Field(False, description="Analyze company branding using Vision AI")
    parallel_agents: int = Field(5, description="Number of parallel crawl agents", ge=1, le=10)

class AgenticResearchResponse(BaseModel):
    success: bool
    company_name: str
    research_id: str
    agents_deployed: List[str]
    total_sources_crawled: int
    data_quality: DataQuality
    confidence_score: float
    processing_time_seconds: float
    intelligence_data: Dict[str, Any]
    agent_reports: List[Dict[str, Any]]
    recommendations: List[str]

# ============================================================================
# INTELLIGENT CACHE LAYER
# ============================================================================

class IntelligentCache:
    """Smart caching with TTL and URL normalization"""
    
    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._default_ttl = default_ttl
        
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent caching"""
        url = url.lower().rstrip('/')
        if '?' in url:
            base, params = url.split('?', 1)
            clean_params = []
            for param in params.split('&'):
                if not any(track in param.lower() for track in ['utm_', 'ref=', 'source=']):
                    clean_params.append(param)
            url = f"{base}{'?' + '&'.join(clean_params) if clean_params else ''}"
        return hashlib.md5(url.encode()).hexdigest()
    
    def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached result if valid"""
        key = self._normalize_url(url)
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() < entry["expires_at"]:
                logger.info(f"🎯 Cache hit for URL: {url[:50]}...")
                return entry["data"]
            else:
                del self._cache[key]
        return None
    
    def set(self, url: str, data: Dict[str, Any], ttl: int = None):
        """Cache result with TTL"""
        key = self._normalize_url(url)
        self._cache[key] = {
            "data": data,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=ttl or self._default_ttl)
        }
    
    def invalidate_domain(self, domain: str):
        """Invalidate all cache entries for a domain"""
        keys_to_delete = [k for k, v in self._cache.items() 
                         if domain in v.get("data", {}).get("url", "")]
        for key in keys_to_delete:
            del self._cache[key]

# Global cache instance
_crawl_cache = IntelligentCache(default_ttl=1800)

# ============================================================================
# TAVILY API SERVICE (For JavaScript-heavy sites)
# ============================================================================

class TavilyService:
    """Tavily API integration for web search and content extraction"""
    
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com"
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using Tavily API"""
        if not self.api_key:
            logger.warning("Tavily API key not configured")
            return []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "max_results": max_results,
                        "include_raw_content": True
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("results", [])
                    else:
                        logger.error(f"Tavily API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []
    
    async def extract_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract content from a specific URL using Tavily"""
        if not self.api_key:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/extract",
                    json={
                        "api_key": self.api_key,
                        "urls": [url]
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        return results[0] if results else None
                    return None
        except Exception as e:
            logger.error(f"Tavily extract failed: {e}")
            return None

# Global Tavily service
_tavily_service = TavilyService()

# ============================================================================
# DISCOVERY AGENT
# ============================================================================

class DiscoveryAgent:
    """
    Intelligent URL discovery agent that:
    - Identifies relevant URLs from official website
    - Prioritizes sources by authority and relevance
    - Discovers hidden/nested pages
    - Generates external source URLs
    """
    
    def __init__(self, company_name: str, website: Optional[str]):
        self.company_name = company_name
        self.website = website
        self.discovered_urls: List[CrawlTask] = []
        
    async def discover_sources(
        self,
        include_competitors: bool,
        include_news: bool,
        include_social: bool,
        max_sources: int = 30
    ) -> Tuple[List[CrawlTask], AgentReport]:
        """Discover and prioritize all sources for comprehensive research."""
        start_time = datetime.now()
        report = AgentReport(
            agent=AgentRole.DISCOVERY,
            status="in_progress",
            start_time=start_time
        )
        
        tasks: List[CrawlTask] = []
        task_id = 0
        company_slug = self.company_name.lower().replace(' ', '-').replace(',', '').replace('.', '')
        
        try:
            # PHASE 1: Official Website (Highest Priority)
            if self.website:
                logger.info(f"🎯 Discovery: Official website - {self.website}")
                
                tasks.append(CrawlTask(
                    id=f"task-{task_id}",
                    url=self.website,
                    source_type=SourceType.OFFICIAL_WEBSITE,
                    priority=100,
                    strategy=CrawlStrategy.DEEP
                ))
                task_id += 1
                
                essential_paths = [
                    "/about", "/about-us", "/company", "/who-we-are",
                    "/products", "/services", "/solutions", "/offerings", "/platform",
                    "/technology", "/tech", "/engineering",
                    "/team", "/leadership", "/management",
                    "/careers", "/jobs", "/work-with-us",
                    "/culture", "/values", "/mission",
                    "/news", "/press", "/blog", "/insights",
                    "/contact", "/locations",
                    "/pricing", "/plans",
                    "/customers", "/case-studies", "/testimonials",
                    "/partners", "/integrations",
                    "/features", "/capabilities"
                ]
                
                for path in essential_paths:
                    tasks.append(CrawlTask(
                        id=f"task-{task_id}",
                        url=f"{self.website.rstrip('/')}{path}",
                        source_type=SourceType.OFFICIAL_WEBSITE,
                        priority=90,
                        strategy=CrawlStrategy.STANDARD
                    ))
                    task_id += 1
                    
                report.insights.append(f"Discovered {len(essential_paths) + 1} official website URLs")
            
            # PHASE 2: LinkedIn (Use Tavily for protected sites)
            if include_social:
                linkedin_urls = [
                    f"https://www.linkedin.com/company/{company_slug}",
                ]
                for url in linkedin_urls:
                    tasks.append(CrawlTask(
                        id=f"task-{task_id}",
                        url=url,
                        source_type=SourceType.LINKEDIN,
                        priority=85,
                        strategy=CrawlStrategy.STEALTH  # Use Tavily
                    ))
                    task_id += 1
                report.insights.append(f"Added {len(linkedin_urls)} LinkedIn URLs (via Tavily)")
            
            # PHASE 3: Business Intelligence Sources
            tasks.append(CrawlTask(
                id=f"task-{task_id}",
                url=f"https://www.crunchbase.com/organization/{company_slug}",
                source_type=SourceType.CRUNCHBASE,
                priority=80,
                strategy=CrawlStrategy.STEALTH
            ))
            task_id += 1
            
            # PHASE 4: GitHub (for tech companies)
            github_urls = [
                f"https://github.com/{company_slug}",
            ]
            for url in github_urls:
                tasks.append(CrawlTask(
                    id=f"task-{task_id}",
                    url=url,
                    source_type=SourceType.GITHUB,
                    priority=70,
                    strategy=CrawlStrategy.FAST
                ))
                task_id += 1
            
            # PHASE 5: News Sources (via Tavily search)
            if include_news:
                tasks.append(CrawlTask(
                    id=f"task-{task_id}",
                    url=f"tavily:search:{self.company_name} company news",
                    source_type=SourceType.NEWS,
                    priority=65,
                    strategy=CrawlStrategy.STEALTH
                ))
                task_id += 1
            
            # Sort by priority and limit
            tasks = sorted(tasks, key=lambda x: x.priority, reverse=True)[:max_sources]
            
            report.status = "completed"
            report.end_time = datetime.now()
            report.duration_ms = int((report.end_time - start_time).total_seconds() * 1000)
            report.metrics = {
                "total_urls_discovered": len(tasks),
                "official_website_urls": sum(1 for t in tasks if t.source_type == SourceType.OFFICIAL_WEBSITE),
                "linkedin_urls": sum(1 for t in tasks if t.source_type == SourceType.LINKEDIN),
                "other_sources": sum(1 for t in tasks if t.source_type not in [SourceType.OFFICIAL_WEBSITE, SourceType.LINKEDIN])
            }
            report.insights.append(f"Total: {len(tasks)} URLs ready for crawling")
            
            return tasks, report
            
        except Exception as e:
            logger.error(f"❌ Discovery failed: {e}")
            report.status = "failed"
            report.errors.append(str(e))
            report.end_time = datetime.now()
            return tasks, report

# ============================================================================
# HTTP-BASED EXTRACTION AGENT (No Playwright!)
# ============================================================================

class ExtractionAgent:
    """
    Lightweight HTTP-based content extraction agent:
    - Uses aiohttp for HTTP requests
    - Uses BeautifulSoup for HTML parsing
    - Falls back to Tavily for protected/JS-heavy sites
    - No browser or Playwright required!
    """
    
    def __init__(self):
        self.cache = _crawl_cache
        self.tavily = _tavily_service
        self.retry_delays = [1, 2, 4]
        
        # User agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        ]
    
    def _get_headers(self) -> Dict[str, str]:
        """Get randomized headers for requests"""
        import random
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    def _get_timeout(self, strategy: CrawlStrategy) -> int:
        """Get timeout based on strategy"""
        timeouts = {
            CrawlStrategy.FAST: 10,
            CrawlStrategy.STANDARD: 20,
            CrawlStrategy.DEEP: 30,
            CrawlStrategy.INTELLIGENT: 25,
            CrawlStrategy.STEALTH: 30
        }
        return timeouts.get(strategy, 20)
    
    async def extract_content(
        self,
        task: CrawlTask,
        force_refresh: bool = False
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Extract content from a URL using HTTP or Tavily.
        Returns (result_dict, success_bool)
        """
        # Check cache first
        if not force_refresh:
            cached = self.cache.get(task.url)
            if cached:
                return cached, True
        
        # Handle Tavily search queries
        if task.url.startswith("tavily:search:"):
            return await self._tavily_search(task)
        
        # Use Tavily for protected sites (LinkedIn, Glassdoor, etc.)
        if task.strategy == CrawlStrategy.STEALTH or task.source_type in [SourceType.LINKEDIN, SourceType.CRUNCHBASE, SourceType.GLASSDOOR]:
            return await self._tavily_extract(task)
        
        # Standard HTTP extraction
        return await self._http_extract(task)
    
    async def _http_extract(self, task: CrawlTask) -> Tuple[Dict[str, Any], bool]:
        """Extract content using HTTP requests and BeautifulSoup"""
        timeout = aiohttp.ClientTimeout(total=self._get_timeout(task.strategy))
        
        for retry in range(task.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        task.url,
                        headers=self._get_headers(),
                        allow_redirects=True,
                        ssl=False  # Skip SSL verification for problematic sites
                    ) as response:
                        if response.status == 200:
                            html = await response.text()
                            
                            # Parse with BeautifulSoup
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Extract title
                            title = ""
                            if soup.title:
                                title = soup.title.string or ""
                            
                            # Extract meta description
                            description = ""
                            meta_desc = soup.find("meta", attrs={"name": "description"})
                            if meta_desc:
                                description = meta_desc.get("content", "")
                            
                            # Remove unwanted elements
                            for element in soup(["script", "style", "meta", "link", "noscript", "header", "footer", "nav", "aside"]):
                                element.decompose()
                            
                            # Extract text content
                            content = soup.get_text(separator='\n', strip=True)
                            
                            # Clean up content
                            content = re.sub(r'\n{3,}', '\n\n', content)
                            content = re.sub(r' {2,}', ' ', content)
                            
                            # Extract links
                            links = []
                            for a_tag in soup.find_all('a', href=True)[:50]:
                                href = a_tag['href']
                                if href.startswith('/'):
                                    # Convert relative to absolute URL
                                    from urllib.parse import urljoin
                                    href = urljoin(task.url, href)
                                if href.startswith('http'):
                                    links.append(href)
                            
                            extracted = {
                                "url": task.url,
                                "source_type": task.source_type.value,
                                "priority": task.priority,
                                "success": True,
                                "title": title.strip(),
                                "content": content[:50000],  # Limit content size
                                "content_length": len(content),
                                "links": links,
                                "metadata": {
                                    "description": description,
                                    "status_code": response.status,
                                    "method": "http"
                                },
                                "crawled_at": datetime.now().isoformat(),
                                "strategy_used": task.strategy.value
                            }
                            
                            self.cache.set(task.url, extracted)
                            logger.info(f"✅ HTTP Extracted {len(content)} chars from {task.url[:50]}...")
                            return extracted, True
                        
                        elif response.status == 403 or response.status == 429:
                            # Try Tavily as fallback for blocked requests
                            logger.warning(f"⚠️ HTTP {response.status} for {task.url}, trying Tavily...")
                            return await self._tavily_extract(task)
                        
                        else:
                            if retry < task.max_retries - 1:
                                await asyncio.sleep(self.retry_delays[retry])
                                continue
                            return {
                                "url": task.url,
                                "source_type": task.source_type.value,
                                "success": False,
                                "error": f"HTTP {response.status}"
                            }, False
                            
            except asyncio.TimeoutError:
                if retry < task.max_retries - 1:
                    await asyncio.sleep(self.retry_delays[retry])
                    continue
                return {
                    "url": task.url,
                    "source_type": task.source_type.value,
                    "success": False,
                    "error": "Timeout"
                }, False
                
            except Exception as e:
                logger.error(f"❌ HTTP extraction failed for {task.url}: {e}")
                if retry < task.max_retries - 1:
                    await asyncio.sleep(self.retry_delays[retry])
                    continue
                # Try Tavily as last resort
                return await self._tavily_extract(task)
        
        return {
            "url": task.url,
            "source_type": task.source_type.value,
            "success": False,
            "error": "Max retries exceeded"
        }, False
    
    async def _tavily_extract(self, task: CrawlTask) -> Tuple[Dict[str, Any], bool]:
        """Extract content using Tavily API"""
        try:
            result = await self.tavily.extract_url(task.url)
            
            if result:
                extracted = {
                    "url": task.url,
                    "source_type": task.source_type.value,
                    "priority": task.priority,
                    "success": True,
                    "title": result.get("title", ""),
                    "content": result.get("raw_content", result.get("content", "")),
                    "content_length": len(result.get("raw_content", result.get("content", ""))),
                    "links": [],
                    "metadata": {
                        "method": "tavily",
                        "score": result.get("score", 0)
                    },
                    "crawled_at": datetime.now().isoformat(),
                    "strategy_used": "tavily"
                }
                
                self.cache.set(task.url, extracted)
                logger.info(f"✅ Tavily Extracted content from {task.url[:50]}...")
                return extracted, True
            
            return {
                "url": task.url,
                "source_type": task.source_type.value,
                "success": False,
                "error": "Tavily extraction returned no content"
            }, False
            
        except Exception as e:
            logger.error(f"❌ Tavily extraction failed for {task.url}: {e}")
            return {
                "url": task.url,
                "source_type": task.source_type.value,
                "success": False,
                "error": str(e)
            }, False
    
    async def _tavily_search(self, task: CrawlTask) -> Tuple[Dict[str, Any], bool]:
        """Perform Tavily search and aggregate results"""
        try:
            query = task.url.replace("tavily:search:", "")
            results = await self.tavily.search(query, max_results=5)
            
            if results:
                # Aggregate content from search results
                aggregated_content = []
                for r in results:
                    aggregated_content.append(f"## {r.get('title', 'Unknown')}\n{r.get('content', '')}")
                
                extracted = {
                    "url": task.url,
                    "source_type": task.source_type.value,
                    "priority": task.priority,
                    "success": True,
                    "title": f"Search: {query}",
                    "content": "\n\n".join(aggregated_content),
                    "content_length": sum(len(r.get("content", "")) for r in results),
                    "links": [r.get("url") for r in results if r.get("url")],
                    "metadata": {
                        "method": "tavily_search",
                        "result_count": len(results)
                    },
                    "crawled_at": datetime.now().isoformat(),
                    "strategy_used": "tavily_search"
                }
                
                logger.info(f"✅ Tavily Search found {len(results)} results for: {query[:30]}...")
                return extracted, True
            
            return {
                "url": task.url,
                "source_type": task.source_type.value,
                "success": False,
                "error": "No search results"
            }, False
            
        except Exception as e:
            logger.error(f"❌ Tavily search failed: {e}")
            return {
                "url": task.url,
                "source_type": task.source_type.value,
                "success": False,
                "error": str(e)
            }, False
    
    async def batch_extract(
        self,
        tasks: List[CrawlTask],
        max_parallel: int = 5
    ) -> Tuple[List[Dict[str, Any]], AgentReport]:
        """Extract content from multiple URLs in parallel batches"""
        start_time = datetime.now()
        report = AgentReport(
            agent=AgentRole.EXTRACTION,
            status="in_progress",
            start_time=start_time
        )
        
        results = []
        successful = 0
        failed = 0
        
        try:
            for i in range(0, len(tasks), max_parallel):
                batch = tasks[i:i + max_parallel]
                
                batch_coros = [self.extract_content(task) for task in batch]
                batch_results = await asyncio.gather(*batch_coros, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, tuple):
                        data, success = result
                        results.append(data)
                        if success:
                            successful += 1
                        else:
                            failed += 1
                    elif isinstance(result, Exception):
                        failed += 1
                        results.append({"error": str(result), "success": False})
                
                # Rate limiting between batches
                if i + max_parallel < len(tasks):
                    await asyncio.sleep(0.5)
            
            report.status = "completed"
            report.end_time = datetime.now()
            report.duration_ms = int((report.end_time - start_time).total_seconds() * 1000)
            report.metrics = {
                "total_tasks": len(tasks),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(tasks) if tasks else 0,
                "avg_time_per_url": report.duration_ms / len(tasks) if tasks else 0
            }
            report.insights.append(f"Extracted {successful}/{len(tasks)} URLs successfully (HTTP + Tavily)")
            
            return results, report
            
        except Exception as e:
            logger.error(f"❌ Batch extraction failed: {e}")
            report.status = "failed"
            report.errors.append(str(e))
            return results, report

# ============================================================================
# BROWSER EXTRACTION AGENT (Playwright)
# ============================================================================

class BrowserExtractionAgent:
    """
    Headless Browser Extraction Agent using Playwright:
    - Handles JavaScript-heavy sites (SPAs)
    - Captures screenshots for Vision analysis
    - Simulates human behavior (scrolling, waiting)
    """
    
    def __init__(self):
        self.cache = _crawl_cache
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    async def extract_content(
        self,
        task: CrawlTask,
        capture_screenshot: bool = False
    ) -> Tuple[Dict[str, Any], bool]:
        """Extract content using Playwright"""
        
        # Check cache (skip if screenshot needed as we don't cache those usually to save space)
        if not capture_screenshot:
            cached = self.cache.get(task.url)
            if cached and cached.get("metadata", {}).get("method") == "browser_playwright":
                return cached, True
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=self.user_agent,
                    viewport={'width': 1280, 'height': 800}
                )
                page = await context.new_page()
                
                # Set timeout
                timeout = 30000 if task.strategy != CrawlStrategy.DEEP else 60000
                page.set_default_timeout(timeout)
                
                try:
                    await page.goto(task.url, wait_until="domcontentloaded")
                    
                    # Wait for body
                    await page.wait_for_selector("body")
                    
                    # Scroll to bottom to trigger lazy loading
                    await self._auto_scroll(page)
                    
                    # Extract content
                    title = await page.title()
                    
                    # Playwright has inner_text which is good.
                    text_content = await page.evaluate("document.body.innerText")
                    
                    screenshot_b64 = None
                    if capture_screenshot:
                        screenshot_bytes = await page.screenshot(full_page=False)
                        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                    
                    # Extract links
                    links = await page.evaluate("""
                        Array.from(document.querySelectorAll('a[href]')).map(a => a.href)
                    """)
                    
                    extracted = {
                        "url": task.url,
                        "source_type": task.source_type.value,
                        "priority": task.priority,
                        "success": True,
                        "title": title,
                        "content": text_content[:50000],
                        "links": links[:100],
                        "screenshot": screenshot_b64,
                        "metadata": {
                            "method": "browser_playwright",
                            "strategy": task.strategy.value
                        },
                        "crawled_at": datetime.now().isoformat()
                    }
                    
                    if not capture_screenshot:
                        self.cache.set(task.url, extracted)
                    
                    logger.info(f"✅ Browser Extracted {len(text_content)} chars from {task.url[:50]}...")
                    return extracted, True
                    
                finally:
                    await browser.close()
                    
        except Exception as e:
            logger.error(f"❌ Browser extraction failed for {task.url}: {e}")
            return {
                "url": task.url,
                "source_type": task.source_type.value,
                "success": False,
                "error": str(e)
            }, False

    async def _auto_scroll(self, page: Page):
        await page.evaluate("""
            async () => {
                await new Promise((resolve) => {
                    var totalHeight = 0;
                    var distance = 100;
                    var timer = setInterval(() => {
                        var scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalHeight += distance;

                        if(totalHeight >= scrollHeight - window.innerHeight || totalHeight > 5000){
                            clearInterval(timer);
                            resolve();
                        }
                    }, 100);
                });
            }
        """)

    async def batch_extract(
        self,
        tasks: List[CrawlTask],
        max_parallel: int = 3,
        capture_screenshots: bool = False
    ) -> Tuple[List[Dict[str, Any]], AgentReport]:
        """Extract content from multiple URLs in parallel batches using Browser"""
        start_time = datetime.now()
        report = AgentReport(
            agent=AgentRole.BROWSER,
            status="in_progress",
            start_time=start_time
        )
        
        results = []
        successful = 0
        failed = 0
        
        try:
            # Limit concurrency for browser to avoid resource exhaustion
            browser_concurrency = min(max_parallel, 3)
            
            for i in range(0, len(tasks), browser_concurrency):
                batch = tasks[i:i + browser_concurrency]
                
                batch_coros = [self.extract_content(task, capture_screenshot=capture_screenshots) for task in batch]
                batch_results = await asyncio.gather(*batch_coros, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, tuple):
                        data, success = result
                        results.append(data)
                        if success:
                            successful += 1
                        else:
                            failed += 1
                    elif isinstance(result, Exception):
                        failed += 1
                        results.append({"error": str(result), "success": False})
            
            report.status = "completed"
            report.end_time = datetime.now()
            report.duration_ms = int((report.end_time - start_time).total_seconds() * 1000)
            report.metrics = {
                "total_tasks": len(tasks),
                "successful": successful,
                "failed": failed,
                "screenshots_captured": successful if capture_screenshots else 0
            }
            report.insights.append(f"Browser extracted {successful}/{len(tasks)} URLs")
            
            return results, report
            
        except Exception as e:
            logger.error(f"❌ Browser batch extraction failed: {e}")
            report.status = "failed"
            report.errors.append(str(e))
            return results, report

# ============================================================================
# ANALYSIS AGENT
# ============================================================================

class AnalysisAgent:
    """AI-powered analysis agent for deep content analysis"""
    
    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    
    async def analyze_company_data(
        self,
        extracted_data: List[Dict[str, Any]],
        company_name: str
    ) -> Tuple[Dict[str, Any], AgentReport]:
        """Analyze extracted data and build company intelligence"""
        start_time = datetime.now()
        report = AgentReport(
            agent=AgentRole.ANALYSIS,
            status="in_progress",
            start_time=start_time
        )
        
        try:
            official_data = [d for d in extracted_data if d.get("source_type") == SourceType.OFFICIAL_WEBSITE.value and d.get("success")]
            linkedin_data = [d for d in extracted_data if d.get("source_type") == SourceType.LINKEDIN.value and d.get("success")]
            other_data = [d for d in extracted_data if d.get("success") and d not in official_data and d not in linkedin_data]
            
            context_parts = []
            
            for data in official_data[:10]:
                context_parts.append(f"[OFFICIAL - HIGH TRUST] {data.get('title', '')}\n{data.get('content', '')[:3000]}")
            
            for data in linkedin_data[:3]:
                context_parts.append(f"[LINKEDIN] {data.get('title', '')}\n{data.get('content', '')[:2000]}")
            
            for data in other_data[:5]:
                context_parts.append(f"[{data.get('source_type', 'OTHER')}] {data.get('title', '')}\n{data.get('content', '')[:1500]}")
            
            combined_context = "\n\n---\n\n".join(context_parts)
            
            if not combined_context:
                report.status = "completed"
                report.insights.append("No content to analyze")
                return self._basic_analysis(company_name), report
            
            intelligence = await self._llm_analyze(combined_context, company_name)
            
            report.status = "completed"
            report.end_time = datetime.now()
            report.duration_ms = int((report.end_time - start_time).total_seconds() * 1000)
            report.metrics = {
                "official_sources_used": len(official_data),
                "linkedin_sources_used": len(linkedin_data),
                "other_sources_used": len(other_data),
                "total_context_length": len(combined_context),
                "products_identified": len(intelligence.get("products_services", []))
            }
            report.insights.append(f"Analyzed {len(context_parts)} sources for comprehensive intelligence")
            
            return intelligence, report
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            report.status = "failed"
            report.errors.append(str(e))
            return self._basic_analysis(company_name), report
    
    async def _llm_analyze(self, context: str, company_name: str) -> Dict[str, Any]:
        """Use LLM to extract structured company intelligence"""
        
        if not self.azure_endpoint or not self.azure_key:
            logger.warning("Azure OpenAI not configured, using basic analysis")
            return self._basic_analysis(company_name)
        
        prompt = f"""You are an expert Business Intelligence Analyst. Analyze the following data about {company_name} and extract COMPREHENSIVE structured information.

Context from {company_name} (multiple sources):
{context[:20000]}

Return ONLY valid JSON with this structure:
{{
    "company_overview": {{
        "name": "{company_name}",
        "description": "2-3 sentence comprehensive overview",
        "founded": "year if found",
        "headquarters": "location if found",
        "ceo": "name if found",
        "website": "official URL if found"
    }},
    "business_model": {{
        "type": "B2B|B2C|B2B2C|Platform|Marketplace",
        "revenue_model": "SaaS|Subscription|Transaction|etc",
        "target_customers": ["customer segment 1", "customer segment 2"],
        "market_position": "description of market positioning"
    }},
    "products_services": [
        {{
            "name": "ACTUAL product/service name from content",
            "category": "SaaS|Platform|Service|Hardware|API",
            "description": "Detailed description from content",
            "target_audience": "Who it's for",
            "key_features": ["feature1", "feature2"],
            "differentiators": ["what makes it unique"]
        }}
    ],
    "technology": {{
        "tech_stack": ["technology1", "technology2"],
        "engineering_practices": ["practice1", "practice2"],
        "innovation_focus": ["AI", "ML", "Cloud", etc.]
    }},
    "company_culture": {{
        "mission": "mission statement if found",
        "vision": "vision statement if found",
        "values": ["value1", "value2"],
        "employee_benefits": ["benefit1", "benefit2"]
    }},
    "market_intelligence": {{
        "competitors": ["competitor1", "competitor2"],
        "competitive_advantages": ["advantage1", "advantage2"]
    }},
    "growth_metrics": {{
        "team_size": "number or range if found",
        "funding_stage": "Bootstrap|Seed|Series A/B/C|Public",
        "total_funding": "amount if found"
    }},
    "hiring_insights": {{
        "in_demand_roles": ["role1", "role2"],
        "growth_areas": ["area1", "area2"]
    }}
}}

Return ONLY the JSON, no markdown or explanations."""

        try:
            azure_url = f"{self.azure_endpoint.rstrip('/')}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    azure_url,
                    headers={
                        "Content-Type": "application/json",
                        "api-key": self.azure_key
                    },
                    json={
                        "messages": [
                            {"role": "system", "content": "You are a business intelligence expert. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4000
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content.replace('```json', '').replace('```', '')
                        elif content.startswith('```'):
                            content = content.replace('```', '')
                        
                        return json.loads(content.strip())
                    else:
                        logger.error(f"Azure OpenAI error: {response.status}")
                        return self._basic_analysis(company_name)
                        
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return self._basic_analysis(company_name)
    
    def _basic_analysis(self, company_name: str) -> Dict[str, Any]:
        """Fallback basic analysis structure"""
        return {
            "company_overview": {
                "name": company_name,
                "description": f"{company_name} - Company data being gathered",
                "founded": "Not found",
                "headquarters": "Not found",
                "ceo": "Not found",
                "website": "Not found"
            },
            "business_model": {"type": "Unknown", "revenue_model": "Unknown", "target_customers": [], "market_position": "Under analysis"},
            "products_services": [],
            "technology": {"tech_stack": [], "engineering_practices": [], "innovation_focus": []},
            "company_culture": {"mission": "Not found", "vision": "Not found", "values": [], "employee_benefits": []},
            "market_intelligence": {"competitors": [], "competitive_advantages": []},
            "growth_metrics": {"team_size": "Not found", "funding_stage": "Not found", "total_funding": "Not found"},
            "hiring_insights": {"in_demand_roles": [], "growth_areas": []}
        }

# ============================================================================
# VISION AGENT
# ============================================================================

class VisionAgent:
    """
    AI Vision Agent for Branding & UI/UX Analysis:
    - Analyzes screenshots using GPT-4o Vision
    - Extracts color palettes, typography, and design sentiment
    """
    def __init__(self):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    async def analyze_branding(
        self,
        screenshots: List[str],
        company_name: str
    ) -> Tuple[Dict[str, Any], AgentReport]:
        """Analyze company branding from screenshots"""
        start_time = datetime.now()
        report = AgentReport(
            agent=AgentRole.VISION,
            status="in_progress",
            start_time=start_time
        )
        
        if not screenshots:
            report.status = "skipped"
            report.insights.append("No screenshots to analyze")
            return {}, report
            
        try:
            # Use the first valid screenshot
            screenshot_b64 = screenshots[0]
            
            prompt = f"""You are a Brand Identity Expert. Analyze this screenshot of {company_name}'s website.
            
            Extract the following visual identity elements:
            1. Color Palette (Primary, Secondary, Accent colors with hex codes if possible)
            2. Typography Style (Serif/Sans-serif, Modern/Classic, etc.)
            3. Brand Personality (Professional, Playful, Innovative, Traditional, etc.)
            4. UI/UX Quality Assessment (Modernity, Cleanliness, Accessibility)
            5. Key Visual Elements (Imagery style, Iconography)
            
            Return ONLY valid JSON:
            {{
                "colors": [{{"name": "Primary Blue", "hex": "#0000FF", "usage": "Buttons"}}],
                "typography": {{"style": "Modern Sans-serif", "readability": "High"}},
                "personality": ["Innovative", "Tech-forward"],
                "ui_quality": {{"score": 85, "assessment": "Clean and modern"}},
                "visual_summary": "Brief summary of visual identity"
            }}"""
            
            azure_url = f"{self.azure_endpoint.rstrip('/')}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    azure_url,
                    headers={
                        "Content-Type": "application/json",
                        "api-key": self.azure_key
                    },
                    json={
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{screenshot_b64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 1000
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        
                        # Clean JSON
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content.replace('```json', '').replace('```', '')
                        elif content.startswith('```'):
                            content = content.replace('```', '')
                            
                        branding_data = json.loads(content.strip())
                        
                        report.status = "completed"
                        report.end_time = datetime.now()
                        report.duration_ms = int((report.end_time - start_time).total_seconds() * 1000)
                        report.insights.append("Successfully analyzed brand identity")
                        
                        return branding_data, report
                    else:
                        logger.error(f"Vision API error: {response.status}")
                        raise Exception(f"Vision API error: {response.status}")

        except Exception as e:
            logger.error(f"❌ Vision analysis failed: {e}")
            report.status = "failed"
            report.errors.append(str(e))
            return {}, report

# ============================================================================
# CROSS-REFERENCE AGENT
# ============================================================================

class CrossReferenceAgent:
    """Validates data across multiple sources"""
    
    def cross_reference(
        self,
        intelligence: Dict[str, Any],
        extracted_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], AgentReport]:
        """Cross-reference and validate intelligence data"""
        start_time = datetime.now()
        report = AgentReport(
            agent=AgentRole.CROSS_REFERENCE,
            status="in_progress",
            start_time=start_time
        )
        
        try:
            source_counts = {}
            for data in extracted_data:
                if data.get("success"):
                    source_type = data.get("source_type", "other")
                    source_counts[source_type] = source_counts.get(source_type, 0) + 1
            
            confidence_factors = {
                "official_source_present": 1.0 if source_counts.get(SourceType.OFFICIAL_WEBSITE.value, 0) > 0 else 0.5,
                "multiple_sources": min(1.0, len(extracted_data) / 10),
                "source_diversity": min(1.0, len(source_counts) / 4),
                "products_found": 1.0 if len(intelligence.get("products_services", [])) > 0 else 0.6,
                "tech_stack_found": 1.0 if len(intelligence.get("technology", {}).get("tech_stack", [])) > 0 else 0.7
            }
            
            overall_confidence = sum(confidence_factors.values()) / len(confidence_factors)
            
            intelligence["_cross_reference"] = {
                "source_counts": source_counts,
                "confidence_factors": confidence_factors,
                "overall_confidence": overall_confidence,
                "validated_at": datetime.now().isoformat()
            }
            
            report.status = "completed"
            report.end_time = datetime.now()
            report.duration_ms = int((report.end_time - start_time).total_seconds() * 1000)
            report.metrics = {
                "sources_validated": len(extracted_data),
                "source_types": len(source_counts),
                "confidence_score": overall_confidence
            }
            report.insights.append(f"Cross-referenced {len(extracted_data)} sources with {overall_confidence:.0%} confidence")
            
            return intelligence, report
            
        except Exception as e:
            logger.error(f"❌ Cross-reference failed: {e}")
            report.status = "failed"
            report.errors.append(str(e))
            return intelligence, report

# ============================================================================
# VALIDATION AGENT
# ============================================================================

class ValidationAgent:
    """Validates data quality and completeness"""
    
    def validate(
        self,
        intelligence_data: Dict[str, Any],
        sources_used: List[Dict[str, Any]]
    ) -> Tuple[DataQuality, float, List[str], bool]:
        """Validate intelligence data and provide recommendations."""
        score = 0
        max_score = 100
        recommendations = []
        needs_fallback = False
        
        overview = intelligence_data.get("company_overview", {})
        if overview.get("description") and len(overview.get("description", "")) > 50:
            score += 20
        else:
            recommendations.append("Company description is incomplete")
        
        products = intelligence_data.get("products_services", [])
        if len(products) >= 3:
            score += 30
        elif len(products) >= 1:
            score += 15
            recommendations.append("Limited product/service information found")
        else:
            recommendations.append("❌ CRITICAL: No products/services identified")
            needs_fallback = True
        
        tech = intelligence_data.get("technology", {})
        if len(tech.get("tech_stack", [])) >= 5:
            score += 15
        elif len(tech.get("tech_stack", [])) >= 2:
            score += 8
        else:
            recommendations.append("Technology stack not fully identified")
        
        culture = intelligence_data.get("company_culture", {})
        if len(culture.get("values", [])) >= 3:
            score += 15
        elif len(culture.get("values", [])) >= 1:
            score += 8
        else:
            recommendations.append("Company culture information limited")
        
        market = intelligence_data.get("market_intelligence", {})
        if len(market.get("competitors", [])) >= 2:
            score += 5
        if len(market.get("competitive_advantages", [])) >= 2:
            score += 5
        else:
            recommendations.append("Competitive analysis needs enhancement")
        
        successful_sources = [s for s in sources_used if s.get("success")]
        source_types = set(s.get("source_type") for s in successful_sources)
        if len(source_types) >= 3:
            score += 10
        elif len(source_types) >= 2:
            score += 5
        else:
            recommendations.append("Data from limited source types")
        
        confidence = score / max_score
        if confidence >= 0.85:
            quality = DataQuality.EXCELLENT
        elif confidence >= 0.65:
            quality = DataQuality.HIGH
        elif confidence >= 0.45:
            quality = DataQuality.MEDIUM
        else:
            quality = DataQuality.LOW
            recommendations.insert(0, "⚠️ Low data quality - consider re-running with more sources")
        
        return quality, confidence, recommendations, needs_fallback

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class AgenticCompanyIntelligenceOrchestrator:
    """Main orchestrator that coordinates all agents for comprehensive company research."""
    
    def __init__(self):
        self.discovery_agent = None
        self.extraction_agent = ExtractionAgent()
        self.browser_agent = BrowserExtractionAgent()
        self.vision_agent = VisionAgent()
        self.analysis_agent = AnalysisAgent()
        self.cross_reference_agent = CrossReferenceAgent()
        self.validation_agent = ValidationAgent()
    
    async def orchestrate_research_stream(
        self,
        request: AgenticCompanyResearchRequest
    ):
        """Orchestrate multi-agent company research with streaming updates"""
        start_time = datetime.now()
        research_id = f"company-{request.company_name.lower().replace(' ', '-')}-{int(start_time.timestamp())}"
        
        agent_reports = []
        
        yield json.dumps({
            "type": "start",
            "research_id": research_id,
            "message": f"Starting research for {request.company_name}"
        }) + "\n"
        
        try:
            # PHASE 1: Discovery Agent
            yield json.dumps({
                "type": "agent_start",
                "agent": AgentRole.DISCOVERY.value,
                "message": "Discovering data sources..."
            }) + "\n"
            
            self.discovery_agent = DiscoveryAgent(
                request.company_name,
                str(request.website) if request.website else None
            )
            
            crawl_tasks, discovery_report = await self.discovery_agent.discover_sources(
                include_competitors=request.include_competitors,
                include_news=request.include_news,
                include_social=request.include_social,
                max_sources=request.max_sources
            )
            agent_reports.append(discovery_report)
            
            yield json.dumps({
                "type": "agent_complete",
                "agent": AgentRole.DISCOVERY.value,
                "metrics": discovery_report.metrics,
                "message": f"Found {len(crawl_tasks)} sources"
            }) + "\n"
            
            # PHASE 2: Extraction Agent (HTTP/Browser)
            yield json.dumps({
                "type": "agent_start",
                "agent": AgentRole.EXTRACTION.value,
                "message": f"Extracting content from {len(crawl_tasks)} sources..."
            }) + "\n"
            
            extracted_data = []
            extraction_report = None
            
            if request.use_browser:
                # Use Browser Agent for all tasks if requested
                extracted_data, extraction_report = await self.browser_agent.batch_extract(
                    tasks=crawl_tasks,
                    max_parallel=min(request.parallel_agents, 3), # Limit browser concurrency
                    capture_screenshots=request.analyze_branding
                )
            else:
                # Use HTTP Agent
                extracted_data, extraction_report = await self.extraction_agent.batch_extract(
                    tasks=crawl_tasks,
                    max_parallel=request.parallel_agents
                )
                
            agent_reports.append(extraction_report)
            successful_extractions = [d for d in extracted_data if d.get("success")]
            
            yield json.dumps({
                "type": "agent_complete",
                "agent": AgentRole.EXTRACTION.value,
                "metrics": extraction_report.metrics,
                "message": f"Extracted {len(successful_extractions)} pages"
            }) + "\n"
            
            # PHASE 2.5: Vision Agent (Optional)
            if request.analyze_branding and request.use_browser:
                yield json.dumps({
                    "type": "agent_start",
                    "agent": AgentRole.VISION.value,
                    "message": "Analyzing brand identity..."
                }) + "\n"
                
                # Collect screenshots
                screenshots = [d.get("screenshot") for d in extracted_data if d.get("screenshot")]
                branding_data, vision_report = await self.vision_agent.analyze_branding(
                    screenshots,
                    request.company_name
                )
                agent_reports.append(vision_report)
                
                # Inject branding into extracted data for analysis
                if branding_data:
                    extracted_data.append({
                        "source_type": "branding_analysis",
                        "success": True,
                        "title": "Brand Identity Analysis",
                        "content": json.dumps(branding_data),
                        "metadata": {"type": "vision_analysis"}
                    })
                
                yield json.dumps({
                    "type": "agent_complete",
                    "agent": AgentRole.VISION.value,
                    "message": "Brand analysis complete"
                }) + "\n"
            
            # PHASE 3: Analysis Agent
            yield json.dumps({
                "type": "agent_start",
                "agent": AgentRole.ANALYSIS.value,
                "message": "Synthesizing intelligence..."
            }) + "\n"
            
            intelligence_data, analysis_report = await self.analysis_agent.analyze_company_data(
                extracted_data=extracted_data,
                company_name=request.company_name
            )
            agent_reports.append(analysis_report)
            
            yield json.dumps({
                "type": "agent_complete",
                "agent": AgentRole.ANALYSIS.value,
                "message": "Intelligence synthesis complete"
            }) + "\n"
            
            # PHASE 4: Cross-Reference Agent
            intelligence_data, cross_ref_report = self.cross_reference_agent.cross_reference(
                intelligence=intelligence_data,
                extracted_data=extracted_data
            )
            agent_reports.append(cross_ref_report)
            
            # PHASE 5: Validation Agent
            data_quality, confidence_score, recommendations, needs_fallback = self.validation_agent.validate(
                intelligence_data=intelligence_data,
                sources_used=extracted_data
            )
            
            validation_report = AgentReport(
                agent=AgentRole.VALIDATION,
                status="completed",
                start_time=datetime.now(),
                end_time=datetime.now(),
                metrics={"quality": data_quality.value, "confidence": confidence_score},
                insights=[f"Data quality: {data_quality.value} ({confidence_score:.0%})"]
            )
            agent_reports.append(validation_report)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_response = AgenticResearchResponse(
                success=True,
                company_name=request.company_name,
                research_id=research_id,
                agents_deployed=[r.agent.value if isinstance(r.agent, AgentRole) else str(r.agent) for r in agent_reports],
                total_sources_crawled=len(successful_extractions),
                data_quality=data_quality,
                confidence_score=confidence_score,
                processing_time_seconds=round(processing_time, 2),
                intelligence_data=intelligence_data,
                agent_reports=[
                    {
                        "agent": r.agent.value if isinstance(r.agent, AgentRole) else str(r.agent),
                        "status": r.status,
                        "duration_ms": r.duration_ms,
                        "metrics": r.metrics,
                        "insights": r.insights,
                        "errors": r.errors
                    }
                    for r in agent_reports
                ],
                recommendations=recommendations
            )
            
            yield json.dumps({
                "type": "complete",
                "data": final_response.dict()
            }) + "\n"
            
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {e}")
            yield json.dumps({
                "type": "error",
                "message": str(e)
            }) + "\n"

    async def orchestrate_research(
        self,
        request: AgenticCompanyResearchRequest
    ) -> AgenticResearchResponse:
        """Non-streaming wrapper for backward compatibility"""
        last_response = None
        async for chunk in self.orchestrate_research_stream(request):
            data = json.loads(chunk)
            if data["type"] == "complete":
                last_response = AgenticResearchResponse(**data["data"])
        
        if last_response:
            return last_response
            
        # Fallback if no complete response
        return AgenticResearchResponse(
            success=False,
            company_name=request.company_name,
            research_id="failed",
            agents_deployed=[],
            total_sources_crawled=0,
            data_quality=DataQuality.LOW,
            confidence_score=0,
            processing_time_seconds=0,
            intelligence_data={"error": "Stream failed"},
            agent_reports=[],
            recommendations=[]
        )

# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.post("/agentic/research", response_model=AgenticResearchResponse)
async def agentic_company_research(request: AgenticCompanyResearchRequest):
    """
    **Agentic AI Company Research System v4.0 (Browser + Vision Edition)**
    
    Multi-agent autonomous research system:
    - Headless Browser (Playwright) for complex sites
    - Vision AI for branding analysis
    - HTTP + Tavily for speed and coverage
    - AI-powered data structuring
    """
    try:
        orchestrator = AgenticCompanyIntelligenceOrchestrator()
        result = await orchestrator.orchestrate_research(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic research failed: {str(e)}")

@router.post("/agentic/research/stream")
async def agentic_company_research_stream(request: AgenticCompanyResearchRequest):
    """
    **Streaming Agentic AI Research**
    Returns Server-Sent Events (SSE) with real-time agent updates.
    """
    orchestrator = AgenticCompanyIntelligenceOrchestrator()
    return StreamingResponse(
        orchestrator.orchestrate_research_stream(request),
        media_type="application/x-ndjson"
    )

class AgenticExtractionRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL to extract from")
    company_name: str = Field(..., description="Company name for context")
    strategy: CrawlStrategy = Field(CrawlStrategy.INTELLIGENT, description="Crawling strategy")

class AgenticExtractionResponse(BaseModel):
    success: bool
    url: str
    raw_content: str
    extracted_data: Dict[str, Any]
    confidence: float
    processing_time_seconds: float

@router.post("/agentic/extract", response_model=AgenticExtractionResponse)
async def agentic_extract(request: AgenticExtractionRequest):
    """Agentic AI Single URL Extraction (HTTP + Tavily + Browser Fallback)"""
    try:
        start_time = datetime.now()
        
        task = CrawlTask(
            id="single-extraction",
            url=str(request.url),
            source_type=SourceType.OFFICIAL_WEBSITE,
            priority=100,
            strategy=request.strategy
        )
        
        extraction_agent = ExtractionAgent()
        result, success = await extraction_agent.extract_content(task)
        
        # Fallback to Browser Agent if HTTP/Tavily failed (e.g. 403 Forbidden)
        if not success:
            logger.warning(f"⚠️ Lightweight extraction failed for {request.url}, trying Browser Agent...")
            try:
                browser_agent = BrowserExtractionAgent()
                result, success = await browser_agent.extract_content(task)
            except Exception as e:
                logger.error(f"❌ Browser fallback failed: {e}")
                # Keep original error if browser fails too
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Extraction failed: {result.get('error')}")
        
        analysis_agent = AnalysisAgent()
        intelligence, _ = await analysis_agent.analyze_company_data(
            extracted_data=[result],
            company_name=request.company_name
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        confidence = 0.5
        if intelligence.get("products_services") and len(intelligence["products_services"]) > 0:
            confidence += 0.3
        if intelligence.get("company_overview", {}).get("description"):
            confidence += 0.2
        
        return AgenticExtractionResponse(
            success=True,
            url=str(request.url),
            raw_content=result.get("content", "")[:5000],
            extracted_data=intelligence,
            confidence=confidence,
            processing_time_seconds=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic extraction failed: {str(e)}")

@router.post("/crawl", response_model=CrawlResponse)
async def crawl_url(request: CrawlRequest):
    """Crawl a single URL (HTTP-based, no browser)"""
    try:
        task = CrawlTask(
            id="single-crawl",
            url=str(request.url),
            source_type=SourceType.OTHER,
            priority=50,
            strategy=CrawlStrategy.STANDARD if request.javascript else CrawlStrategy.FAST
        )
        
        extraction_agent = ExtractionAgent()
        result, success = await extraction_agent.extract_content(task)
        
        return CrawlResponse(
            success=success,
            url=str(request.url),
            title=result.get("title"),
            content=result.get("content"),
            markdown=result.get("content"),
            html="",
            links=result.get("links", []),
            images=[],
            metadata=result.get("metadata", {}),
            crawled_at=datetime.now(),
            duration_ms=0,
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crawl failed: {str(e)}")

@router.post("/crawl/batch", response_model=List[CrawlResponse])
async def crawl_batch(request: BatchCrawlRequest):
    """Crawl multiple URLs (HTTP-based, no browser)"""
    try:
        tasks = [
            CrawlTask(
                id=f"batch-{i}",
                url=str(url),
                source_type=SourceType.OTHER,
                priority=50,
                strategy=CrawlStrategy.STANDARD
            )
            for i, url in enumerate(request.urls)
        ]
        
        extraction_agent = ExtractionAgent()
        results, _ = await extraction_agent.batch_extract(tasks, request.concurrency)
        
        responses = []
        for result in results:
            responses.append(CrawlResponse(
                success=result.get("success", False),
                url=result.get("url", ""),
                title=result.get("title"),
                content=result.get("content"),
                markdown=result.get("content"),
                html="",
                links=result.get("links", []),
                images=[],
                metadata=result.get("metadata", {}),
                crawled_at=datetime.now(),
                duration_ms=0,
                error=result.get("error")
            ))
        
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch crawl failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "crawl4ai-agentic-http",
        "version": "3.1.0",
        "mode": "HTTP + Tavily (No Browser)",
        "features": [
            "http_crawling",
            "tavily_integration",
            "multi_agent_orchestration",
            "intelligent_caching"
        ]
    }

# ============================================================================
# SINGLETON FACTORY
# ============================================================================

_company_intelligence_orchestrator = None

def get_company_intelligence_orchestrator() -> AgenticCompanyIntelligenceOrchestrator:
    """Get singleton instance of AgenticCompanyIntelligenceOrchestrator"""
    global _company_intelligence_orchestrator
    if _company_intelligence_orchestrator is None:
        _company_intelligence_orchestrator = AgenticCompanyIntelligenceOrchestrator()
    return _company_intelligence_orchestrator
