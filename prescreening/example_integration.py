# prescreening/example_integration.py - Example of how to integrate with main.py
"""
Example of how to integrate the pre-screening module with your existing main.py file.
Add these imports and modifications to your main FastAPI application.
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pre-screening module
from prescreening.integration import (
    setup_prescreening_module,
    add_prescreening_middleware,
    prescreening_lifespan,
    prescreening_health_check
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan manager including pre-screening"""
    # Startup
    logger.info("Starting HireGenix application...")
    
    # Initialize pre-screening module
    async with prescreening_lifespan(app):
        logger.info("Pre-screening module ready")
        yield
    
    logger.info("Application shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="HireGenix API",
    description="AI-Powered Hiring Platform with Pre-screening",
    version="3.0.0",
    lifespan=lifespan
)

# Setup pre-screening module
setup_prescreening_module(app)

# Add pre-screening middleware
add_prescreening_middleware(app)

# Enhanced health check including pre-screening
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including all modules"""
    health_status = {
        "application": "healthy",
        "timestamp": datetime.now(),
        "modules": {}
    }
    
    # Check pre-screening module
    prescreening_health = await prescreening_health_check()
    health_status["modules"]["prescreening"] = prescreening_health
    
    # Add other module health checks as needed
    
    # Determine overall health
    module_statuses = [module["status"] for module in health_status["modules"].values()]
    overall_healthy = all(status == "healthy" for status in module_statuses)
    
    health_status["status"] = "healthy" if overall_healthy else "degraded"
    
    return health_status

# Example usage endpoint
@app.post("/api/example/prescreening")
async def example_prescreening_workflow():
    """
    Example of how to use the pre-screening module in your existing workflows
    """
    from prescreening.service import create_prescreening_service
    
    # Create service instance
    service = create_prescreening_service()
    
    # Example data (in real use, this would come from your existing resume processing)
    example_data = {
        "candidate_id": "cand_123456",
        "job_id": "job_789012", 
        "resume_text": "Software Engineer with 5 years experience in Python, React, and AWS...",
        "job_description": "Senior Software Engineer role requiring Python, JavaScript, cloud experience...",
        "job_requirements": ["Python", "React", "AWS", "5+ years experience"]
    }
    
    try:
        # Start pre-screening process
        result = await service.start_prescreening(
            candidate_id=example_data["candidate_id"],
            job_id=example_data["job_id"], 
            resume_text=example_data["resume_text"],
            job_description=example_data["job_description"],
            job_requirements=example_data["job_requirements"]
        )
        
        return {
            "success": True,
            "prescreening_id": result.id,
            "status": result.status.value,
            "score": result.overall_score,
            "bucket": result.bucket.value,
            "next_action": result.next_action
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Example of how to add pre-screening to existing resume processing
@app.post("/api/resume/process-with-prescreening")
async def process_resume_with_prescreening(
    resume_file,
    job_id: str
):
    """
    Enhanced resume processing that includes pre-screening
    """
    # Your existing resume processing logic here
    # ... extract text, parse resume, etc ...
    
    # After resume processing, add pre-screening
    from prescreening.service import create_prescreening_service
    
    service = create_prescreening_service()
    
    # Get job details (from your existing job service)
    # job = await get_job_details(job_id)
    
    # Start pre-screening
    prescreening_result = await service.start_prescreening(
        candidate_id="extracted_candidate_id",
        job_id=job_id,
        resume_text="extracted_resume_text",
        job_description="job.description",
        job_requirements=["job.requirements"]
    )
    
    return {
        "resume_processed": True,
        "prescreening": {
            "id": prescreening_result.id,
            "score": prescreening_result.overall_score,
            "status": prescreening_result.status.value,
            "next_action": prescreening_result.next_action
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )