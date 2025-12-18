# prescreening/integration.py - Integration with Main FastAPI App
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from .api import prescreening_router
from .database import init_database, close_database

logger = logging.getLogger(__name__)

@asynccontextmanager
async def prescreening_lifespan(app: FastAPI):
    """Lifespan manager for pre-screening module"""
    # Startup
    try:
        await init_database()
        logger.info("Pre-screening database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize pre-screening database: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        await close_database()
        logger.info("Pre-screening database connections closed")
    except Exception as e:
        logger.error(f"Error closing pre-screening database: {e}")

def setup_prescreening_module(app: FastAPI):
    """Setup pre-screening module with main FastAPI app"""
    
    # Include API router
    app.include_router(prescreening_router)
    
    logger.info("Pre-screening module setup complete")
    
    return app

def add_prescreening_middleware(app: FastAPI):
    """Add pre-screening specific middleware"""
    
    @app.middleware("http")
    async def prescreening_middleware(request, call_next):
        # Add any pre-screening specific request processing
        if request.url.path.startswith("/api/prescreening"):
            # Add pre-screening specific headers
            request.state.module = "prescreening"
        
        response = await call_next(request)
        
        if hasattr(request.state, "module") and request.state.module == "prescreening":
            response.headers["X-HireGenix-Module"] = "prescreening"
        
        return response
    
    logger.info("Pre-screening middleware added")

# Health check specifically for pre-screening
async def prescreening_health_check():
    """Health check for pre-screening module"""
    try:
        from .database import get_database
        from .config import settings
        
        # Check database connection
        db = await get_database()
        if not db.pool:
            await db.init_pool()
        
        # Check configuration
        if not settings.validate_config():
            return {
                "status": "unhealthy",
                "error": "Invalid configuration",
                "module": "prescreening"
            }
        
        return {
            "status": "healthy",
            "module": "prescreening",
            "database": "connected",
            "config": "valid"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "module": "prescreening"
        }