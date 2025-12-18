# prescreening/database.py - Database Connection and Operations for Pre-screening
import os
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import asyncpg
from contextlib import asynccontextmanager
import logging

from .models import (
    CandidatePreScreening, PreScreeningSession, MCQQuestion, 
    HumanReviewTask, PreScreeningNotification, ResumeMatchingResult,
    ProctoringEvent, PreScreeningStatus, ScoreBucket, ReviewStatus,
    NotificationType
)

logger = logging.getLogger(__name__)

class PreScreeningDatabase:
    """Database operations for pre-screening system"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "postgresql://localhost:5432/hiregenix")
        self.pool = None
    
    async def init_pool(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def close_pool(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            await self.init_pool()
        
        async with self.pool.acquire() as conn:
            yield conn
    
    async def create_prescreening(self, prescreening: CandidatePreScreening) -> str:
        """Create new pre-screening record"""
        async with self.get_connection() as conn:
            try:
                query = """
                INSERT INTO "CandidatePreScreening" (
                    id, candidate_id, job_id, resume_score, resume_decision,
                    prescreening_status, prescreening_score, prescreening_decision,
                    human_review_required, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """
                
                result = await conn.fetchrow(
                    query,
                    prescreening.id,
                    prescreening.candidate_id,
                    prescreening.job_id,
                    prescreening.resume_score,
                    prescreening.resume_decision,
                    prescreening.prescreening_status.value,
                    prescreening.prescreening_score,
                    prescreening.prescreening_decision,
                    prescreening.human_review_required,
                    prescreening.created_at,
                    prescreening.updated_at
                )
                
                return result['id']
                
            except Exception as e:
                logger.error(f"Error creating prescreening: {e}")
                raise
    
    async def get_prescreening(self, prescreening_id: str) -> Optional[CandidatePreScreening]:
        """Get pre-screening record by ID"""
        async with self.get_connection() as conn:
            try:
                query = """
                SELECT * FROM "CandidatePreScreening" WHERE id = $1
                """
                
                row = await conn.fetchrow(query, prescreening_id)
                if not row:
                    return None
                
                return CandidatePreScreening(
                    id=row['id'],
                    candidate_id=row['candidate_id'],
                    job_id=row['job_id'],
                    resume_score=float(row['resume_score']) if row['resume_score'] else 0.0,
                    resume_decision=row['resume_decision'],
                    prescreening_status=PreScreeningStatus(row['prescreening_status']),
                    prescreening_score=float(row['prescreening_score']) if row['prescreening_score'] else None,
                    prescreening_decision=row['prescreening_decision'],
                    human_review_required=row['human_review_required'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
                
            except Exception as e:
                logger.error(f"Error getting prescreening: {e}")
                return None
    
    async def update_prescreening_status(
        self,
        prescreening_id: str,
        new_status: PreScreeningStatus,
        notes: Optional[str] = None
    ) -> bool:
        """Update pre-screening status"""
        async with self.get_connection() as conn:
            try:
                query = """
                UPDATE "CandidatePreScreening"
                SET prescreening_status = $2, updated_at = $3
                WHERE id = $1
                """
                
                result = await conn.execute(
                    query,
                    prescreening_id,
                    new_status.value,
                    datetime.now()
                )
                
                return result == "UPDATE 1"
                
            except Exception as e:
                logger.error(f"Error updating prescreening status: {e}")
                return False
    
    async def create_prescreening_session(self, session: PreScreeningSession) -> str:
        """Create pre-screening session"""
        async with self.get_connection() as conn:
            try:
                query = """
                INSERT INTO "PreScreeningSession" (
                    id, candidate_id, job_id, session_token, status,
                    start_time, end_time, expires_at, intro_question,
                    total_questions, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
                """
                
                result = await conn.fetchrow(
                    query,
                    session.id,
                    session.candidate_id,
                    session.job_id,
                    session.session_token,
                    session.status.value,
                    session.start_time,
                    session.end_time,
                    session.expires_at,
                    session.intro_question,
                    session.total_questions,
                    session.created_at,
                    session.updated_at
                )
                
                return result['id']
                
            except Exception as e:
                logger.error(f"Error creating prescreening session: {e}")
                raise
    
    async def create_human_review_task(self, task: HumanReviewTask) -> str:
        """Create human review task"""
        async with self.get_connection() as conn:
            try:
                query = """
                INSERT INTO "HumanReviewTask" (
                    id, candidate_id, job_id, stage, score, review_type,
                    priority, status, assigned_to, assigned_at,
                    decision, decision_reason, decision_notes, override_score,
                    context_data, ai_recommendation, recommendation_reason,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                RETURNING id
                """
                
                result = await conn.fetchrow(
                    query,
                    task.id,
                    task.candidate_id,
                    task.job_id,
                    task.stage,
                    task.score,
                    task.review_type,
                    task.priority,
                    task.status.value,
                    task.assigned_to,
                    task.assigned_at,
                    task.decision.value if task.decision else None,
                    task.decision_reason,
                    task.decision_notes,
                    task.override_score,
                    json.dumps(task.context_data),
                    task.ai_recommendation,
                    task.recommendation_reason,
                    task.created_at,
                    task.updated_at
                )
                
                return result['id']
                
            except Exception as e:
                logger.error(f"Error creating human review task: {e}")
                raise
    
    async def create_notification(self, notification: PreScreeningNotification) -> str:
        """Create notification"""
        async with self.get_connection() as conn:
            try:
                query = """
                INSERT INTO "PreScreeningNotification" (
                    id, prescreening_id, candidate_id, job_id,
                    notification_type, title, message, metadata,
                    sent, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
                """
                
                result = await conn.fetchrow(
                    query,
                    notification.id,
                    notification.prescreening_id,
                    notification.candidate_id,
                    notification.job_id,
                    notification.notification_type.value,
                    notification.title,
                    notification.message,
                    json.dumps(notification.metadata),
                    notification.sent,
                    notification.created_at
                )
                
                return result['id']
                
            except Exception as e:
                logger.error(f"Error creating notification: {e}")
                raise
    
    async def record_proctoring_event(self, event: ProctoringEvent) -> str:
        """Record proctoring event"""
        async with self.get_connection() as conn:
            try:
                query = """
                INSERT INTO "ProctoringEvent" (
                    id, session_id, event_type, timestamp, confidence_score,
                    severity, metadata, reviewed, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """
                
                result = await conn.fetchrow(
                    query,
                    event.id,
                    event.session_id,
                    event.event_type,
                    event.timestamp,
                    event.confidence_score,
                    event.severity,
                    json.dumps(event.metadata),
                    event.reviewed,
                    event.created_at
                )
                
                return result['id']
                
            except Exception as e:
                logger.error(f"Error recording proctoring event: {e}")
                raise
    
    async def get_prescreening_analytics(self) -> Dict[str, Any]:
        """Get real-time pre-screening analytics"""
        async with self.get_connection() as conn:
            try:
                # Get status counts
                status_query = """
                SELECT prescreening_status, COUNT(*) as count
                FROM "CandidatePreScreening"
                GROUP BY prescreening_status
                """
                status_rows = await conn.fetch(status_query)
                status_counts = {row['prescreening_status']: row['count'] for row in status_rows}
                
                # Get average scores
                avg_query = """
                SELECT
                    AVG(resume_score) as avg_resume,
                    AVG(prescreening_score) as avg_prescreening
                FROM "CandidatePreScreening"
                WHERE resume_score > 0
                """
                avg_row = await conn.fetchrow(avg_query)
                
                # Get processing metrics
                metrics_query = """
                SELECT
                    COUNT(*) as total_processed,
                    COUNT(CASE WHEN prescreening_status != 'cancelled' THEN 1 END) as successful
                FROM "CandidatePreScreening"
                """
                metrics_row = await conn.fetchrow(metrics_query)
                
                total = metrics_row['total_processed']
                successful = metrics_row['successful']
                success_rate = successful / total if total > 0 else 0
                
                return {
                    "total_prescreenings": total,
                    "by_status": status_counts,
                    "average_scores": {
                        "resume": float(avg_row['avg_resume'] or 0),
                        "prescreening": float(avg_row['avg_prescreening'] or 0)
                    },
                    "processing_metrics": {
                        "success_rate": success_rate,
                        "total_processed": total,
                        "successful_processed": successful
                    },
                    "generated_at": datetime.now()
                }
                
            except Exception as e:
                logger.error(f"Error getting analytics: {e}")
                return {}

# Global database instance
db_instance = PreScreeningDatabase()

async def get_database() -> PreScreeningDatabase:
    """Get database instance"""
    return db_instance

async def init_database():
    """Initialize database connection"""
    await db_instance.init_pool()

async def close_database():
    """Close database connection"""
    await db_instance.close_pool()