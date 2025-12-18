# services/resume/processor.py
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import httpx
import asyncio
from fastapi import UploadFile
from ...api.models.schemas import ParseResult, BatchStatus, ResumeStatus, StatusEnum
from ...utils.file_utils import extract_text_from_file
from .parser import process_resume_file
from .analyzer import analyze_resume_match
from ...config import settings

# Initialize batch status
async def initialize_batch(batch_id: str, user_id: str, job_id: str, file_names: List[str]) -> BatchStatus:
    """Initialize a new batch processing status"""
    expires_at = datetime.now() + timedelta(hours=24)
    
    status = BatchStatus(
        id=batch_id,
        userId=user_id,
        jobId=job_id,
        total=len(file_names),
        processed=0,
        successful=0,
        failed=0,
        errors=[],
        resumeIds=[],
        resumeStatuses=[
            ResumeStatus(
                id="",
                fileName=file_name,
                parseStatus=0,
                matchStatus=0,
                careerAnalysisStatus=0,
                completed=False
            )
            for file_name in file_names
        ],
        expiresAt=expires_at,
        createdAt=datetime.now(),
        updatedAt=datetime.now()
    )
    
    # In a real implementation, you would save this to your database
    # For now, we'll simulate this by sending it to the Next.js API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.NEXTJS_API_BASE_URL}/resume-batches",
                json=status.dict(),
                headers={"Content-Type": "application/json"}
            )
            
        if response.status_code != 200:
            print(f"Error initializing batch status: {response.status_code}")
            
    except Exception as e:
        print(f"Error initializing batch status: {str(e)}")
        
    return status

# Update batch status
async def update_batch_status(batch_id: str, status_update: Dict[str, Any]) -> bool:
    """Update batch status in the Next.js API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{settings.NEXTJS_API_BASE_URL}/resume-batches/{batch_id}",
                json=status_update,
                headers={"Content-Type": "application/json"}
            )
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error updating batch status: {str(e)}")
        return False

# Save resume to Next.js API
async def save_resume_to_nextjs(
    parsed_content: ParseResult, 
    file_content: bytes,
    file_name: str,
    job_id: str,
    user_id: str,
    match_score: float
) -> Dict[str, Any]:
    """Save resume data to Next.js API"""
    try:
        # Prepare data for API
        resume_data = {
            "fileName": file_name,
            "fileContent": file_content.decode('latin1'),  # Encode for JSON transfer
            "jobId": job_id,
            "userId": user_id,
            "status": StatusEnum.PARSED,
            "matchScore": match_score,
            "parsedContent": parsed_content.dict()
        }
        
        # Send to Next.js API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.NEXTJS_API_BASE_URL}/resumes",
                json=resume_data,
                headers={"Content-Type": "application/json"}
            )
            
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error saving resume: {response.status_code}")
            return {}
            
    except Exception as e:
        print(f"Error saving resume: {str(e)}")
        return {}

# Create candidate in Next.js API
async def create_candidate_in_nextjs(
    parsed_content: ParseResult,
    resume_id: str,
    job_id: str,
    user_id: str
) -> Dict[str, Any]:
    """Create candidate in Next.js API"""
    try:
        # Prepare candidate data
        candidate_data = {
            "name": parsed_content.name,
            "email": parsed_content.email or "",
            "phone": parsed_content.phone,
            "experience": parsed_content.experience,
            "education": parsed_content.education,
            "skills": parsed_content.skills.dict(),
            "yearsOfExperience": parsed_content.yearsOfExperience,
            "resumeId": resume_id,
            "jobId": job_id,
            "userId": user_id
        }
        
        # Send to Next.js API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.NEXTJS_API_BASE_URL}/candidates",
                json=candidate_data,
                headers={"Content-Type": "application/json"}
            )
            
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error creating candidate: {response.status_code}")
            return {}
            
    except Exception as e:
        print(f"Error creating candidate: {str(e)}")
        return {}

# Process single resume
async def process_single_resume(file: UploadFile, job_id: str, user_id: str) -> Dict[str, Any]:
    """Process a single resume file"""
    try:
        # Extract text from file
        text, file_ext = await extract_text_from_file(file)
        file.file.seek(0)  # Reset file pointer
        file_content = await file.read()  # Read file content again
        
        # Parse resume
        parsed_content = await process_resume_file(text, file.filename, file_ext)
        
        # Analyze match score if job_id provided
        match_score = 0
        if job_id:
            match_score = await analyze_resume_match(parsed_content, job_id)
            parsed_content.matchScore = match_score
        
        # Save resume to Next.js
        resume_result = await save_resume_to_nextjs(
            parsed_content, 
            file_content,
            file.filename,
            job_id,
            user_id,
            match_score
        )
        
        if not resume_result:
            return {
                "success": False,
                "error": "Failed to save resume"
            }
            
        # Create candidate
        candidate_result = await create_candidate_in_nextjs(
            parsed_content,
            resume_result.get("id", ""),
            job_id,
            user_id
        )
        
        if not candidate_result:
            return {
                "success": False,
                "error": "Failed to create candidate"
            }
            
        # Return success result
        return {
            "success": True,
            "resumeId": resume_result.get("id", ""),
            "candidateId": candidate_result.get("id", ""),
            "parsedContent": parsed_content.dict(),
            "matchScore": match_score
        }
        
    except Exception as e:
        print(f"Error processing resume {file.filename}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Process a batch of resumes
async def process_resume_batch(
    batch_id: str,
    files: List[UploadFile],
    job_id: str,
    user_id: str
) -> None:
    """Process a batch of resume files"""
    # Get batch status
    batch_status = BatchStatus(
        id=batch_id,
        userId=user_id,
        jobId=job_id,
        total=len(files),
        processed=0,
        successful=0,
        failed=0,
        errors=[],
        resumeIds=[],
        resumeStatuses=[
            ResumeStatus(
                id="",
                fileName=file.filename,
                parseStatus=0,
                matchStatus=0,
                careerAnalysisStatus=0,
                completed=False
            )
            for file in files
        ],
        expiresAt=datetime.now() + timedelta(hours=24),
        createdAt=datetime.now(),
        updatedAt=datetime.now()
    )
    
    # Process each file
    for file in files:
        try:
            # Update status to processing
            resume_index = next((i for i, rs in enumerate(batch_status.resumeStatuses) 
                               if rs.fileName == file.filename), -1)
                               
            if resume_index != -1:
                batch_status.resumeStatuses[resume_index].parseStatus = 50
                batch_status.updatedAt = datetime.now()
                
                # Update batch status
                await update_batch_status(batch_id, {
                    "resumeStatuses": [rs.dict() for rs in batch_status.resumeStatuses],
                    "updatedAt": batch_status.updatedAt.isoformat()
                })
            
            # Process the file
            result = await process_single_resume(file, job_id, user_id)
            
            # Update batch status
            batch_status.processed += 1
            
            if result["success"]:
                batch_status.successful += 1
                batch_status.resumeIds.append(result["resumeId"])
                
                if resume_index != -1:
                    batch_status.resumeStatuses[resume_index].id = result["resumeId"]
                    batch_status.resumeStatuses[resume_index].parseStatus = 100
                    batch_status.resumeStatuses[resume_index].matchStatus = 100
                    batch_status.resumeStatuses[resume_index].careerAnalysis
                    # Continuing from services/resume/processor.py
                    batch_status.resumeStatuses[resume_index].careerAnalysisStatus = 100
                    batch_status.resumeStatuses[resume_index].completed = True
            else:
                batch_status.failed += 1
                batch_status.errors.append(f"Error processing {file.filename}: {result.get('error', 'Unknown error')}")
                
                if resume_index != -1:
                    batch_status.resumeStatuses[resume_index].parseStatus = 0
                    batch_status.resumeStatuses[resume_index].completed = True
            
            # Update batch status
            batch_status.updatedAt = datetime.now()
            await update_batch_status(batch_id, batch_status.dict())
                
        except Exception as e:
            print(f"Error processing file {file.filename}: {str(e)}")
            batch_status.processed += 1
            batch_status.failed += 1
            batch_status.errors.append(f"Error processing {file.filename}: {str(e)}")
            batch_status.updatedAt = datetime.now()
            
            # Update batch status
            await update_batch_status(batch_id, batch_status.dict())