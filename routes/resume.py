# api/routes/resume.py
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Depends
from typing import List, Optional
import uuid
from ...services.resume.processor import process_single_resume, process_resume_batch, initialize_batch
from ...api.models.schemas import BatchStatus
from ...config import settings

router = APIRouter(prefix="/resumes", tags=["resumes"])

# Endpoint to process a single resume
@router.post("/parse")
async def parse_resume(
    file: UploadFile = File(...),
    jobId: Optional[str] = Form(None),
    userId: str = Form(...)
):
    """Parse a single resume file"""
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset file pointer
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
        
    # Check file type
    file_ext = file.filename.split('.')[-1].lower() if file.filename else ""
    if file_ext not in settings.ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    # Process the resume
    result = await process_single_resume(file, jobId, userId)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to process resume"))
        
    return result

# Endpoint to start batch processing
@router.post("/batch")
async def start_batch_processing(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    jobId: str = Form(...),
    userId: str = Form(...)
):
    """Start batch processing of multiple resumes"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
        
    # Validate files
    for file in files:
        file_ext = file.filename.split('.')[-1].lower() if file.filename else ""
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    # Create batch ID
    batch_id = str(uuid.uuid4())
    
    # Initialize batch status
    batch_status = await initialize_batch(
        batch_id,
        userId,
        jobId,
        [file.filename for file in files]
    )
    
    # Start processing in the background
    background_tasks.add_task(
        process_resume_batch,
        batch_id,
        files,
        jobId,
        userId
    )
    
    return {
        "batchId": batch_id,
        "message": f"Processing {len(files)} resumes in the background",
        "status": "PROCESSING"
    }

# Endpoint to check batch status
@router.get("/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get the status of a batch processing job"""
    # In a real implementation, you would fetch this from your database
    # For now, we'll simulate this by sending a request to the Next.js API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.NEXTJS_API_BASE_URL}/resume-batches/{batch_id}",
                headers={"Content-Type": "application/json"}
            )
            
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise HTTPException(status_code=404, detail="Batch not found")
        else:
            raise HTTPException(status_code=500, detail="Failed to get batch status")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))