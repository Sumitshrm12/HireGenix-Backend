# api/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ResumeExperience(BaseModel):
    title: str = ""
    company: str = ""
    startDate: str = ""
    endDate: str = ""
    description: str = ""

class ResumeEducation(BaseModel):
    degree: str = ""
    institution: str = ""
    startDate: str = ""
    endDate: str = ""

class ResumeSkills(BaseModel):
    technical: List[str] = Field(default_factory=list)
    soft: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)

class ResumeData(BaseModel):
    name: str = "Unknown"
    email: Optional[str] = None
    phone: Optional[str] = None
    experience: List[ResumeExperience] = Field(default_factory=list)
    education: List[ResumeEducation] = Field(default_factory=list)
    skills: ResumeSkills = Field(default_factory=ResumeSkills)
    yearsOfExperience: int = 0

class ParseResult(ResumeData):
    fileName: str = ""
    fileType: str = ""
    parseDate: str = ""
    rawText: str = ""
    textLength: int = 0
    matchScore: Optional[float] = None
    parseWarning: Optional[str] = None
    needsReview: bool = False
    parseSuccess: bool = True
    error: Optional[str] = None

class StatusEnum(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    PARSED = "PARSED"
    FAILED = "FAILED"

class ResumeStatus(BaseModel):
    id: str = ""
    fileName: str
    parseStatus: int = 0
    matchStatus: int = 0
    careerAnalysisStatus: int = 0
    completed: bool = False

class BatchRequest(BaseModel):
    jobId: str
    userId: str
    files: List[str]  # List of file paths or identifiers

class BatchStatus(BaseModel):
    id: str
    userId: str
    jobId: str
    total: int
    processed: int
    successful: int
    failed: int
    errors: List[str] = Field(default_factory=list)
    resumeIds: List[str] = Field(default_factory=list)
    resumeStatuses: List[ResumeStatus] = Field(default_factory=list)
    expiresAt: datetime
    createdAt: datetime = Field(default_factory=datetime.now)
    updatedAt: datetime = Field(default_factory=datetime.now)

class JobData(BaseModel):
    id: str
    title: str
    description: str
    requirements: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    
class ResumeRequest(BaseModel):
    jobId: Optional[str] = None
    userId: str