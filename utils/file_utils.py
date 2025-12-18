# utils/file_utils.py
import io
from typing import Tuple
import pdfplumber
import docx2txt
import csv
import pandas as pd
from io import StringIO
from fastapi import UploadFile
import mammoth

async def extract_text_from_file(file: UploadFile) -> Tuple[str, str]:
    """Extract text from various file formats."""
    content = await file.read()
    file_ext = file.filename.split('.')[-1].lower() if file.filename else ""
    
    text = ""
    
    try:
        if file_ext == 'pdf':
            # Use pdfplumber which has better text extraction
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
                
        elif file_ext in ['docx', 'doc']:
            # Try mammoth first for better DOCX handling
            try:
                result = mammoth.extract_raw_text({"buffer": content})
                text = result.value
            except:
                # Fallback to docx2txt
                text = docx2txt.process(io.BytesIO(content))
                
        elif file_ext == 'txt':
            text = content.decode('utf-8', errors='replace')
            
        elif file_ext == 'csv':
            # Parse CSV using pandas for better handling
            df = pd.read_csv(io.BytesIO(content))
            text = df.to_string(index=False)
            
        elif file_ext in ['xlsx', 'xls']:
            # Read Excel using pandas
            df = pd.read_excel(io.BytesIO(content))
            text = df.to_string(index=False)
            
        else:
            text = content.decode('utf-8', errors='replace')
            
    except Exception as e:
        print(f"Error extracting text from {file.filename}: {str(e)}")
        text = ""
        
    # Clean up text
    text = text.replace('\x00', '')  # Remove null bytes
    text = ' '.join(text.split())     # Normalize whitespace
    
    return text, file_ext