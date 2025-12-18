import re
import PyPDF2
import docx
from typing import Dict, Optional
import io

class ResumeParser:
    def __init__(self):
        # Email pattern
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        # Phone patterns (supports various formats)
        self.phone_pattern = r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        # LinkedIn pattern
        self.linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
        
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_file = io.BytesIO(file_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            return ""
    
    def extract_text_from_docx(self, file_bytes: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            doc_file = io.BytesIO(file_bytes)
            doc = docx.Document(doc_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error extracting DOCX text: {e}")
            return ""
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract name from resume (usually the first line or near the top)"""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line) > 0 and 2 <= len(line.split()) <= 4:  # Name usually has 2-4 words
                # Skip lines with email, phone, or common headers
                if not re.search(self.email_pattern, line) and \
                   not re.search(self.phone_pattern, line) and \
                   not any(keyword in line.lower() for keyword in ['resume', 'cv', 'curriculum']):
                    return line
        return ""
    
    def extract_email(self, text: str) -> str:
        """Extract email address"""
        emails = re.findall(self.email_pattern, text)
        return emails[0] if emails else ""
    
    def extract_phone(self, text: str) -> str:
        """Extract phone number"""
        phones = re.findall(self.phone_pattern, text)
        return phones[0] if phones else ""
    
    def extract_location(self, text: str) -> str:
        """Extract location/address"""
        # Common location patterns
        location_patterns = [
            r'(?:Location|Address|City):\s*([^\n]+)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2})\b',  # City, STATE
            r'\b([A-Z][a-z]+,\s*[A-Z][a-z]+)\b',  # City, Country
        ]
        
        for pattern in location_patterns:
            locations = re.findall(pattern, text, re.IGNORECASE)
            if locations:
                return locations[0].strip()
        
        # Look for common city names in the first 10 lines
        lines = text.split('\n')[:10]
        for line in lines:
            # Check if line contains location keywords
            if any(keyword in line.lower() for keyword in ['location:', 'address:', 'city:', 'based in']):
                location = line.split(':')[-1].strip()
                if location and len(location) < 50:  # Reasonable location length
                    return location
        
        return ""
    
    def extract_linkedin(self, text: str) -> str:
        """Extract LinkedIn profile URL"""
        linkedin = re.findall(self.linkedin_pattern, text, re.IGNORECASE)
        if linkedin:
            url = linkedin[0]
            # Ensure it's a proper URL
            if not url.startswith('http'):
                url = 'https://' + url
            return url
        return ""
    
    def parse(self, file_bytes: bytes, filename: str) -> Dict:
        """Parse resume and extract only required fields: name, email, phone, location, linkedin"""
        # Extract text based on file type
        if filename.lower().endswith('.pdf'):
            text = self.extract_text_from_pdf(file_bytes)
        elif filename.lower().endswith(('.docx', '.doc')):
            text = self.extract_text_from_docx(file_bytes)
        else:
            return {
                "error": "Unsupported file format. Please upload PDF or DOCX file."
            }
        
        if not text:
            return {
                "error": "Could not extract text from resume"
            }
        
        # Extract only the 5 required fields
        parsed_data = {
            "name": self.extract_name(text),
            "email": self.extract_email(text),
            "phone": self.extract_phone(text),
            "location": self.extract_location(text),
            "linkedin": self.extract_linkedin(text),
        }
        
        return parsed_data