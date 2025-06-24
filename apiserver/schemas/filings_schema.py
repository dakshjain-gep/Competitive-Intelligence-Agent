from pydantic import BaseModel
from typing import List

class FilingItem(BaseModel):
    form_type: str
    filed_at: str
    description: str
    document_url: str

class FilingRequest(BaseModel):
    company: str  # Can be CIK or name

class FilingText(BaseModel):
    text: str
    document_url: str
    
class FilingResponse(BaseModel):
    company: str
    filings: List[FilingText]
    
