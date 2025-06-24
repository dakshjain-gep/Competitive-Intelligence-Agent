from pydantic import BaseModel

class CompanyRequest(BaseModel):
    company: str

class PromptResponse(BaseModel):
    company: str
    articles: str
    filings: str