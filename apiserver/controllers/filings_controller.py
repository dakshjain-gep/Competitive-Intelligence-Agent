from fastapi import APIRouter, HTTPException
from schemas.filings_schema import FilingRequest, FilingResponse
from services.filings_service import fetch_sec_filings

router = APIRouter()

@router.post("/get-filings", response_model=FilingResponse)
def get_filings(payload: FilingRequest):
    try:
        filings = fetch_sec_filings(payload.company)
        return FilingResponse(company=payload.company, filings=filings)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))