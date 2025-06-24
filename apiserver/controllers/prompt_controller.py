from fastapi import APIRouter, HTTPException
from schemas.prompt_schema import CompanyRequest, PromptResponse
from services.prompt_service import generate_prompt

router = APIRouter()

@router.post("/prepare-prompt", response_model=PromptResponse)
def prepare_prompt(data: CompanyRequest):
    company = data.company
    if not company:
        raise HTTPException(status_code=400, detail="Company name is required")

    try:
        prompt = generate_prompt(company)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return prompt
