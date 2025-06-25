from fastapi import APIRouter, HTTPException
from services.finance_service import get_financial_data
from schemas.finance_schema import FinancialDataResponse, FinancialDataRequest

router = APIRouter()

@router.post("/finance", response_model=FinancialDataResponse)
def get_financial_data_controller(payload: FinancialDataRequest):
    try:
        result = get_financial_data(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
