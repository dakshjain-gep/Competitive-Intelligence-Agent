from fastapi import APIRouter
from schemas.data_schema import DataRequest, DataResponse
from services.external_api_service import fetch_news

router = APIRouter()

@router.post("/get-data", response_model=DataResponse)
def get_data(payload: DataRequest):
    news = fetch_news(payload.company)
    return DataResponse(company=payload.company, news=news)
