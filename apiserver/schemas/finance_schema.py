from pydantic import BaseModel
from typing import List, Dict, Optional

class FinancialDataRequest(BaseModel):
    companyticker: str
    competitortickers: List[str]

class FinancialSnapshot(BaseModel):
    ticker: str
    company_name: Optional[str]
    market_cap: Optional[float]
    revenue: Optional[float]
    net_income: Optional[float]
    pe_ratio: Optional[float]
    price: Optional[float]

class FinancialDataResponse(BaseModel):
    target_ticker: str
    competitor_tickers: List[str]
    snapshots: List[FinancialSnapshot]
