import yfinance as yf
from typing import List, Dict
from schemas.finance_schema import FinancialSnapshot, FinancialDataResponse, FinancialDataRequest

def get_financial_snapshot(ticker: str) -> FinancialSnapshot:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return FinancialSnapshot(
            ticker=ticker,
            company_name=info.get("longName"),
            market_cap=info.get("marketCap"),
            revenue=info.get("totalRevenue"),
            net_income=info.get("netIncomeToCommon"),
            pe_ratio=info.get("trailingPE"),
            price=info.get("currentPrice")
        )
    except Exception:
        return FinancialSnapshot(
            ticker=ticker,
            company_name=None,
            market_cap=None,
            revenue=None,
            net_income=None,
            pe_ratio=None,
            price=None
        )

def get_financial_data(tickers: FinancialDataRequest) -> FinancialDataResponse:

    target_ticker = tickers.companyticker
    peers = tickers.competitortickers
    ticker_list = [target_ticker] + peers
    
    snapshots = []
    for ticker in ticker_list:
        snapshot = get_financial_snapshot(ticker)
        snapshots.append(snapshot)

    return FinancialDataResponse(
        target_ticker=target_ticker,
        competitor_tickers=peers,
        snapshots=snapshots
    )
