import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from typing import Dict, Tuple, Optional, List
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Analyzer",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitFinancialAnalyzer:
    """Enhanced Financial Analyzer for Streamlit with dynamic date handling and optimization"""
    
    def __init__(self):
        self.current_date = datetime.now()
        self.five_years_ago = self.current_date - timedelta(days=5*365)
        
    def get_date_range(self, years: int = 5) -> Tuple[str, str]:
        """Get dynamic date range for the specified number of years from current date"""
        end_date = self.current_date.strftime('%Y-%m-%d')
        start_date = (self.current_date - timedelta(days=years*365)).strftime('%Y-%m-%d')
        return start_date, end_date
    
    def format_large_number(self, number) -> str:
        """Format large numbers in readable format with enhanced precision"""
        if number is None or pd.isna(number):
            return "N/A"
        
        try:
            num = float(number)
            if abs(num) >= 1e12:
                return f"${num/1e12:.2f}T"
            elif abs(num) >= 1e9:
                return f"${num/1e9:.2f}B"
            elif abs(num) >= 1e6:
                return f"${num/1e6:.2f}M"
            elif abs(num) >= 1e3:
                return f"${num/1e3:.2f}K"
            else:
                return f"${num:.2f}"
        except (ValueError, TypeError):
            return "N/A"

    def format_percentage(self, value) -> str:
        """Format decimal values as percentages with better handling"""
        if value is None or pd.isna(value):
            return "N/A"
        
        try:
            return f"{float(value) * 100:.2f}%"
        except (ValueError, TypeError):
            return "N/A"

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_financial_summary(_self, ticker_symbol: str) -> Tuple[Optional[Dict], Optional[Dict], Optional[pd.DataFrame]]:
        """Get comprehensive financial summary with dynamic date handling"""
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Get historical data for dynamic 5-year period
            start_date, end_date = _self.get_date_range(5)
            history = ticker.history(start=start_date, end=end_date)
            
            if history.empty:
                st.warning(f"No historical data found for {ticker_symbol}")
                return None, None, None

            # Get basic info with comprehensive error handling
            info = ticker.info
            
            # Enhanced financial metrics
            financials = {
                "company_name": info.get("longName", "N/A"),
                "ticker": ticker_symbol.upper(),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "country": info.get("country", "N/A"),
                "website": info.get("website", "N/A"),
                "market_cap": _self.format_large_number(info.get("marketCap")),
                "enterprise_value": _self.format_large_number(info.get("enterpriseValue")),
                "analysis_date": _self.current_date.strftime('%Y-%m-%d'),
                "data_period": f"{start_date} to {end_date}",
                
                # Financial Performance
                "revenue": _self.format_large_number(info.get("totalRevenue")),
                "revenue_per_share": round(info.get("revenuePerShare", 0), 2) if info.get("revenuePerShare") else "N/A",
                "net_income": _self.format_large_number(info.get("netIncomeToCommon")),
                "EPS": round(info.get("trailingEps", 0), 3) if info.get("trailingEps") else "N/A",
                "EPS_forward": round(info.get("forwardEps", 0), 3) if info.get("forwardEps") else "N/A",
                "EBITDA": _self.format_large_number(info.get("ebitda")),
                "gross_profit": _self.format_large_number(info.get("grossProfits")),
                
                # Margins and Ratios
                "profit_margin": _self.format_percentage(info.get("profitMargins")),
                "operating_margin": _self.format_percentage(info.get("operatingMargins")),
                "gross_margin": _self.format_percentage(info.get("grossMargins")),
                "return_on_equity": _self.format_percentage(info.get("returnOnEquity")),
                "return_on_assets": _self.format_percentage(info.get("returnOnAssets")),
                
                # Growth Metrics
                "revenue_growth": _self.format_percentage(info.get("revenueGrowth")),
                "earnings_growth": _self.format_percentage(info.get("earningsGrowth")),
                "revenue_growth_quarterly": _self.format_percentage(info.get("revenueQuarterlyGrowth")),
                "earnings_growth_quarterly": _self.format_percentage(info.get("earningsQuarterlyGrowth")),
                
                # Balance Sheet - Expanded
                "total_assets": _self.format_large_number(info.get("totalAssets")),
                "current_assets": _self.format_large_number(info.get("currentAssets")),
                "cash_reserves": _self.format_large_number(info.get("totalCash")),
                "total_debt": _self.format_large_number(info.get("totalDebt")),
                "current_liabilities": _self.format_large_number(info.get("currentLiabilities")),
                "shareholders_equity": _self.format_large_number(info.get("totalStockholderEquity")),
                "book_value": _self.format_large_number(info.get("bookValue")),
                
                # Stock Metrics
                "current_price": round(history['Close'].iloc[-1], 2) if not history.empty else "N/A",
                "52_week_high": round(info.get("fiftyTwoWeekHigh", 0), 2) if info.get("fiftyTwoWeekHigh") else "N/A",
                "52_week_low": round(info.get("fiftyTwoWeekLow", 0), 2) if info.get("fiftyTwoWeekLow") else "N/A",
                "beta": round(info.get("beta", 0), 3) if info.get("beta") else "N/A",
                "shares_outstanding": _self.format_large_number(info.get("sharesOutstanding")),
                "float_shares": _self.format_large_number(info.get("floatShares")),
                
                # Valuation Ratios
                "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
                "forward_pe": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else "N/A",
                "pb_ratio": round(info.get("priceToBook", 0), 2) if info.get("priceToBook") else "N/A",
                "ps_ratio": round(info.get("priceToSalesTrailing12Months", 0), 2) if info.get("priceToSalesTrailing12Months") else "N/A",
                "peg_ratio": round(info.get("pegRatio", 0), 2) if info.get("pegRatio") else "N/A",
                "ev_revenue": round(info.get("enterpriseToRevenue", 0), 2) if info.get("enterpriseToRevenue") else "N/A",
                "ev_ebitda": round(info.get("enterpriseToEbitda", 0), 2) if info.get("enterpriseToEbitda") else "N/A",
                
                # Dividend Information - Expanded
                "dividend_yield": _self.format_percentage(info.get("dividendYield")),
                "dividend_rate": round(info.get("dividendRate", 0), 3) if info.get("dividendRate") else "N/A",
                "payout_ratio": _self.format_percentage(info.get("payoutRatio")),
                "ex_dividend_date": str(datetime.fromtimestamp(info.get("exDividendDate"))) if info.get("exDividendDate") else "N/A",
                
                # Analyst Information - Expanded
                "analyst_rating": info.get("recommendationKey", "N/A").replace("_", " ").title(),
                "target_price": round(info.get("targetMeanPrice", 0), 2) if info.get("targetMeanPrice") else "N/A",
                "target_high": round(info.get("targetHighPrice", 0), 2) if info.get("targetHighPrice") else "N/A",
                "target_low": round(info.get("targetLowPrice", 0), 2) if info.get("targetLowPrice") else "N/A",
                "num_analysts": info.get("numberOfAnalystOpinions", "N/A"),
            }

            # Enhanced time-series data with better error handling
            time_series_data = _self._get_time_series_data(ticker, history)

            return financials, time_series_data, history

        except Exception as e:
            st.error(f"Error getting financial data for {ticker_symbol}: {str(e)}")
            return None, None, None

    def _get_time_series_data(self, ticker, history: pd.DataFrame) -> Dict:
        """Get comprehensive time-series data, safely handling date formats."""
        time_series = {"revenue": {}, "stock_price": {}, "volume": {}, "quarterly_earnings": {}}
        
        try:
            # Quarterly financials
            quarterly_financials = ticker.quarterly_financials
            if not quarterly_financials.empty and "Total Revenue" in quarterly_financials.index:
                # Filter out NaT values from the index before processing
                revenue_data = quarterly_financials.loc["Total Revenue"].sort_index()
                revenue_data = revenue_data[pd.notna(revenue_data.index)]
                time_series["revenue"] = {
                    date.strftime("%Y-Q") + str(((date.month - 1) // 3 + 1)): float(value) 
                    for date, value in revenue_data.items() 
                    if pd.notna(value)
                }
            
            # Monthly stock prices
            if not history.empty:
                monthly_prices = history["Close"].resample("M").mean()
                time_series["stock_price"] = {
                    date.strftime("%Y-%m"): round(float(price), 2) 
                    for date, price in monthly_prices.items() 
                    if pd.notna(price)
                }
                
                # Monthly volume
                monthly_volume = history["Volume"].resample("M").sum()
                time_series["volume"] = {
                    date.strftime("%Y-%m"): int(volume) 
                    for date, volume in monthly_volume.items() 
                    if pd.notna(volume)
                }
            
            # Quarterly earnings
            quarterly_earnings = ticker.quarterly_earnings
            if not quarterly_earnings.empty:
                # Filter out NaT values from the index before processing
                quarterly_earnings_filtered = quarterly_earnings[pd.notna(quarterly_earnings.index)]
                time_series["quarterly_earnings"] = {
                    date.strftime("%Y-Q") + str(((date.month - 1) // 3 + 1)): float(value) 
                    for date, value in quarterly_earnings_filtered["Earnings"].items() 
                    if pd.notna(value)
                }
                
        except Exception as e:
            # Catching the specific ValueError for format string issues
            # Or general exceptions if the data structure is unexpected
            st.warning(f"Could not retrieve some time-series data due to data format issues: {str(e)}")
            
        return time_series

    def calculate_advanced_technical_indicators(self, history: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        if history is None or history.empty:
            return {}
        
        try:
            indicators = {}
            close_prices = history['Close']
            high_prices = history['High']
            low_prices = history['Low']
            volume = history['Volume']
            
            # Moving Averages
            indicators.update({
                "sma_10": round(close_prices.rolling(window=10).mean().iloc[-1], 2) if len(close_prices) >= 10 and not close_prices.rolling(window=10).mean().empty else "N/A",
                "sma_20": round(close_prices.rolling(window=20).mean().iloc[-1], 2) if len(close_prices) >= 20 and not close_prices.rolling(window=20).mean().empty else "N/A",
                "sma_50": round(close_prices.rolling(window=50).mean().iloc[-1], 2) if len(close_prices) >= 50 and not close_prices.rolling(window=50).mean().empty else "N/A",
                "sma_200": round(close_prices.rolling(window=200).mean().iloc[-1], 2) if len(close_prices) >= 200 and not close_prices.rolling(window=200).mean().empty else "N/A",
            })
            
            # Exponential Moving Averages
            indicators.update({
                "ema_12": round(close_prices.ewm(span=12, adjust=False).mean().iloc[-1], 2) if len(close_prices) >= 12 and not close_prices.ewm(span=12, adjust=False).mean().empty else "N/A",
                "ema_26": round(close_prices.ewm(span=26, adjust=False).mean().iloc[-1], 2) if len(close_prices) >= 26 and not close_prices.ewm(span=26, adjust=False).mean().empty else "N/A",
            })
            
            # MACD
            if len(close_prices) >= 26:
                ema_12 = close_prices.ewm(span=12, adjust=False).mean()
                ema_26 = close_prices.ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                indicators.update({
                    "macd": round(macd.iloc[-1], 4) if not macd.empty else "N/A",
                    "macd_signal": round(macd_signal.iloc[-1], 4) if not macd_signal.empty else "N/A",
                    "macd_histogram": round((macd - macd_signal).iloc[-1], 4) if not (macd - macd_signal).empty else "N/A",
                })
            else:
                indicators.update({"macd": "N/A", "macd_signal": "N/A", "macd_histogram": "N/A"})

            # RSI
            if len(close_prices) >= 14:
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                # Handle division by zero for rs
                rs = gain / loss if (loss != 0).all() else pd.Series([np.nan]*len(gain), index=gain.index) 
                rsi = 100 - (100 / (1 + rs))
                indicators["rsi"] = round(rsi.iloc[-1], 2) if not rsi.empty and pd.notna(rsi.iloc[-1]) else "N/A"
            else:
                indicators["rsi"] = "N/A"
            
            # Bollinger Bands
            if len(close_prices) >= 20:
                sma_20 = close_prices.rolling(window=20).mean()
                std_20 = close_prices.rolling(window=20).std()
                indicators.update({
                    "bollinger_upper": round((sma_20 + (std_20 * 2)).iloc[-1], 2) if not sma_20.empty and not std_20.empty else "N/A",
                    "bollinger_lower": round((sma_20 - (std_20 * 2)).iloc[-1], 2) if not sma_20.empty and not std_20.empty else "N/A",
                    "bollinger_width": round(((sma_20 + (std_20 * 2)) - (sma_20 - (std_20 * 2))).iloc[-1], 2) if not sma_20.empty and not std_20.empty else "N/A",
                })
            else:
                indicators.update({"bollinger_upper": "N/A", "bollinger_lower": "N/A", "bollinger_width": "N/A"})
            
            # Stochastic Oscillator
            if len(history) >= 14:
                lowest_low = low_prices.rolling(window=14).min()
                highest_high = high_prices.rolling(window=14).max()
                # Handle division by zero
                k_percent_denominator = (highest_high - lowest_low)
                k_percent = 100 * ((close_prices - lowest_low) / k_percent_denominator) if (k_percent_denominator != 0).all() else pd.Series([np.nan]*len(close_prices), index=close_prices.index)
                
                indicators["stochastic_k"] = round(k_percent.iloc[-1], 2) if not k_percent.empty and pd.notna(k_percent.iloc[-1]) else "N/A"
                indicators["stochastic_d"] = round(k_percent.rolling(window=3).mean().iloc[-1], 2) if len(k_percent) >= 3 and not k_percent.rolling(window=3).mean().empty else "N/A"
            else:
                indicators["stochastic_k"] = "N/A"
                indicators["stochastic_d"] = "N/A"
            
            # Volatility and Risk Metrics
            if len(close_prices) >= 90:
                returns = close_prices.pct_change().dropna()
                indicators.update({
                    "volatility_30d": f"{returns.rolling(window=30).std().iloc[-1] * np.sqrt(252) * 100:.2f}%" if len(returns) >= 30 and not returns.rolling(window=30).std().empty else "N/A",
                    "volatility_90d": f"{returns.rolling(window=90).std().iloc[-1] * np.sqrt(252) * 100:.2f}%" if len(returns) >= 90 and not returns.rolling(window=90).std().empty else "N/A",
                    "sharpe_ratio": round(returns.mean() / returns.std() * np.sqrt(252), 2) if returns.std() != 0 and len(returns) > 0 and pd.notna(returns.mean()) and pd.notna(returns.std()) else "N/A",
                })
            else:
                indicators.update({"volatility_30d": "N/A", "volatility_90d": "N/A", "sharpe_ratio": "N/A"})
            
            # Price relative to moving averages
            current_price = close_prices.iloc[-1] if not close_prices.empty else None
            if current_price is not None and pd.notna(current_price):
                for period in [10, 20, 50, 200]:
                    if len(close_prices) >= period:
                        sma = close_prices.rolling(window=period).mean().iloc[-1]
                        if pd.notna(sma) and sma != 0:
                            indicators[f"price_vs_sma{period}"] = f"{((current_price - sma) / sma * 100):.2f}%"
                        else:
                            indicators[f"price_vs_sma{period}"] = "N/A"
                    else:
                        indicators[f"price_vs_sma{period}"] = "N/A"
            else:
                for period in [10, 20, 50, 200]:
                    indicators[f"price_vs_sma{period}"] = "N/A"


            # Volume indicators
            if len(volume) >= 30:
                avg_volume_30d = volume.rolling(window=30).mean().iloc[-1]
                indicators.update({
                    "avg_volume_30d": int(avg_volume_30d) if pd.notna(avg_volume_30d) else "N/A",
                    "volume_ratio": round(volume.iloc[-1] / avg_volume_30d, 2) if pd.notna(avg_volume_30d) and avg_volume_30d != 0 else "N/A",
                })
            else:
                indicators.update({"avg_volume_30d": "N/A", "volume_ratio": "N/A"})
            
            return indicators
            
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
            return {}

    def create_interactive_charts(self, history: pd.DataFrame, ticker_symbol: str, 
                                  financials: Dict, technical_indicators: Dict):
        """Create interactive charts using Plotly for Streamlit"""
        if history is None or history.empty:
            st.warning(f"No data available to plot for {ticker_symbol}")
            return
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    f'{ticker_symbol} Stock Price with Moving Averages',
                    'Trading Volume',
                    'RSI (Relative Strength Index)',
                    'MACD'
                ),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.2, 0.15, 0.15]
            )
            
            # Calculate moving averages for plotting
            # Ensure enough data points before calculating
            history['SMA_20'] = history['Close'].rolling(window=20).mean() if len(history) >= 20 else pd.Series(np.nan, index=history.index)
            history['SMA_50'] = history['Close'].rolling(window=50).mean() if len(history) >= 50 else pd.Series(np.nan, index=history.index)
            history['SMA_200'] = history['Close'].rolling(window=200).mean() if len(history) >= 200 else pd.Series(np.nan, index=history.index)
            
            # Stock price with moving averages
            fig.add_trace(go.Scatter(x=history.index, y=history['Close'], 
                                     name='Close Price', line=dict(color='blue', width=2),
                                     hovertemplate='<b>Date</b>: %{x}<br><b>Close</b>: $%{y:.2f}<extra></extra>'), row=1, col=1)
            if 'SMA_20' in history.columns and not history['SMA_20'].isnull().all():
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_20'], 
                                         name='SMA 20', line=dict(color='orange', width=1),
                                         hovertemplate='<b>Date</b>: %{x}<br><b>SMA 20</b>: $%{y:.2f}<extra></extra>'), row=1, col=1)
            if 'SMA_50' in history.columns and not history['SMA_50'].isnull().all():
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_50'], 
                                         name='SMA 50', line=dict(color='green', width=1),
                                         hovertemplate='<b>Date</b>: %{x}<br><b>SMA 50</b>: $%{y:.2f}<extra></extra>'), row=1, col=1)
            if 'SMA_200' in history.columns and not history['SMA_200'].isnull().all():
                fig.add_trace(go.Scatter(x=history.index, y=history['SMA_200'], 
                                         name='SMA 200', line=dict(color='red', width=1),
                                         hovertemplate='<b>Date</b>: %{x}<br><b>SMA 200</b>: $%{y:.2f}<extra></extra>'), row=1, col=1)
            
            # Volume
            fig.add_trace(go.Bar(x=history.index, y=history['Volume'], 
                                 name='Volume', marker_color='lightblue',
                                 hovertemplate='<b>Date</b>: %{x}<br><b>Volume</b>: %{y:,.0f}<extra></extra>'), row=2, col=1)
            
            # RSI
            if technical_indicators.get("rsi") != "N/A" and history['Close'].count() >= 14: # Check data count for RSI plotting
                delta = history['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                fig.add_trace(go.Scatter(x=history.index, y=rsi, 
                                         name='RSI', line=dict(color='purple'),
                                         hovertemplate='<b>Date</b>: %{x}<br><b>RSI</b>: %{y:.2f}<extra></extra>'), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, annotation_text="Overbought (70)", annotation_position="top left")
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, annotation_text="Oversold (30)", annotation_position="bottom left")
            
            # MACD
            if technical_indicators.get("macd") != "N/A" and history['Close'].count() >= 26: # Check data count for MACD plotting
                ema_12 = history['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = history['Close'].ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                macd_histogram = macd - macd_signal
                
                fig.add_trace(go.Scatter(x=history.index, y=macd, 
                                         name='MACD Line', line=dict(color='blue'),
                                         hovertemplate='<b>Date</b>: %{x}<br><b>MACD</b>: %{y:.4f}<extra></extra>'), row=4, col=1)
                fig.add_trace(go.Scatter(x=history.index, y=macd_signal, 
                                         name='Signal Line', line=dict(color='red'),
                                         hovertemplate='<b>Date</b>: %{x}<br><b>Signal</b>: %{y:.4f}<extra></extra>'), row=4, col=1)
                fig.add_trace(go.Bar(x=history.index, y=macd_histogram, 
                                     name='Histogram', marker_color='gray',
                                     hovertemplate='<b>Date</b>: %{x}<br><b>Histogram</b>: %{y:.4f}<extra></extra>'), row=4, col=1)
            
            # Update layout
            fig.update_layout(
                title_text=f"Comprehensive Technical Analysis - {ticker_symbol}",
                height=1000,
                showlegend=True,
                template="plotly_white",
                hovermode="x unified", # Better hover experience across subplots
            )
            
            # Update X-axis labels for better readability
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider_visible=False, # Hide the default range slider for cleaner look
                type="date"
            )

            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            fig.update_yaxes(title_text="MACD", row=4, col=1)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating charts: {str(e)}")
            return None

    def generate_investment_summary(self, financials: Dict, technical_indicators: Dict) -> Dict:
        """Generate intelligent investment summary based on metrics"""
        summary = {
            "overall_rating": "NEUTRAL",
            "risk_level": "MEDIUM",
            "investment_horizon": "MEDIUM_TERM",
            "key_strengths": [],
            "key_concerns": [],
            "price_targets": {},
            "recommendations": [],
            "analyst_info": {}, # New field for analyst info
        }
        
        try:
            # Analyze financial strength
            if financials.get("profit_margin", "N/A") != "N/A":
                try:
                    margin_str = financials["profit_margin"].replace("%", "")
                    margin = float(margin_str) if margin_str != 'N/A' else None
                    if margin is not None:
                        if margin > 15:
                            summary["key_strengths"].append("Strong profit margins (over 15%)")
                        elif margin < 5:
                            summary["key_concerns"].append("Low profit margins (under 5%)")
                except ValueError:
                    pass
            
            # Analyze valuation
            pe_ratio = financials.get("pe_ratio", "N/A")
            if pe_ratio != "N/A" and isinstance(pe_ratio, (int, float)):
                if pe_ratio < 15:
                    summary["key_strengths"].append("Potentially attractive valuation (P/E < 15)")
                elif pe_ratio > 30:
                    summary["key_concerns"].append("Potentially high valuation (P/E > 30)")
            
            # Analyze technical indicators
            rsi = technical_indicators.get("rsi", "N/A")
            if rsi != "N/A" and isinstance(rsi, (int, float)):
                if rsi < 30:
                    summary["key_strengths"].append("Stock appears oversold (RSI < 30)")
                elif rsi > 70:
                    summary["key_concerns"].append("Stock appears overbought (RSI > 70)")
            
            # Debt analysis
            debt_equity = financials.get("debt_to_equity_ratio", "N/A")
            if debt_equity != "N/A" and isinstance(debt_equity, (int, float)):
                if debt_equity < 0.5:
                    summary["key_strengths"].append("Low debt levels (Debt/Equity < 0.5)")
                elif debt_equity > 1.0:
                    summary["key_concerns"].append("High debt levels (Debt/Equity > 1.0)")
            
            # Growth analysis
            revenue_growth_str = financials.get("revenue_growth", "N/A")
            if revenue_growth_str != "N/A" and revenue_growth_str != "N/A%":
                try:
                    growth = float(revenue_growth_str.replace("%", ""))
                    if growth > 10:
                        summary["key_strengths"].append("Strong revenue growth (over 10%)")
                    elif growth < 0:
                        summary["key_concerns"].append("Declining revenue")
                except ValueError:
                    pass
            
            # Dividend analysis
            dividend_yield_str = financials.get("dividend_yield", "N/A")
            if dividend_yield_str != "N/A" and dividend_yield_str != "N/A%":
                try:
                    dividend_yield = float(dividend_yield_str.replace("%", ""))
                    if dividend_yield > 2.0:
                        summary["key_strengths"].append("Attractive dividend yield")
                except ValueError:
                    pass

            # Determine overall rating based on strengths vs. concerns
            strength_count = len(summary["key_strengths"])
            concern_count = len(summary["key_concerns"])
            
            if strength_count > concern_count + 1:
                summary["overall_rating"] = "POSITIVE"
                summary["recommendations"].append("Consider a BUY, especially if it aligns with your investment strategy.")
            elif concern_count > strength_count + 1:
                summary["overall_rating"] = "NEGATIVE"
                summary["recommendations"].append("Consider holding or selling, caution advised.")
            else:
                summary["overall_rating"] = "NEUTRAL"
                summary["recommendations"].append("Monitor closely for further developments.")
            
            # Price targets (simplified, use analyst targets if available)
            target_price = financials.get("target_price", "N/A")
            if target_price != "N/A" and isinstance(target_price, (int, float)):
                summary["price_targets"] = {
                    "analyst_mean": f"${target_price:.2f}",
                    "analyst_high": f"${financials.get('target_high', 'N/A'):.2f}" if isinstance(financials.get('target_high'), (int, float)) else "N/A",
                    "analyst_low": f"${financials.get('target_low', 'N/A'):.2f}" if isinstance(financials.get('target_low'), (int, float)) else "N/A",
                }
            else:
                # Fallback to calculated targets if analyst data is not available
                current_price = financials.get("current_price", "N/A")
                if current_price != "N/A" and isinstance(current_price, (int, float)):
                    price = float(current_price)
                    summary["price_targets"] = {
                        "conservative": f"${round(price * 1.10, 2):.2f}",
                        "moderate": f"${round(price * 1.20, 2):.2f}",
                        "optimistic": f"${round(price * 1.35, 2):.2f}"
                    }
                else:
                    summary["price_targets"] = {"info": "Calculated targets unavailable without current price."}

            # Analyst Information for summary
            summary["analyst_info"] = {
                "recommendation_key": financials.get("analyst_rating", "N/A"),
                "number_of_analysts": financials.get("num_analysts", "N/A"),
                "target_mean_price": financials.get("target_price", "N/A"),
            }
            
        except Exception as e:
            st.error(f"Error generating investment summary: {str(e)}")
            
        return summary

# Initialize the analyzer
@st.cache_resource
def get_analyzer():
    return StreamlitFinancialAnalyzer()

def display_metric_card(title, value, delta=None):
    """Display a metric card in Streamlit"""
    if delta:
        st.metric(title, value, delta)
    else:
        st.metric(title, value)

def display_section_data(header_text: str, data_items: Dict, columns_per_row: int = 2):
    """
    Helper function to display a section of key-value pairs in a more appealing, column-based format.
    Only displays items that are not "N/A".
    """
    st.subheader(header_text)
    
    # Filter out N/A values and prepare for display
    displayable_items = [(key, value) for key, value in data_items.items() 
                         if value not in ["N/A", None, ""]] # More robust N/A check
    
    if not displayable_items:
        st.info("No data available for this section.")
        return

    # Create columns to arrange items
    cols = st.columns(columns_per_row)
    
    for i, (key, value) in enumerate(displayable_items):
        with cols[i % columns_per_row]:
            # Convert snake_case keys to Title Case for display
            display_key = key.replace('_', ' ').title()
            st.markdown(f"**{display_key}:** {value}")


def display_financial_data(financials: Dict, technical_indicators: Dict, investment_summary: Dict, history: pd.DataFrame, analyzer: StreamlitFinancialAnalyzer):
    ticker_input = financials.get('ticker', 'N/A')

    # Company Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.header(f"{financials.get('company_name', 'N/A')} ({ticker_input})")
        st.write(f"**Sector:** {financials.get('sector', 'N/A')} | **Industry:** {financials.get('industry', 'N/A')}")
        st.write(f"**Country:** {financials.get('country', 'N/A')}")
    with col2:
        if financials.get('website', 'N/A') != 'N/A':
            st.markdown(f"\U0001F310 [Company Website]({financials['website']})")
    with col3:
        st.write(f"**Analysis Date:** {financials.get('analysis_date', 'N/A')}")

    st.markdown("---")

    # Key Metrics Row
    st.subheader("\U0001F4CA Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        display_metric_card("Current Price", f"${financials.get('current_price', 'N/A')}")
    with col2:
        display_metric_card("Market Cap", financials.get('market_cap', 'N/A'))
    with col3:
        display_metric_card("P/E Ratio", financials.get('pe_ratio', 'N/A'))
    with col4:
        display_metric_card("Revenue (TTM)", financials.get('revenue', 'N/A'))
    with col5:
        display_metric_card("EPS (Trailing)", financials.get('EPS', 'N/A'))

    st.markdown("---")

    # Charts Section
    st.subheader("\U0001F4C8 Technical Analysis Charts")
    fig = analyzer.create_interactive_charts(history, ticker_input, financials, technical_indicators)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Replace tab-based UI with expanders
    with st.expander("\U0001F4B0 Financial Performance"):
        display_section_data("Revenue & Profitability", {
            "Revenue (TTM)": financials.get('revenue'),
            "Revenue Per Share": financials.get('revenue_per_share'),
            "Net Income": financials.get('net_income'),
            "EBITDA": financials.get('EBITDA'),
            "Gross Profit": financials.get('gross_profit'),
            "EPS (Trailing)": financials.get('EPS'),
            "EPS (Forward)": financials.get('EPS_forward'),
        })
        display_section_data("Margins & Growth", {
            "Profit Margin": financials.get('profit_margin'),
            "Operating Margin": financials.get('operating_margin'),
            "Gross Margin": financials.get('gross_margin'),
            "Return on Equity (ROE)": financials.get('return_on_equity'),
            "Return on Assets (ROA)": financials.get('return_on_assets'),
            "Revenue Growth (YOY)": financials.get('revenue_growth'),
            "Earnings Growth (YOY)": financials.get('earnings_growth'),
            "Quarterly Revenue Growth": financials.get('revenue_growth_quarterly'),
            "Quarterly Earnings Growth": financials.get('earnings_growth_quarterly'),
        })

    with st.expander("\U0001F4CB Valuation & Ratios"):
        display_section_data("Valuation Metrics", {
            "P/E Ratio (Trailing)": financials.get('pe_ratio'),
            "Forward P/E": financials.get('forward_pe'),
            "PEG Ratio": financials.get('peg_ratio'),
            "Price to Book (P/B)": financials.get('pb_ratio'),
            "Price to Sales (P/S)": financials.get('ps_ratio'),
            "Enterprise Value to Revenue": financials.get('ev_revenue'),
            "Enterprise Value to EBITDA": financials.get('ev_ebitda'),
        })
        display_section_data("Financial Ratios & Stock Info", {
            "Current Ratio": financials.get('current_ratio'),
            "Quick Ratio": financials.get('quick_ratio'),
            "Debt to Equity": financials.get('debt_to_equity_ratio'),
            "Beta": financials.get('beta'),
            "Shares Outstanding": financials.get('shares_outstanding'),
            "Float Shares": financials.get('float_shares'),
            "52 Week High": f"${financials.get('52_week_high')}",
            "52 Week Low": f"${financials.get('52_week_low')}",
        })

    with st.expander("\U0001F3E6 Balance Sheet"):
        display_section_data("Assets", {
            "Total Assets": financials.get('total_assets'),
            "Current Assets": financials.get('current_assets'),
            "Cash Reserves": financials.get('cash_reserves'),
            "Book Value": financials.get('book_value'),
        })
        display_section_data("Liabilities & Equity", {
            "Total Debt": financials.get('total_debt'),
            "Current Liabilities": financials.get('current_liabilities'),
            "Shareholders' Equity": financials.get('shareholders_equity'),
        })

    with st.expander("\U0001F4C9 Technical Indicators"):
        display_section_data("Moving Averages", {
            "SMA 10": technical_indicators.get('sma_10'),
            "SMA 20": technical_indicators.get('sma_20'),
            "SMA 50": technical_indicators.get('sma_50'),
            "SMA 200": technical_indicators.get('sma_200'),
            "EMA 12": technical_indicators.get('ema_12'),
            "EMA 26": technical_indicators.get('ema_26'),
            "Price vs. SMA 50": technical_indicators.get('price_vs_sma50'),
            "Price vs. SMA 200": technical_indicators.get('price_vs_sma200'),
        })
        display_section_data("Oscillators & Volatility", {
            "RSI": technical_indicators.get('rsi'),
            "MACD": technical_indicators.get('macd'),
            "MACD Signal": technical_indicators.get('macd_signal'),
            "MACD Histogram": technical_indicators.get('macd_histogram'),
            "Bollinger Upper Band": technical_indicators.get('bollinger_upper'),
            "Bollinger Lower Band": technical_indicators.get('bollinger_lower'),
            "Bollinger Band Width": technical_indicators.get('bollinger_width'),
            "Stochastic %K": technical_indicators.get('stochastic_k'),
            "Stochastic %D": technical_indicators.get('stochastic_d'),
            "Volatility (30D Ann.)": technical_indicators.get('volatility_30d'),
            "Volatility (90D Ann.)": technical_indicators.get('volatility_90d'),
            "Sharpe Ratio": technical_indicators.get('sharpe_ratio'),
            "Avg. Volume (30D)": technical_indicators.get('avg_volume_30d'),
            "Volume Ratio (Today/30D Avg)": technical_indicators.get('volume_ratio'),
        })

    with st.expander("\U0001F48E Dividends"):
        display_section_data("Key Dividend Metrics", {
            "Dividend Yield": financials.get('dividend_yield'),
            "Annual Dividend Rate": f"${financials.get('dividend_rate')}",
            "Payout Ratio": financials.get('payout_ratio'),
            "Ex-Dividend Date": financials.get('ex_dividend_date'),
        })

    with st.expander("\U0001F3AF Investment Summary & Analyst Views"):
        st.subheader("Investment Summary")
        if investment_summary:
            st.info(f"**Overall Rating:** {investment_summary.get('overall_rating', 'N/A')}")
            st.write(f"**Risk Level:** {investment_summary.get('risk_level', 'N/A')}")
            st.write(f"**Investment Horizon:** {investment_summary.get('investment_horizon', 'N/A')}")

            st.markdown("---")
            st.write("**Key Strengths:**")
            if investment_summary["key_strengths"]:
                for strength in investment_summary["key_strengths"]:
                    st.success(f"• {strength}")
            else:
                st.write("No major strengths identified.")

            st.write("**Key Concerns:**")
            if investment_summary["key_concerns"]:
                for concern in investment_summary["key_concerns"]:
                    st.warning(f"• {concern}")
            else:
                st.write("No major concerns identified.")

            st.markdown("---")
            st.write("**Analyst Price Targets:**")
            if investment_summary["price_targets"]:
                for target_type, price_val in investment_summary["price_targets"].items():
                    if price_val not in ["N/A", "", None]:
                        st.write(f"• {target_type.replace('_', ' ').title()}: {price_val}")
            else:
                st.write("Price targets not available.")

            st.write("**Analyst Consensus:**")
            if investment_summary['analyst_info'].get('recommendation_key', 'N/A') != 'N/A':
                st.write(f"• Recommendation Key: {investment_summary['analyst_info'].get('recommendation_key', 'N/A')}")
            if investment_summary['analyst_info'].get('number_of_analysts', 'N/A') != 'N/A':
                st.write(f"• Number of Analysts: {investment_summary['analyst_info'].get('number_of_analysts', 'N/A')}")
            if investment_summary['analyst_info'].get('target_mean_price', 'N/A') != 'N/A':
                st.write(f"• Target Mean Price: ${investment_summary['analyst_info'].get('target_mean_price', 'N/A'):.2f}")

            st.markdown("---")
            st.write("**General Recommendations:**")
            for rec in investment_summary.get("recommendations", []):
                st.write(f"• {rec}")
        else:
            st.write("Investment summary not available.")


def display_comparison_table(company1_data: Dict, company2_data: Dict):
    """
    Displays a side-by-side comparison table for two companies.
    """
    st.subheader("🆚 Company Comparison")
    st.markdown("---")

    metrics_to_compare = [
        ("Company Name", "company_name", "financials"),
        ("Ticker", "ticker", "financials"),
        ("Current Price", "current_price", "financials"),
        ("Market Cap", "market_cap", "financials"),
        ("Sector", "sector", "financials"),
        ("Industry", "industry", "financials"),
        ("P/E Ratio (Trailing)", "pe_ratio", "financials"),
        ("Forward P/E", "forward_pe", "financials"),
        ("Price to Book", "pb_ratio", "financials"),
        ("Revenue (TTM)", "revenue", "financials"),
        ("Net Income (TTM)", "net_income", "financials"),
        ("Profit Margin", "profit_margin", "financials"),
        ("Operating Margin", "operating_margin", "financials"),
        ("Revenue Growth (YOY)", "revenue_growth", "financials"),
        ("Earnings Growth (YOY)", "earnings_growth", "financials"),
        ("Return on Equity (ROE)", "return_on_equity", "financials"),
        ("Return on Assets (ROA)", "return_on_assets", "financials"),
        ("Dividend Yield", "dividend_yield", "financials"),
        ("52 Week High", "52_week_high", "financials"),
        ("52 Week Low", "52_week_low", "financials"),
        ("Beta", "beta", "financials"),
        ("RSI", "rsi", "technical_indicators"),
        ("MACD", "macd", "technical_indicators"),
        ("Analyst Rating", "analyst_rating", "financials"),
        ("Target Price (Mean)", "target_price", "financials"),
    ]

    compare_df_data = []

    company1_name = company1_data["financials"].get("company_name", company1_data["financials"].get("ticker", "Company 1"))
    company2_name = company2_data["financials"].get("company_name", company2_data["financials"].get("ticker", "Company 2"))

    for metric_name, key, source in metrics_to_compare:
        val1 = company1_data[source].get(key, "N/A")
        val2 = company2_data[source].get(key, "N/A")

        def format_value(val, key):
            try:
                if key in ["current_price", "52_week_high", "52_week_low", "target_price"]:
                    return f"${float(val):.2f}"
                elif any(k in metric_name for k in ["Yield", "Growth", "Margin"]):
                    return f"{float(val):.2f}%"
                elif key == "macd":
                    return f"{float(val):.4f}"
                else:
                    return val
            except (ValueError, TypeError):
                return val

        if key not in ["market_cap", "revenue", "net_income", "EBITDA"]:
            val1 = format_value(val1, key)
            val2 = format_value(val2, key)

        compare_df_data.append([metric_name, val1, val2])

    compare_df = pd.DataFrame(compare_df_data, columns=["Metric", company1_name, company2_name])

    st.dataframe(compare_df.style.set_properties(**{"text-align": "left"}), hide_index=True, use_container_width=True)
    st.markdown("---")
    st.info("Note: This comparison provides a snapshot. For deeper insights, analyze individual company reports.")

def main():
    st.title("📈 Financial Stock Analyzer")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Initialize session state variables if not already present
    if 'company1_financials' not in st.session_state:
        st.session_state.company1_financials = None
        st.session_state.company1_tech_indicators = None
        st.session_state.company1_history = None
        st.session_state.company1_investment_summary = None

    if 'company2_financials' not in st.session_state:
        st.session_state.company2_financials = None
        st.session_state.company2_tech_indicators = None
        st.session_state.company2_history = None
        st.session_state.company2_investment_summary = None

    if 'compare_mode' not in st.session_state:
        st.session_state.compare_mode = False

    # Input for the first ticker
    st.sidebar.markdown("### Analyze First Company")
    ticker1_input = st.sidebar.text_input(
        "Enter Ticker 1", 
        value=st.session_state.get('ticker1_value', "AAPL"), # Persist value
        key="ticker1_input_widget",
        help="Enter the stock ticker (e.g., AAPL, GOOGL, TSLA)"
    ).upper()
    st.session_state.ticker1_value = ticker1_input # Update session state on input

    analyze_button1 = st.sidebar.button("🔍 Analyze Company 1", type="primary", key="analyze_btn1")
    
    # Sample tickers for the first input
    st.sidebar.markdown("#### Or choose from samples:")
    sample_tickers1 = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
    selected_sample1 = st.sidebar.selectbox("Select Ticker 1 Sample:", [""] + sample_tickers1, key="sample_select1")
    
    if selected_sample1 and selected_sample1 != ticker1_input: # Only update if a new sample is chosen
        st.session_state.ticker1_value = selected_sample1
        ticker1_input = selected_sample1 # Update current input for immediate effect
        analyze_button1 = True # Trigger analysis
        st.session_state.compare_mode = False # Reset compare mode on new primary analysis
        st.session_state.company2_financials = None # Clear second company data

    if analyze_button1 and ticker1_input:
        analyzer = get_analyzer()
        st.session_state.company1_financials = None # Clear previous data before new analysis
        st.session_state.company1_tech_indicators = None
        st.session_state.company1_history = None
        st.session_state.company1_investment_summary = None

        st.session_state.company2_financials = None # Clear comparison data too
        st.session_state.company2_tech_indicators = None
        st.session_state.compare_mode = False # Exit compare mode if analyzing primary ticker again

        with st.spinner(f"Analyzing {ticker1_input}... Please wait..."):
            financials, timeseries, history = analyzer.get_financial_summary(ticker1_input)
            
            if financials is None:
                st.error(f"❌ Could not retrieve data for ticker: {ticker1_input}")
                st.session_state.company1_financials = None # Ensure it's explicitly None on error
            else:
                technical_indicators = analyzer.calculate_advanced_technical_indicators(history)
                investment_summary = analyzer.generate_investment_summary(financials, technical_indicators)
                
                st.session_state.company1_financials = financials
                st.session_state.company1_tech_indicators = technical_indicators
                st.session_state.company1_history = history
                st.session_state.company1_investment_summary = investment_summary
                st.success(f"✅ Analysis complete for {ticker1_input}")

    # Display results for Company 1 if available
    if st.session_state.company1_financials:
        st.markdown("---")
        st.subheader(f"Detailed Analysis for {st.session_state.company1_financials['company_name']}")
        display_financial_data(
            st.session_state.company1_financials,
            st.session_state.company1_tech_indicators,
            st.session_state.company1_investment_summary,
            st.session_state.company1_history,
            get_analyzer() # Pass analyzer instance
        )

        # "Compare" button appears after first company is analyzed
        st.markdown("---")
        if st.button("Compare with another company", key="compare_btn"):
            st.session_state.compare_mode = not st.session_state.compare_mode # Toggle compare mode
            # Clear second company data when toggling into compare mode or re-entering
            if st.session_state.compare_mode:
                st.session_state.company2_financials = None
                st.session_state.company2_tech_indicators = None

    # Comparison section
    if st.session_state.compare_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Compare with Second Company")
        ticker2_input = st.sidebar.text_input(
            "Enter Ticker 2", 
            value=st.session_state.get('ticker2_value', ""), # Persist value
            key="ticker2_input_widget",
            help="Enter the stock ticker for comparison"
        ).upper()
        st.session_state.ticker2_value = ticker2_input # Update session state on input

        analyze_button2 = st.sidebar.button("📊 Analyze Company 2 (for comparison)", type="secondary", key="analyze_btn2")

        # Sample tickers for the second input
        st.sidebar.markdown("#### Or choose from samples:")
        sample_tickers2 = ["TSLA", "NFLX", "GOOG", "AMZN", "FB", "KO"] # Different set for variety
        selected_sample2 = st.sidebar.selectbox("Select Ticker 2 Sample:", [""] + sample_tickers2, key="sample_select2")
        
        if selected_sample2 and selected_sample2 != ticker2_input:
            st.session_state.ticker2_value = selected_sample2
            ticker2_input = selected_sample2
            analyze_button2 = True

        if analyze_button2 and ticker2_input:
            if st.session_state.company1_financials and ticker2_input == st.session_state.company1_financials['ticker']:
                st.error("Please enter a different ticker symbol for comparison.")
            else:
                analyzer = get_analyzer()
                st.session_state.company2_financials = None # Clear previous data before new comparison analysis
                st.session_state.company2_tech_indicators = None

                with st.spinner(f"Analyzing {ticker2_input} for comparison..."):
                    financials2, timeseries2, history2 = analyzer.get_financial_summary(ticker2_input)
                    
                    if financials2 is None:
                        st.error(f"❌ Could not retrieve data for ticker: {ticker2_input}")
                        st.session_state.company2_financials = None
                    else:
                        technical_indicators2 = analyzer.calculate_advanced_technical_indicators(history2)
                        investment_summary2 = analyzer.generate_investment_summary(financials2, technical_indicators2)

                        st.session_state.company2_financials = financials2
                        st.session_state.company2_tech_indicators = technical_indicators2
                        st.session_state.company2_history = history2
                        st.session_state.company2_investment_summary = investment_summary2
                        st.success(f"✅ Comparison analysis complete for {ticker2_input}")

        # Display comparison table if both companies have data
        if st.session_state.company1_financials and st.session_state.company2_financials:
            st.markdown("---")
            display_comparison_table(
                {
                    "financials": st.session_state.company1_financials,
                    "technical_indicators": st.session_state.company1_tech_indicators
                },
                {
                    "financials": st.session_state.company2_financials,
                    "technical_indicators": st.session_state.company2_tech_indicators
                }
            )
            st.markdown("---")
            st.subheader(f"Detailed Analysis for {st.session_state.company2_financials['company_name']}")
            display_financial_data(
                st.session_state.company2_financials,
                st.session_state.company2_tech_indicators,
                st.session_state.company2_investment_summary,
                st.session_state.company2_history,
                get_analyzer()
            )

# Don't forget to add this at the end if it's not already there
if __name__ == "__main__":
    main()

