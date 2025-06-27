import streamlit as st
import yfinance as yf
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
from financial_chatbot import render_financial_chatbot


# Enhanced page configuration
st.set_page_config(
    page_title="📊 Competitor Stock Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': None,
        'About': "# Stock Competitor Analyzer\nAnalyze and compare stocks with comprehensive metrics and visualizations."
    }
)

class CompetitorAnalyzer:
    def __init__(self):
        self.current_date = datetime.now()
        # Fetch data for the last 5 years
        self.start_date = (self.current_date - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
        self.end_date = self.current_date.strftime("%Y-%m-%d")

    @st.cache_data(ttl=3600) # Cache data for 1 hour to reduce API calls
    def get_summary(_self, ticker: str) -> Optional[Dict]:
        """
        Fetches a summary of stock information and historical data for a given ticker.
        Includes common financial metrics, technical indicators (RSI, MACD).
        """
        try:
            with st.spinner(f'Fetching data for {ticker}...'):
                t = yf.Ticker(ticker)
                info = t.info
                # st.write("📦 Raw Info:", info)

                hist = t.history(start=_self.start_date, end=_self.end_date)
                # st.write("📈 Historical Data:", hist)s


                if hist.empty:
                    st.warning(f"⚠️ No data found for {ticker}. Please check the ticker symbol.")
                    return None

                # Helper function to format large numbers (Billions, Millions)
                def format_num(x):
                    if x is None: return "N/A"
                    try:
                        return f"${x / 1e9:.2f}B" if x >= 1e9 else f"${x / 1e6:.2f}M"
                    except: return "N/A"

                # Helper function to format percentages
                def pct(x):
                    try: return f"{x * 100:.2f}%" if x is not None else "N/A"
                    except: return "N/A"

                return {
                    "company_name": info.get("longName", "N/A"),
                    "ticker": ticker.upper(),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "current_price": round(hist["Close"].iloc[-1], 2),
                    "market_cap": format_num(info.get("marketCap")),
                    "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
                    "forward_pe": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else "N/A",
                    "pb_ratio": round(info.get("priceToBook", 0), 2) if info.get("priceToBook") else "N/A",
                    "revenue": format_num(info.get("totalRevenue")),
                    "net_income": format_num(info.get("netIncomeToCommon")),
                    "profit_margin": pct(info.get("profitMargins")),
                    "operating_margin": pct(info.get("operatingMargins")),
                    "revenue_growth": pct(info.get("revenueGrowth")),
                    "earnings_growth": pct(info.get("earningsGrowth")),
                    "return_on_equity": pct(info.get("returnOnEquity")),
                    "return_on_assets": pct(info.get("returnOnAssets")),
                    "beta": round(info.get("beta", 0), 2) if info.get("beta") else "N/A",
                    "rsi": _self.get_rsi(hist), # Call instance method for RSI
                    "macd": _self.get_macd(hist), # Call instance method for MACD
                    "analyst_rating": info.get("recommendationKey", "N/A").replace("_", " ").title(),
                    "target_price": round(info.get("targetMeanPrice", 0), 2) if info.get("targetMeanPrice") else "N/A",
                    "history": hist
                }
        except Exception as e:
            st.error(f"❌ Error fetching data for {ticker}: {str(e)}. Please try again or check the ticker symbol.")
            return None

    def get_rsi(self, hist: pd.DataFrame) -> str:
        """Calculates the Relative Strength Index (RSI)."""
        try:
            delta = hist["Close"].diff()
            # Calculate gains (upward changes) and losses (downward changes)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return f"{rsi.iloc[-1]:.2f}"
        except:
            return "N/A"

    def get_macd(self, hist: pd.DataFrame) -> str:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        try:
            # Calculate 12-period and 26-period Exponential Moving Averages
            ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
            ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26 # MACD line
            return f"{macd.iloc[-1]:.2f}"
        except:
            return "N/A"

    def plot_chart(self, hist: pd.DataFrame, ticker: str):
        """Plots the stock price performance with moving averages."""
        fig = go.Figure()

        # Add price line
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['Close'],
            name='Close Price',
            line=dict(color='#667eea', width=2), # Streamlit's default blues work well
            fill='tozeroy', # Fill to zero for a subtle area chart effect
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))

        # Add moving averages
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()

        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['MA20'],
            name='20-Day MA',
            line=dict(color='#2d3436', width=1, dash='dash') # Darker color for MA
        ))

        fig.add_trace(go.Scatter(
            x=hist.index,
            y=hist['MA50'],
            name='50-Day MA',
            line=dict(color='#e17055', width=1, dash='dot') # A reddish color for the second MA
        ))

        fig.update_layout(
            title=f"📈 **{ticker}** - Stock Price Performance",
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_dark', # Use Streamlit's built-in theme for consistency
            showlegend=True,
            height=500,
            xaxis_rangeslider_visible=True # Add a range slider for easier navigation
        )

        st.plotly_chart(fig, use_container_width=True)
def compare_multiple_tickers(self, tickers_list):
        """
        Fetches data for multiple company tickers and returns the data without displaying.
        
        Args:
            tickers_list: List of ticker symbols (strings) to compare
            
        Returns:
            dict: {
                'companies_data': List of company data dictionaries,
                'failed_tickers': List of tickers that failed to fetch,
                'success_count': Number of successfully fetched companies,
                'total_count': Total number of tickers requested
            }
            
        Example:
            result = analyzer.compare_multiple_tickers(['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
            if result['companies_data']:
                # Process the data
                display_multi_comparison(result['companies_data'])
        """
        # Validation
        if not tickers_list or len(tickers_list) < 2:
            return {
                'companies_data': [],
                'failed_tickers': tickers_list if tickers_list else [],
                'success_count': 0,
                'total_count': len(tickers_list) if tickers_list else 0,
                'error': "Please provide at least 2 ticker symbols for comparison."
            }
        
        companies_data = []
        failed_tickers = []
        
        # Fetch data for each ticker
        for ticker in tickers_list:
            try:
                # Use the existing get_summary method
                company_data = self.get_summary(ticker)
                
                if company_data:
                    companies_data.append(company_data)
                else:
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                failed_tickers.append(ticker)
        
        # Return structured data
        return {
            'companies_data': companies_data,
            'failed_tickers': failed_tickers,
            'success_count': len(companies_data),
            'total_count': len(tickers_list),
            'error': None
        }


def format_num(value):
    try:
        return f"${float(value):,.2f}"
    except:
        return "N/A"

def pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except:
        return "N/A"

def display_comparison_results(comparison_result, show_download=True):
    """
    Display the results from compare_multiple_tickers with UI elements.
    """
    if not comparison_result or not isinstance(comparison_result, dict):
        return

    result = comparison_result

    # Handle errors
    if result.get('error'):
        st.warning(f"⚠️ {result['error']}")
        return

    if result.get('failed_tickers'):
        st.warning(f"⚠️ Could not fetch data for: {', '.join(result['failed_tickers'])}")

    if result.get('companies_data'):
        st.success(f"✅ Successfully fetched data for {result['success_count']} out of {result['total_count']} companies!")

        # Display comparison (assumes display_multi_comparison is defined)
        from updates import display_multi_comparison
        display_multi_comparison(result['companies_data'])

        if show_download:
            try:
                all_data = []
                expected_keys = [
                    'company_name', 'ticker', 'sector', 'industry',
                    'current_price', 'market_cap', 'pe_ratio', 'forward_pe', 'pb_ratio',
                    'totalRevenue', 'netIncomeToCommon', 'profitMargins', 'operatingMargins',
                    'revenueGrowth', 'earningsGrowth', 'returnOnEquity', 'returnOnAssets',
                    'beta', 'recommendationKey', 'targetMeanPrice', 'marketCap'
                ]

                for company in result['companies_data']:
                    if not isinstance(company, dict):
                        st.warning("⚠️ Skipped invalid company entry.")
                        continue

                    for key in expected_keys:
                        if key not in company:
                            company[key] = 'N/A'

                    row = {
                        'Company Name': company['company_name'],
                        'Ticker': company['ticker'],
                        'Sector': company['sector'],
                        'Industry': company['industry'],
                        'Current Price': company['current_price'],
                        'Market Cap': format_num(company.get("marketCap")),
                        'P/E Ratio': company['pe_ratio'],
                        'Forward P/E': company['forward_pe'],
                        'P/B Ratio': company['pb_ratio'],
                        'Revenue': company['totalRevenue'],
                        'Net Income': company['netIncomeToCommon'],
                        'Profit Margin': pct(company.get("profitMargins")),
                        'Operating Margin': pct(company.get("operatingMargins")),
                        'Revenue Growth': pct(company.get("revenueGrowth")),
                        'Earnings Growth': pct(company.get("earningsGrowth")),
                        'ROE': pct(company.get("returnOnEquity")),
                        'ROA': pct(company.get("returnOnAssets")),
                        'Beta': company['beta'],
                        'Analyst Rating': company['recommendationKey'],
                        'Target Price': company['targetMeanPrice'],
                    }

                    all_data.append(row)

                df_download = pd.DataFrame(all_data)
                csv = df_download.to_csv(index=False)

                tickers_str = '_'.join([company.get('ticker', 'UNK') for company in result['companies_data'] if isinstance(company, dict)])
                st.download_button(
                    label="📥 Download Comparison Data as CSV",
                    data=csv,
                    file_name=f"multi_company_comparison_{tickers_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_{hash(tickers_str)}"
                )

            except Exception as e:
                st.error(f"❌ Error creating download file: {str(e)}")

    else:
        st.error("❌ No valid company data received. Check tickers or try again.")



def get_sentiment_icon(value: str) -> str:
    """Returns an emoji icon based on a string value's inferred sentiment."""
    if value == "N/A":
        return ""
    try:
        # For percentages, check if positive or negative
        if "%" in value:
            num_val = float(value.replace("%", "").strip())
            if num_val > 0: return "⬆️"
            if num_val < 0: return "⬇️"
        # For numeric values (e.g., RSI for overbought/oversold)
        elif value.replace('.', '', 1).isdigit(): # Check if it's a number
            num_val = float(value)
            if 30 <= num_val <= 70: return "👍" # Good RSI range
            else: return "⚠️" # Potentially overbought/oversold
    except ValueError:
        pass # Not a number, or not a percentage
    return "" # Default neutral
def display_multi_comparison(companies_list):
    """
    Displays a beautiful multi-company comparison similar to the 2-company version.
    
    Args:
        companies_list: List of company dictionaries (each containing financial data)
    """
    if not companies_list or len(companies_list) < 2:
        st.warning("⚠️ Please provide at least 2 companies for comparison.")
        return
    
    # Beautiful header with company count
    num_companies = len(companies_list)
    st.subheader(f"🏆 Multi-Company Comparison ({num_companies} Companies)")
    st.info(f"📊 Comparing key metrics across {num_companies} selected companies.")
    
    # Create tabs for different comparison views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "💰 Financial Metrics", "📈 Technical Analysis"])

    with tab1:
        st.markdown("### 🎯 Company Overview & Valuation")
        
        metrics = [
            ("Company Name", "company_name"),
            ("Ticker", "ticker"),
            ("Sector", "sector"),
            ("Industry", "industry"),
            ("Current Price", "current_price"),
            ("Market Cap", "market_cap"),
            ("P/E Ratio", "pe_ratio"),
            ("Forward P/E", "forward_pe"),
            ("Target Price", "target_price"),
        ]

        comparison_data = []
        for label, key in metrics:
            row = [label]  # Start with metric name
            
            # Get value for each company
            for company in companies_list:
                value = company.get(key, "N/A")
                # Format current price with $ symbol
                if key == "current_price" and value != "N/A":
                    value = f"${value}"
                elif key == "target_price" and value != "N/A" and value != 0:
                    value = f"${value}"
                row.append(value)
            
            comparison_data.append(row)

        # Create column names: Metric + Company names
        columns = ["Metric"] + [f"{comp['company_name']}" for comp in companies_list]
        df_overview = pd.DataFrame(comparison_data, columns=columns)
        
        # Beautiful dataframe display with styling
        st.dataframe(
            df_overview, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("📋 Metric", width="medium"),
            }
        )

    with tab2:
        st.markdown("### 💹 Financial Performance & Profitability")
        
        metrics = [
            ("Revenue", "revenue"),
            ("Net Income", "net_income"),
            ("Profit Margin", "profit_margin"),
            ("Operating Margin", "operating_margin"),
            ("Revenue Growth", "revenue_growth"),
            ("Earnings Growth", "earnings_growth"),
            ("Return on Equity", "return_on_equity"),
            ("Return on Assets", "return_on_assets"),
            ("P/B Ratio", "pb_ratio"),
        ]

        comparison_data = []
        for label, key in metrics:
            row = [label]
            for company in companies_list:
                value = company.get(key, "N/A")
                row.append(value)
            comparison_data.append(row)

        columns = ["Metric"] + [f"{comp['company_name']}" for comp in companies_list]
        df_financial = pd.DataFrame(comparison_data, columns=columns)
        
        st.dataframe(
            df_financial, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("💰 Financial Metric", width="medium"),
            }
        )

    with tab3:
        st.markdown("### 📈 Technical Analysis & Market Sentiment")
        
        metrics = [
            ("Beta", "beta"),
            ("RSI (14-day)", "rsi"),
            ("MACD", "macd"),
            ("Analyst Rating", "analyst_rating"),
        ]

        comparison_data = []
        for label, key in metrics:
            row = [label]
            for company in companies_list:
                value = company.get(key, "N/A")
                row.append(value)
            comparison_data.append(row)

        columns = ["Metric"] + [f"{comp['company_name']}" for comp in companies_list]
        df_technical = pd.DataFrame(comparison_data, columns=columns)
        
        st.dataframe(
            df_technical, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("📊 Technical Metric", width="medium"),
            }
        )
    
    

def display_company_info(company: Dict):
    """Displays detailed information for a single company."""
    st.header(f"🏢 {company['company_name']} ({company['ticker']})")
    st.markdown(f"**Sector:** {company['sector']} | **Industry:** {company['industry']}")

    st.markdown("---") # Visual separator

    # Key metrics in columns using st.metric for a clean look
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="💰 Current Price", value=f"${company['current_price']:.2f}")
        st.metric(label="📊 Market Cap", value=company['market_cap'])

    with col2:
        st.metric(label="📈 P/E Ratio", value=company['pe_ratio'])
        st.metric(label="🔮 Forward P/E", value=company['forward_pe'])

    with col3:
        st.metric(label="💼 Revenue", value=company['revenue'])
        st.metric(label="💸 Net Income", value=company['net_income'])

    with col4:
        st.metric(label="📊 Beta", value=company['beta'])
        st.metric(label="🎯 Target Price", value=f"${company['target_price']}" if company['target_price'] != "N/A" else "N/A")

    # Additional metrics in expandable sections
    st.markdown("---")
    with st.expander("📊 Detailed Financial Metrics", expanded=False):
        col_profit, col_growth = st.columns(2)

        with col_profit:
            st.subheader("💹 Profitability")
            st.markdown(f"**Profit Margin:** {company['profit_margin']} {get_sentiment_icon(company['profit_margin'])}")
            st.markdown(f"**Operating Margin:** {company['operating_margin']} {get_sentiment_icon(company['operating_margin'])}")
            st.markdown(f"**P/B Ratio:** {company['pb_ratio']}")

        with col_growth:
            st.subheader("📈 Growth & Returns")
            st.markdown(f"**Revenue Growth:** {company['revenue_growth']} {get_sentiment_icon(company['revenue_growth'])}")
            st.markdown(f"**Earnings Growth:** {company['earnings_growth']} {get_sentiment_icon(company['earnings_growth'])}")
            st.markdown(f"**ROE:** {company['return_on_equity']} {get_sentiment_icon(company['return_on_equity'])}")
            st.markdown(f"**ROA:** {company['return_on_assets']} {get_sentiment_icon(company['return_on_assets'])}")

    with st.expander("⚡ Technical Analysis & Outlook", expanded=False):
        col_tech1, col_tech2, col_tech3 = st.columns(3)

        with col_tech1:
            st.markdown(f"**RSI (14):** {company['rsi']} {get_sentiment_icon(company['rsi'])}")

        with col_tech2:
            st.markdown(f"**MACD:** {company['macd']}")

        with col_tech3:
            # Simple color based on analyst rating
            rating = company['analyst_rating'].lower()
            if "buy" in rating:
                st.markdown(f"**Analyst Rating:** 👍 {company['analyst_rating']}")
            elif "sell" in rating:
                st.markdown(f"**Analyst Rating:** 👎 {company['analyst_rating']}")
            else:
                st.markdown(f"**Analyst Rating:** neutral {company['analyst_rating']}")
import pandas as pd
from datetime import datetime
import streamlit as st

def compare_multiple_tickers(self, tickers_list):
    """
    Fetches data for multiple company tickers and displays a comprehensive comparison.
    
    Args:
        tickers_list: List of ticker symbols (strings) to compare
        
    Example:
        compare_multiple_tickers(['AAPL', 'GOOGL', 'MSFT', 'TSLA'])
    """
    if not tickers_list or len(tickers_list) < 2:
        st.warning("⚠️ Please provide at least 2 ticker symbols for comparison.")
        return
    
    st.info(f"🔄 Fetching data for {len(tickers_list)} companies: {', '.join(tickers_list)}")
    analyzer = CompetitorAnalyzer()
    
    # Progress bar for better UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    companies_data = []
    failed_tickers = []
    
    # Fetch data for each ticker
    for i, ticker in enumerate(tickers_list):
        try:
            status_text.text(f"Fetching data for {ticker}... ({i+1}/{len(tickers_list)})")
            progress_bar.progress((i + 1) / len(tickers_list))
            
            # Use the existing get_summary method
            company_data = analyzer.get_summary(ticker)
            
            if company_data:
                companies_data.append(company_data)
            else:
                failed_tickers.append(ticker)
                
        except Exception as e:
            st.error(f"❌ Failed to fetch data for {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if companies_data:
        if failed_tickers:
            st.warning(f"⚠️ Could not fetch data for the following tickers: {', '.join(failed_tickers)}")
        
        st.success(f"✅ Successfully fetched data for {len(companies_data)} companies!")
        
        # Display the comparison using the existing function
        display_multi_comparison(companies_data)
        
        # Create download data
        try:
            # Create a comprehensive DataFrame for download
            all_data = []
            for company in companies_data:
                company_row = {
                    'Company Name': company.get('company_name', 'N/A'),
                    'Ticker': company.get('ticker', 'N/A'),
                    'Sector': company.get('sector', 'N/A'),
                    'Industry': company.get('industry', 'N/A'),
                    'Current Price': company.get('current_price', 'N/A'),
                    'Market Cap': company.get('market_cap', 'N/A'),
                    'P/E Ratio': company.get('pe_ratio', 'N/A'),
                    'Forward P/E': company.get('forward_pe', 'N/A'),
                    'P/B Ratio': company.get('pb_ratio', 'N/A'),
                    'Revenue': company.get('revenue', 'N/A'),
                    'Net Income': company.get('net_income', 'N/A'),
                    'Profit Margin': company.get('profit_margin', 'N/A'),
                    'Operating Margin': company.get('operating_margin', 'N/A'),
                    'Revenue Growth': company.get('revenue_growth', 'N/A'),
                    'Earnings Growth': company.get('earnings_growth', 'N/A'),
                    'ROE': company.get('return_on_equity', 'N/A'),
                    'ROA': company.get('return_on_assets', 'N/A'),
                    'Beta': company.get('beta', 'N/A'),
                    'Analyst Rating': company.get('analyst_rating', 'N/A'),
                    'Target Price': company.get('target_price', 'N/A'),
                }
                all_data.append(company_row)
            
            df_download = pd.DataFrame(all_data)
            csv = df_download.to_csv(index=False)
            
            # Fixed: Download button outside of conditional logic
            st.download_button(
                label="📥 Download Comparison Data as CSV",
                data=csv,
                file_name=f"multi_company_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_{hash(str(tickers_list))}"  # Unique key to avoid conflicts
            )
            
        except Exception as e:
            st.error(f"❌ Error creating download file: {str(e)}")
    
    else:
        st.error("❌ Could not fetch data for any of the provided tickers. Please check the ticker symbols and try again.")


def display_comparison(company1: Dict, company2: Dict):
    """Displays a side-by-side comparison of two companies."""
    st.subheader("🆚 Head-to-Head Comparison")
    st.info("Compare key metrics between the two selected companies.")

    # Create tabs for different comparison views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "💰 Financial Metrics", "📈 Technical Analysis"])

    with tab1:
        metrics = [
            ("Company Name", "company_name"),
            ("Ticker", "ticker"),
            ("Sector", "sector"),
            ("Industry", "industry"),
            ("Current Price", "current_price"),
            ("Market Cap", "market_cap"),
            ("P/E Ratio", "pe_ratio"),
            ("Forward P/E", "forward_pe"),
            ("Target Price", "target_price"),
        ]

        comparison_data = []
        for label, key in metrics:
            val1 = company1.get(key, "N/A")
            val2 = company2.get(key, "N/A")
            comparison_data.append([label, val1, val2])

        df_overview = pd.DataFrame(comparison_data, columns=["Metric", company1["company_name"], company2["company_name"]])
        st.dataframe(df_overview, use_container_width=True)

    with tab2:
        metrics = [
            ("Revenue", "revenue"),
            ("Net Income", "net_income"),
            ("Profit Margin", "profit_margin"),
            ("Operating Margin", "operating_margin"),
            ("Revenue Growth", "revenue_growth"),
            ("Earnings Growth", "earnings_growth"),
            ("Return on Equity", "return_on_equity"),
            ("Return on Assets", "return_on_assets"),
            ("P/B Ratio", "pb_ratio"),
        ]

        comparison_data = []
        for label, key in metrics:
            val1 = company1.get(key, "N/A")
            val2 = company2.get(key, "N/A")
            comparison_data.append([label, val1, val2])

        df_financial = pd.DataFrame(comparison_data, columns=["Metric", company1["company_name"], company2["company_name"]])
        st.dataframe(df_financial, use_container_width=True)

    with tab3:
        metrics = [
            ("Beta", "beta"),
            ("RSI", "rsi"),
            ("MACD", "macd"),
            ("Analyst Rating", "analyst_rating"),
        ]

        comparison_data = []
        for label, key in metrics:
            val1 = company1.get(key, "N/A")
            val2 = company2.get(key, "N/A")
            comparison_data.append([label, val1, val2])

        df_technical = pd.DataFrame(comparison_data, columns=["Metric", company1["company_name"], company2["company_name"]])
        st.dataframe(df_technical, use_container_width=True)
def main():
    st.title("📈 Financial Stock Analyzer")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Settings")

    # Initialize session state
    session_vars = [
        'company1_data', 'company2_data',
        'show_chatbot', 'show_chatbot_comparison', 'compare_mode',
        'ticker1_value', 'ticker2_value'
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None if 'data' in var else False

    analyzer = CompetitorAnalyzer()

    # Primary Company Input
    st.sidebar.markdown("### Analyze First Company")
    ticker1_input = st.sidebar.text_input(
        "Enter Ticker 1",
        value=st.session_state.get('ticker1_value', 'AAPL'),
        key="ticker1_input_widget"
    ).upper()
    st.session_state.ticker1_value = ticker1_input

    analyze_button1 = st.sidebar.button("🔍 Analyze Company 1", key="analyze_btn1")

    sample_tickers1 = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
    selected_sample1 = st.sidebar.selectbox("Or choose sample ticker:", [""] + sample_tickers1, key="sample1")
    if selected_sample1 and selected_sample1 != ticker1_input:
        st.session_state.ticker1_value = selected_sample1
        ticker1_input = selected_sample1
        analyze_button1 = True
        st.session_state.compare_mode = False
        st.session_state.company2_data = None

    if analyze_button1 and ticker1_input:
        with st.spinner(f"Analyzing {ticker1_input}..."):
            data = analyzer.get_summary(ticker1_input)
            if data:
                st.session_state.company1_data = data
                st.success(f"✅ Analysis complete for {ticker1_input}")
            else:
                st.error(f"❌ Could not fetch data for {ticker1_input}")
                st.session_state.company1_data = None

    if st.session_state.company1_data:
        st.markdown("---")
        display_company_info(st.session_state.company1_data)

        st.markdown("---")
        st.subheader("📈 Price Chart & Technical Analysis")
        analyzer.plot_chart(st.session_state.company1_data["history"], ticker1_input)

        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Compare with another company", key="compare_btn"):
                st.session_state.compare_mode = not st.session_state.compare_mode
                st.session_state.show_chatbot = False
                st.session_state.show_chatbot_comparison = False
                st.session_state.company2_data = None

        with col2:
            if st.button("🤖 Ask AI Assistant", key="chatbot_btn", disabled=st.session_state.compare_mode):
                st.session_state.show_chatbot = not st.session_state.show_chatbot

    # Comparison Mode
    if st.session_state.compare_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Compare with Second Company")

        ticker2_input = st.sidebar.text_input(
            "Enter Ticker 2",
            value=st.session_state.get('ticker2_value', ""),
            key="ticker2_input_widget"
        ).upper()
        st.session_state.ticker2_value = ticker2_input

        analyze_button2 = st.sidebar.button("📊 Analyze Company 2", key="analyze_btn2")

        sample_tickers2 = ["TSLA", "NFLX", "GOOG", "AMZN", "META", "KO"]
        selected_sample2 = st.sidebar.selectbox("Or choose sample ticker:", [""] + sample_tickers2, key="sample2")
        if selected_sample2 and selected_sample2 != ticker2_input:
            st.session_state.ticker2_value = selected_sample2
            ticker2_input = selected_sample2
            analyze_button2 = True

        if analyze_button2 and ticker2_input:
            if st.session_state.company1_data and ticker2_input == st.session_state.company1_data["ticker"]:
                st.error("Please enter a different ticker symbol for comparison.")
            else:
                with st.spinner(f"Analyzing {ticker2_input}..."):
                    data2 = analyzer.get_summary(ticker2_input)
                    if data2:
                        st.session_state.company2_data = data2
                        st.success(f"✅ Analysis complete for {ticker2_input}")
                    else:
                        st.error(f"❌ Could not fetch data for {ticker2_input}")
                        st.session_state.company2_data = None

        if st.session_state.company1_data and st.session_state.company2_data:
            st.markdown("---")
            st.subheader(f"Detailed Analysis for {st.session_state.company2_data['company_name']}")
            display_company_info(st.session_state.company2_data)

            st.markdown("---")
            st.subheader("📈 Competitor Price Chart")
            analyzer.plot_chart(st.session_state.company2_data["history"], ticker2_input)

            st.markdown("---")
            display_comparison(st.session_state.company1_data, st.session_state.company2_data)

            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🤖 Ask AI Assistant (Compare)", key="chatbot_comparison_btn"):
                    st.session_state.show_chatbot_comparison = True

    # Functional AI Chatbot — Single Company
    if st.session_state.show_chatbot and st.session_state.company1_data and not st.session_state.compare_mode:
        company1_data = {
            'financials': st.session_state.company1_data,
        }
        render_financial_chatbot(company1_data, None)

    # Functional AI Chatbot — Comparison Mode
    if st.session_state.show_chatbot_comparison and st.session_state.company1_data and st.session_state.company2_data:
        company1_data = {'financials': st.session_state.company1_data}
        company2_data = {'financials': st.session_state.company2_data}
        render_financial_chatbot(company1_data, company2_data)

if __name__ == "__main__":
    main()
