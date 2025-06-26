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
