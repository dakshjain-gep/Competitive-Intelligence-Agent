import streamlit as st
import requests
from updates import (
    CompetitorAnalyzer,
    display_company_info,
    compare_multiple_tickers,
    display_comparison_results
)
from financial_chatbot import render_financial_chatbot

# Page configuration
st.set_page_config(page_title="🧠 CI + Financial Analyzer", layout="wide")
analyzer = CompetitorAnalyzer()

# Title
st.title("🧠 Competitive Intelligence Brief + Financial Analyzer")

# --- Initialize Session State ---
defaults = {
    "data_fetched": False,
    "company_financial_data": None,
    "company_ticker": None,
    "competitor_tickers": [],
    "company_name": None,
    "ci_markdown": None,
    "comparison_results": None,
    "comparison_done": False,
    "show_chatbot": False, # Moved here for clarity
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# --- User Input ---
company_name = st.text_input("Enter Company Name", placeholder="e.g. Tesla")

if st.button("🚀 Generate CI + Financial Report"):
    if not company_name.strip():
        st.warning("⚠️ Please enter a valid company name.")
    else:
        with st.spinner("⏳ Generating Report..."):
            st.session_state.company_name = company_name
            # Reset data when a new company is searched
            st.session_state.data_fetched = False
            st.session_state.company_financial_data = None
            st.session_state.company_ticker = None
            st.session_state.competitor_tickers = []
            st.session_state.ci_markdown = None
            st.session_state.comparison_results = None
            st.session_state.comparison_done = False
            st.session_state.show_chatbot = False

            fallback_data = {
                "ticker": "TSLA",
                "competitors": ["AAPL", "GOOGL", "AMZN"],
                "markdown": f"# Competitive Intelligence Brief for {company_name}\n\n*API unavailable - using demo content*"
            }

            # --- Fetch CI Data ---
            try:
                response = requests.post(
                    "http://127.0.0.1:8001/chat",
                    json={"message": company_name},
                    timeout=30
                )
                response.raise_for_status()
                ci_data = response.json().get("reply", {})

                if not ci_data or not isinstance(ci_data, dict):
                    raise ValueError("Invalid CI data received.")

                st.session_state.ci_markdown = ci_data.get("markdown", fallback_data["markdown"])
                st.session_state.company_ticker = ci_data.get("companyticker", fallback_data["ticker"])
                st.session_state.competitor_tickers = ci_data.get("competitortickers", fallback_data["competitors"])

            except Exception as e:
                st.error(f"❌ CI API Error: {str(e)}")
                st.session_state.ci_markdown = fallback_data["markdown"]
                st.session_state.company_ticker = fallback_data["ticker"]
                st.session_state.competitor_tickers = fallback_data["competitors"]

            # --- Fetch Financial Summary ---
            try:
                summary = analyzer.get_summary(st.session_state.company_ticker)
                if not summary:
                    raise ValueError(f"No data found for {st.session_state.company_ticker}")

                st.session_state.company_financial_data = summary
                st.session_state.data_fetched = True
                st.success("✅ Financial data fetched successfully!")

            except Exception as e:
                st.error(f"⚠️ Financial data error: {str(e)}")
                st.session_state.data_fetched = False

# --- Display Tabs ---
if st.session_state.data_fetched and st.session_state.company_financial_data:
    tabs = st.tabs(["📄 CI Report", "📊 Financial Analysis", "🏆 Competitor Comparison"])

    # -- Tab 1: CI Report --
    with tabs[0]:
        st.subheader("📄 Competitive Intelligence Brief")
        st.info(f"CI Report for **{st.session_state.company_name}** ({st.session_state.company_ticker})")
        st.markdown(st.session_state.ci_markdown, unsafe_allow_html=True)

    # -- Tab 2: Financial Analysis --
    with tabs[1]:
        st.subheader("💼 Financial Snapshot")
        display_company_info(st.session_state.company_financial_data)

        st.subheader("📈 Price Chart & Technical Analysis")
        try:
            analyzer.plot_chart(
                st.session_state.company_financial_data["history"],
                st.session_state.company_ticker
            )
        except Exception as e:
            st.error(f"Chart rendering error: {str(e)}")

    # -- Tab 3: Competitor Comparison --
    with tabs[2]:
        st.subheader("🏆 Competitive Landscape Analysis")
        all_tickers = [st.session_state.company_ticker] + st.session_state.competitor_tickers

        st.info(f"📊 Comparing {st.session_state.company_name} with competitors")
        st.markdown(f"**Companies to compare:** {', '.join(all_tickers)}")

        if st.session_state.comparison_results:
            st.subheader("📋 Comparison Results")
            display_comparison_results(st.session_state.comparison_results)

        if st.button("🚀 Compare with All Competitors", key="auto_compare"):
            with st.spinner("⏳ Comparing companies..."):
                try:
                    results = compare_multiple_tickers(st.session_state, all_tickers)
                    st.session_state.comparison_results = results
                    display_comparison_results(results)
                except Exception as e:
                    st.error(f"Comparison error: {str(e)}")
            

# AI Assistant in Sidebar
with st.sidebar:
    st.header("🤖 AI Financial Assistant")
    # Only show the toggle button if data is fetched
    if st.session_state.data_fetched and st.session_state.company_financial_data:
        if st.button("Toggle AI Assistant", key="ai_toggle_sidebar", help="Show/Hide AI chatbot"):
            st.session_state.show_chatbot = not st.session_state.show_chatbot

        if st.session_state.show_chatbot:
            st.success("🤖 AI Assistant is active!")
            
            # Prepare data for chatbot (only CI and Financials)
            ci_data_for_chatbot = {
                "ci": st.session_state.get("ci_markdown"),
                "financials": st.session_state.get("company_financial_data"),
                # We are explicitly *not* sending comparison data unless explicitly requested by the user,
                # as per your requirement "we have to only send the financials and the CI report as data"
                # If you later want to include comparison data when available, you can add it here conditionally.
            }
            render_financial_chatbot(ci_data_for_chatbot)
        else:
            st.info("Click 'Toggle AI Assistant' to activate the chatbot.")
    else:
        st.info("Generate a CI + Financial Report first to enable the AI Assistant.")


# --- Reset Option ---
if st.session_state.data_fetched:
    st.markdown("---")
    if st.button("🔄 Clear Data & Start Over", key="clear_data_main"):
        for key in defaults:
            if key in st.session_state: # Safely remove only if it exists
                del st.session_state[key]
        st.rerun()