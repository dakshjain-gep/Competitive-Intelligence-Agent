import streamlit as st
import requests
from updates import CompetitorAnalyzer, display_company_info, display_comparison
from financial_chatbot import render_financial_chatbot  # Assuming it's already implemented

st.set_page_config(page_title="🧠 CI + Financial Analyzer", layout="wide")
analyzer = CompetitorAnalyzer()

st.title("🧠 Competitive Intelligence Brief + Financial Analyzer")

# User Input
company_name = st.text_input("Enter Company Name", placeholder="e.g. Tesla")
if st.button("🚀 Generate CI + Financial Report"):

    if not company_name.strip():
        st.warning("⚠️ Please enter a valid company name.")
    else:
        with st.spinner("⏳ Generating Report..."):
            try:
                # Fetch CI Brief
                ci_response = requests.post("http://127.0.0.1:8001/chat", json={"message": company_name})
                print(ci_response.json())
                ci_data = ci_response.json()["reply"]

                if not ci_data or not isinstance(ci_data, dict):
                    st.error("❌ No CI data found.")
                else:
                    markdown_text = ci_data["markdown"]
                    company_ticker = ci_data["companyticker"]
                    competitor_tickers = ci_data["competitortickers"]
                    # Show CI Report
                    tabs = st.tabs(["📄 CI Report", "📊 Financial Analysis"])
                    with tabs[0]:
                        st.subheader("📄 Competitive Intelligence Brief")
                        st.markdown(markdown_text, unsafe_allow_html=True)

                    # Fetch Financial Summary
                    with tabs[1]:
                        financial_data = analyzer.get_summary(company_ticker)
                        if financial_data:
                            st.subheader("💼 Financial Snapshot")
                            display_company_info(financial_data)
                            st.subheader("📈 Price Chart & Technical Analysis")
                            analyzer.plot_chart(financial_data["history"], company_ticker)


                            # Store in session
                            st.session_state["company1_data"] = financial_data
                            st.session_state["ticker1_value"] = company_ticker
                            st.session_state["company1_name"] = financial_data["company_name"]
                            st.session_state["competitor_tickers"] = competitor_tickers
                            
                        else:
                            st.error("⚠️ Could not fetch financial data.")
            except Exception as e:
                st.exception(e)
