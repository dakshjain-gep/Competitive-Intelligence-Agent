import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import tempfile
from fpdf import FPDF
from io import BytesIO

# ---------- Streamlit Dark Theme Config ----------
st.set_page_config(page_title="🧠 CI Brief Generator", layout="wide", initial_sidebar_state="collapsed")

# ---------- Matplotlib Dark Style ----------
mplstyle.use('dark_background')
plt.rcParams.update({
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#161b22'
})

def plot_bar(df, column, title, ylabel, color):
    if column in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        df_sorted = df.sort_values(by=column, ascending=False)
        bars = ax.bar(df_sorted["ticker"], df_sorted[column], color=color, edgecolor='white')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, color='white')

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Ticker")
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)

# def generate_pdf_from_markdown(markdown_text):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)
    
#     # Split markdown by lines and write each line
#     for line in markdown_text.split('\n'):
#         if line.strip().startswith('#'):
#             # Simple handling for markdown headers
#             header_level = line.count('#')
#             text = line.replace('#', '').strip()
#             pdf.set_font("Arial", 'B', max(16 - (header_level * 2), 10))
#             pdf.cell(0, 10, text, ln=True)
#             pdf.set_font("Arial", size=12)  # Reset back
#         else:
#             pdf.multi_cell(0, 10, line)

#     # Save PDF to BytesIO stream
#     pdf_stream = BytesIO()
#     pdf.output(pdf_stream)
#     pdf_stream.seek(0)
#     return pdf_stream

st.title("🧠 Competitive Intelligence Brief Generator")

company_name = st.text_input("Enter Company Name", placeholder="e.g. Tesla")

if st.button("🚀 Generate CI Brief"):
    if not company_name.strip():
        st.warning("⚠️ Please enter a valid company name.")
    else:
        with st.spinner("Thinking Hard To Generate The Best Report..."):

            try:
                chat_response = requests.post(
                    "http://127.0.0.1:8001/chat",
                    json={"message": company_name}
                )
                
                print(chat_response.json()["reply"])
                

                if chat_response.status_code == 200:
                    response_json = chat_response.json()["reply"]
                    if isinstance(response_json, (str, type(None))):
                        st.warning("Are you sure this company exists?")
                    else:   
                        markdown_text = response_json["markdown"]
                        company_ticker = response_json["companyticker"]
                        competitor_tickers = response_json["competitortickers"]

                        tabs = st.tabs(["📄 CI Report", "📊 Financials"])

                        # ---------- CI Report Tab ----------
                        with tabs[0]:
                            st.markdown(markdown_text, unsafe_allow_html=True)

                            # Download Button
                            # pdf_file = generate_pdf_from_markdown(markdown_text)

                            # st.download_button(
                            #     label="📥 Download CI Brief as PDF",
                            #     data=pdf_file,
                            #     file_name=f"{company_name}_CI_Brief.pdf",
                            #     mime="application/pdf"
                            # )

                        # ---------- Financials Tab ----------
                        with tabs[1]:
                            finance_response = requests.post(
                                "http://127.0.0.1:8000/finance",
                                json={
                                    "companyticker": company_ticker,
                                    "competitortickers": competitor_tickers
                                }
                            )

                            if finance_response.status_code == 200:
                                finance_data = finance_response.json()["snapshots"]
                                df = pd.DataFrame(finance_data)

                                # Summary Card (Company Only)
                                main_company = df[df['ticker'] == company_ticker]

                                if not main_company.empty:
                                    st.markdown(f"### 📌 Key Financials for {company_ticker}")

                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("💰 Market Cap (B USD)", f"{main_company.iloc[0]['market_cap'] / 1e9:.2f}" if pd.notna(main_company.iloc[0]['market_cap']) else "N/A")
                                    col2.metric("📈 Revenue (B USD)", f"{main_company.iloc[0]['revenue'] / 1e9:.2f}" if pd.notna(main_company.iloc[0]['revenue']) else "N/A")
                                    col3.metric("📊 PE Ratio", f"{main_company.iloc[0]['pe_ratio']:.2f}" if pd.notna(main_company.iloc[0]['pe_ratio']) else "N/A")

                                    col1.metric("💵 Net Income (B USD)", f"{main_company.iloc[0]['net_income'] / 1e9:.2f}" if pd.notna(main_company.iloc[0]['net_income']) else "N/A")
                                    col2.metric("💸 Current Price (USD)", f"{main_company.iloc[0]['price']:.2f}" if pd.notna(main_company.iloc[0]['price']) else "N/A")

                                st.divider()
                                st.markdown("### 📊 Company vs Competitors Charts")

                                # Graphs for Market Cap, Revenue, PE Ratio, Net Income, Price
                                plot_bar(df, "market_cap", "💰 Market Cap (B USD)", "Market Cap (B USD)", '#58a6ff')
                                plot_bar(df, "revenue", "📈 Revenue (B USD)", "Revenue (B USD)", '#3fb950')
                                plot_bar(df, "pe_ratio", "📊 PE Ratio", "PE Ratio", '#f2cc60')
                                plot_bar(df, "net_income", "💵 Net Income (B USD)", "Net Income (B USD)", '#d62728')
                                plot_bar(df, "price", "💸 Current Price (USD)", "Price (USD)", '#9467bd')

                            else:
                                st.error(f"❌ Error from /finance API: {finance_response.status_code}")
                                st.text(finance_response.text)

            except Exception as e:
                st.error("❌ Backend error during CI Brief generation or financials fetching.")
                st.exception(e)
