# Financial Chatbot with Graph Support using LangChain and Streamlit
import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from textwrap import dedent
import os
import re

# LangChain imports
try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    st.error("""
    ❌ Missing required packages. Please install them:
    
    ```bash
    pip install langchain langchain-groq plotly streamlit
    ```
    """)
    st.stop()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Your Groq API key - replace with your actual key or use environment variable
GROQ_API_KEY = ""

# Sample financial data - replace with your actual data loading logic
FINANCIAL_DATA = """
{"address1":"One Microsoft Way","city":"Redmond","state":"WA","zip":"98052-6399","country":"United States","phone":"425 882 8080","website":"https://www.microsoft.com","industry":"Software - Infrastructure","industryKey":"software-infrastructure","industryDisp":"Software - Infrastructure","sector":"Technology","sectorKey":"technology","sectorDisp":"Technology","longBusinessSummary":"Microsoft Corporation develops and supports software, services, devices and solutions worldwide. The Productivity and Business Processes segment offers office, exchange, SharePoint, Microsoft Teams, office 365 Security and Compliance, Microsoft viva, and Microsoft 365 copilot; and office consumer services, such as Microsoft 365 consumer subscriptions, Office licensed on-premises, and other office services. This segment also provides LinkedIn; and dynamics business solutions, including Dynamics 365, a set of intelligent, cloud-based applications across ERP, CRM, power apps, and power automate; and on-premises ERP and CRM applications. The Intelligent Cloud segment offers server products and cloud services, such as azure and other cloud services; SQL and windows server, visual studio, system center, and related client access licenses, as well as nuance and GitHub; and enterprise services including enterprise support services, industry solutions, and nuance professional services. The More Personal Computing segment offers Windows, including windows OEM licensing and other non-volume licensing of the Windows operating system; Windows commercial comprising volume licensing of the Windows operating system, windows cloud services, and other Windows commercial offerings; patent licensing; and windows Internet of Things; and devices, such as surface, HoloLens, and PC accessories. Additionally, this segment provides gaming, which includes Xbox hardware and content, and first- and third-party content; Xbox game pass and other subscriptions, cloud gaming, advertising, third-party disc royalties, and other cloud services; and search and news advertising, which includes Bing, Microsoft News and Edge, and third-party affiliates. The company sells its products through OEMs, distributors, and resellers; and directly through digital marketplaces, online, and retail stores. The company was founded in 1975 and is headquartered in Redmond, Washington.","fullTimeEmployees":228000,"companyOfficers":[{"maxAge":1,"name":"Mr. Satya  Nadella","age":57,"title":"Chairman & CEO","yearBorn":1967,"fiscalYear":2024,"totalPay":7869791,"exercisedValue":0,"unexercisedValue":0},{"maxAge":1,"name":"Mr. Bradford L. Smith LCA","age":65,"title":"President & Vice Chairman","yearBorn":1959,"fiscalYear":2024,"totalPay":4755618,"exercisedValue":0,"unexercisedValue":0},{"maxAge":1,"name":"Ms. Amy E. Hood","age":52,"title":"Executive VP & CFO","yearBorn":1972,"fiscalYear":2024,"totalPay":4704250,"exercisedValue":0,"unexercisedValue":0},{"maxAge":1,"name":"Mr. Judson B. Althoff","age":50,"title":"Executive VP & Chief Commercial Officer","yearBorn":1974,"fiscalYear":2024,"totalPay":4534974,"exercisedValue":0,"unexercisedValue":0},{"maxAge":1,"name":"Ms. Carolina  Dybeck Happe","age":52,"title":"Executive VP & COO","yearBorn":1972,"fiscalYear":2024,"exercisedValue":0,"unexercisedValue":0},{"maxAge":1,"name":"Ms. Alice L. Jolla","age":58,"title":"Corporate VP & Chief Accounting Officer","yearBorn":1966,"fiscalYear":2024,"exercisedValue":0,"unexercisedValue":0},{"maxAge":1,"name":"Jonathan  Neilson","title":"Vice President of Investor Relations","fiscalYear":2024,"exercisedValue":0,"unexercisedValue":0},{"maxAge":1,"name":"Mr. Hossein  Nowbar","title":"Chief Legal Officer","fiscalYear":2024,"exercisedValue":0,"unexercisedValue":0},{"maxAge":1,"name":"Mr. Frank X. Shaw","title":"Chief Communications Officer","fiscalYear":2024,"exercisedValue":0,"unexercisedValue":0},{"maxAge":1,"name":"Mr. Takeshi  Numoto","age":53,"title":"Executive VP & Chief Marketing Officer","yearBorn":1971,"fiscalYear":2024,"exercisedValue":0,"unexercisedValue":0}],"auditRisk":9,"boardRisk":5,"compensationRisk":4,"shareHolderRightsRisk":2,"overallRisk":3,"governanceEpochDate":1748736000,"compensationAsOfEpochDate":1735603200,"irWebsite":"http://www.microsoft.com/investor/default.aspx","executiveTeam":[],"maxAge":86400,"priceHint":2,"previousClose":490.11,"open":491.61,"dayLow":491.61,"dayHigh":494.5556,"regularMarketPreviousClose":490.11,"regularMarketOpen":491.61,"regularMarketDayLow":491.61,"regularMarketDayHigh":494.5556,"dividendRate":3.32,"dividendYield":0.68,"exDividendDate":1755734400,"payoutRatio":0.2442,"fiveYearAvgDividendYield":0.83,"beta":1.026,"trailingPE":38.0711,"forwardPE":32.95251,"volume":4768440,"regularMarketVolume":4768440,"averageVolume":22997875,"averageVolume10days":20084620,"averageDailyVolume10Day":20084620,"bid":467.03,"ask":492.89,"bidSize":1,"askSize":1,"marketCap":3661566574592,"fiftyTwoWeekLow":344.79,"fiftyTwoWeekHigh":494.5556,"priceToSalesTrailing12Months":13.560855,"fiftyDayAverage":439.2026,"twoHundredDayAverage":421.6553,"trailingAnnualDividendRate":3.24,"trailingAnnualDividendYield":0.006610761,"currency":"USD","tradeable":false,"enterpriseValue":3668165001216,"profitMargins":0.35789,"floatShares":7422063978,"sharesOutstanding":7432540160,"sharesShort":58182215,"sharesShortPriorMonth":50313676,"sharesShortPreviousMonthDate":1745971200,"dateShortInterest":1748563200,"sharesPercentSharesOut":0.0078,"heldPercentInsiders":0.00062,"heldPercentInstitutions":0.74691004,"shortRatio":2.49,"shortPercentOfFloat":0.0078,"impliedSharesOutstanding":7432540160,"bookValue":43.3,"priceToBook":11.377368,"lastFiscalYearEnd":1719705600,"nextFiscalYearEnd":1751241600,"mostRecentQuarter":1743379200,"earningsQuarterlyGrowth":0.177,"netIncomeToCommon":96635002880,"trailingEps":12.94,"forwardEps":14.95,"lastSplitFactor":"2:1","lastSplitDate":1045526400,"enterpriseToRevenue":13.585,"enterpriseToEbitda":24.59,"52WeekChange":0.08393049,"SandP52WeekChange":0.11213791,"lastDividendValue":0.83,"lastDividendDate":1747267200,"quoteType":"EQUITY","currentPrice":492.64,"targetHighPrice":700,"targetLowPrice":432,"targetMeanPrice":518.4428,"targetMedianPrice":507.5,"recommendationMean":1.43333,"recommendationKey":"strong_buy","numberOfAnalystOpinions":50,"totalCash":79617998848,"totalCashPerShare":10.712,"ebitda":149172994048,"totalDebt":105018998784,"quickRatio":1.244,"currentRatio":1.372,"totalRevenue":270010007552,"debtToEquity":32.626,"revenuePerShare":36.325,"returnOnAssets":0.14581999,"returnOnEquity":0.3361,"grossProfits":186509000704,"freeCashflow":54817001472,"operatingCashflow":130710003712,"earningsGrowth":0.177,"revenueGrowth":0.133,"grossMargins":0.69074994,"ebitdaMargins":0.55247,"operatingMargins":0.45671,"financialCurrency":"USD","symbol":"MSFT","language":"en-US","region":"US","typeDisp":"Equity","quoteSourceName":"Nasdaq Real Time Price","triggerable":true,"customPriceAlertConfidence":"HIGH","corporateActions":[],"regularMarketTime":1750863745,"exchange":"NMS","messageBoardId":"finmb_21835","exchangeTimezoneName":"America/New_York","exchangeTimezoneShortName":"EDT","gmtOffSetMilliseconds":-14400000,"market":"us_market","esgPopulated":false,"regularMarketChangePercent":0.51621664,"regularMarketPrice":492.64,"shortName":"Microsoft Corporation","longName":"Microsoft Corporation","cryptoTradeable":false,"marketState":"REGULAR","hasPrePostMarketData":true,"firstTradeDateMilliseconds":511108200000,"regularMarketChange":2.5300293,"regularMarketDayRange":"491.61 - 494.5556","fullExchangeName":"NasdaqGS","averageDailyVolume3Month":22997875,"fiftyTwoWeekLowChange":147.85,"fiftyTwoWeekLowChangePercent":0.42881176,"fiftyTwoWeekRange":"344.79 - 494.5556","fiftyTwoWeekHighChange":-1.9155884,"fiftyTwoWeekHighChangePercent":-0.003873353,"fiftyTwoWeekChangePercent":8.393049,"dividendDate":1757548800,"earningsTimestamp":1746043200,"earningsTimestampStart":1753732800,"earningsTimestampEnd":1754078400,"earningsCallTimestampStart":1746048600,"earningsCallTimestampEnd":1746048600,"isEarningsDateEstimate":true,"epsTrailingTwelveMonths":12.94,"epsForward":14.95,"epsCurrentYear":13.44374,"priceEpsCurrentYear":36.644566,"fiftyDayAverageChange":53.43741,"fiftyDayAverageChangePercent":0.12166915,"twoHundredDayAverageChange":70.98471,"twoHundredDayAverageChangePercent":0.16834773,"sourceInterval":15,"exchangeDataDelayedBy":0,"averageAnalystRating":"1.4 - Strong Buy","displayName":"Microsoft","trailingPegRatio":2.2987}
"""

# =============================================================================
# LANGCHAIN SETUP
# =============================================================================

def create_llm():
    """Initialize the LangChain LLM"""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.5
    )

def create_system_prompt():
    """Create the system prompt for the financial assistant"""
    return f"""You are a highly intelligent Financial Analysis Assistant for TechCorp India.

Your job is to analyze the provided financial data and respond to user questions with both text explanations and graph data when appropriate.

IMPORTANT: You must respond in a specific JSON format that allows for both text and graph responses.

Financial Data Available:
{FINANCIAL_DATA}

RESPONSE FORMAT RULES:
1. Always respond with valid JSON in this exact structure:
```json
{{
    "content": [
        {{
            "type": "text",
            "data": "Your markdown analysis here"
        }},
        {{
            "type": "graph",
            "data": {{
                "chart_type": "line|bar|pie|scatter",
                "title": "Chart Title",
                "x_data": ["Item1", "Item2", "Item3"],
                "y_data": [100, 150, 200],
                "labels": ["Series Name"],
                "x_label": "X-Axis Label",
                "y_label": "Y-Axis Label"
            }}
        }}
    ]
}}
```

2. Chart Types Available:
   - "line": For trends over time (revenue, stock prices)
   - "bar": For comparisons (quarterly data, divisions)
   - "pie": For breakdowns (expenses, market share)
   - "scatter": For correlations

3. When to include graphs:
   - User asks for trends, charts, or visualizations
   - Comparing multiple periods or categories
   - Showing breakdowns or distributions
   - Any financial analysis that would benefit from visualization

4. Text Analysis Guidelines:
   - Use markdown formatting
   - Provide context before and after graphs
   - Include key insights and conclusions
   - Use bullet points for key findings

5. Data Constraints:
   - Only use the provided financial data
   - If information isn't available, say so clearly
   - Make reasonable calculations based on available data

EXAMPLES:

For "Show me revenue trends":
```json
{{
    "content": [
        {{
            "type": "text",
            "data": "## TechCorp India Revenue Analysis\\n\\nHere's the quarterly revenue performance showing strong growth:"
        }},
        {{
            "type": "graph",
            "data": {{
                "chart_type": "line",
                "title": "TechCorp India Quarterly Revenue Growth",
                "x_data": ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
                "y_data": [120, 135, 150, 175],
                "labels": ["Revenue"],
                "x_label": "Quarter",
                "y_label": "Revenue ($ Millions)"
            }}
        }},
        {{
            "type": "text",
            "data": "**Key Insights:**\\n- Consistent growth throughout 2024\\n- 46% total growth from Q1 to Q4\\n- Strongest performance in Q4 with $175M"
        }}
    ]
}}
```

Remember: Always respond with valid JSON. No additional text outside the JSON structure."""

def get_ai_response(user_question, chat_history=None):
    """Get response from the AI model"""
    try:
        llm = create_llm()
        
        messages = [
            SystemMessage(content=create_system_prompt()),
            HumanMessage(content=user_question)
        ]
        
        # Add chat history if available
        if chat_history:
            # Add recent chat history for context
            for msg in chat_history[-4:]:  # Last 4 messages for context
                if msg["role"] == "user":
                    messages.insert(-1, HumanMessage(content=msg["content"]))
        
        response = llm(messages)
        return response.content
        
    except Exception as e:
        return json.dumps({
            "content": [{
                "type": "text",
                "data": f"I encountered an error while processing your request: {str(e)}"
            }]
        })

# =============================================================================
# CHART CREATION FUNCTIONS
# =============================================================================

def create_chart(graph_data):
    """Create a Plotly chart based on graph data"""
    try:
        chart_type = graph_data.get("chart_type", "bar")
        title = graph_data.get("title", "Chart")
        x_data = graph_data.get("x_data", [])
        y_data = graph_data.get("y_data", [])
        labels = graph_data.get("labels", ["Data"])
        x_label = graph_data.get("x_label", "")
        y_label = graph_data.get("y_label", "")
        
        if chart_type == "line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_data, 
                y=y_data, 
                mode='lines+markers',
                name=labels[0] if labels else "Data",
                line=dict(width=3, color='#667eea'),
                marker=dict(size=8, color='#667eea')
            ))
            
        elif chart_type == "bar":
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x_data, 
                y=y_data, 
                name=labels[0] if labels else "Data",
                marker_color='#667eea',
                text=y_data,
                textposition='auto',
            ))
            
        elif chart_type == "pie":
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=x_data, 
                values=y_data,
                hole=0.3,  # Donut style
                textinfo='label+percent+value',
                textfont_size=12,
            ))
            
        elif chart_type == "scatter":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_data, 
                y=y_data, 
                mode='markers',
                name=labels[0] if labels else "Data",
                marker=dict(
                    size=12,
                    color='#667eea',
                    opacity=0.7,
                    line=dict(width=2, color='#4c63d2')
                )
            ))
        else:
            # Default to bar chart
            fig = go.Figure(data=go.Bar(x=x_data, y=y_data))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title, 
                x=0.5, 
                font=dict(size=18, color='#333')
            ),
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_white",
            height=450,
            margin=dict(l=60, r=60, t=80, b=60),
            font=dict(size=12),
            showlegend=len(labels) > 1,
            hovermode='closest'
        )
        
        # Special formatting for pie charts
        if chart_type == "pie":
            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01
                )
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def parse_ai_response(response_text):
    """Parse the AI response and extract content blocks"""
    try:
        # Clean the response text
        response_text = response_text.strip()
        
        # Try to extract JSON from response if it's wrapped in other text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group()
        
        # Parse JSON
        response_data = json.loads(response_text)
        return response_data.get("content", [])
        
    except json.JSONDecodeError as e:
        # Fallback: treat as plain text
        return [{
            "type": "text", 
            "data": f"**Response:** {response_text}\n\n*Note: Response was not in expected JSON format*"
        }]
    except Exception as e:
        return [{
            "type": "text", 
            "data": f"Error parsing response: {str(e)}"
        }]

# =============================================================================
# STREAMLIT UI
# =============================================================================

# Page config
st.set_page_config(
    page_title="TechCorp Financial Assistant",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        padding-top: 20px;
        padding-bottom: 20px;
    }
    
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    
    div.stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        z-index: 1000;
    }
    
    [data-testid="chat-message-container"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin-left: auto;
        margin-right: auto;
        max-width: 800px;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #e0e9ff;
        border-radius: 18px 18px 2px 18px;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #f0f2f6;
        border-radius: 18px 18px 18px 2px;
    }
    
    /* Hide Streamlit elements */
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    
    /* Input styling */
    div.stChatInputContainer input {
        border-radius: 20px;
        border: 1px solid #c9ccd0;
        padding: 10px 15px;
        box-shadow: none;
    }
    
    div.stChatInputContainer input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Button styling */
    div.stChatInputContainer button {
        background-color: #667eea;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    div.stChatInputContainer button:hover {
        background-color: #556cdc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# App header
st.title("📊 TechCorp India Financial Assistant")
st.markdown("""
<div style='text-align: center; color: gray; margin-bottom: 30px;'>
    Ask me about financial data, trends, and get interactive charts and analysis!
</div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            # Display assistant's structured response
            content_blocks = message.get("content_blocks", [])
            for block in content_blocks:
                if block["type"] == "text":
                    st.markdown(block["data"])
                elif block["type"] == "graph":
                    fig = create_chart(block["data"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

# Typing indicator
if st.session_state.is_generating:
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial data and creating visualizations..."):
            pass

# Chat input
user_input = st.chat_input(
    "Ask about revenue trends, create charts, analyze expenses...",
    disabled=st.session_state.is_generating
)

# Handle user input
if user_input and not st.session_state.is_generating:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    st.session_state.is_generating = True
    st.rerun()

# Process AI response
if st.session_state.is_generating and st.session_state.messages:
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user":
        try:
            # Get AI response
            response = get_ai_response(
                last_message["content"], 
                st.session_state.messages[:-1]
            )
            
            # Parse structured response
            content_blocks = parse_ai_response(response)
            
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content_blocks": content_blocks
            })
            
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content_blocks": [{
                    "type": "text",
                    "data": f"**Error:** I encountered an issue processing your request: {str(e)}"
                }]
            })
        
        st.session_state.is_generating = False
        st.rerun()

# Clear chat button
if st.session_state.messages:
    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.is_generating = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Sample questions for new users
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 Trend Analysis:**
        - "Show me revenue trends over quarters"
        - "Create a line chart of stock prices"
        - "Display profit growth as a graph"
        """)
    
    with col2:
        st.markdown("""
        **📊 Comparisons & Breakdowns:**
        - "Compare revenue by division"
        - "Show expense breakdown as pie chart"
        - "Create bar chart of quarterly profits"
        """)
    
    st.markdown("""
    **🔍 General Analysis:**
    - "What are the key financial insights?"
    - "Analyze the profit margins"
    - "Show me the financial performance summary"
    """)