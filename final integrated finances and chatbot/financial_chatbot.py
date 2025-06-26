# financial_chatbot.py
import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
import re
import os

# LangChain imports
try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "put your-groq-api-key-here")

# =============================================================================
# DATA FORMATTING FUNCTIONS
# =============================================================================

def format_financial_data_for_chatbot(company1_data, company2_data=None):
    """Format financial analysis data for the chatbot"""
    formatted_data = {}
    
    # Format Company 1 data
    if company1_data:
        financials = company1_data.get('financials', {})
        tech_indicators = company1_data.get('tech_indicators', {})
        investment_summary = company1_data.get('investment_summary', {})
        
        formatted_data['primary_company'] = {
            'company_name': financials.get('company_name', 'N/A'),
            'ticker': financials.get('ticker', 'N/A'),
            'sector': financials.get('sector', 'N/A'),
            'industry': financials.get('industry', 'N/A'),
            'current_price': financials.get('current_price', 'N/A'),
            'market_cap': financials.get('market_cap', 'N/A'),
            'revenue': financials.get('revenue', 'N/A'),
            'net_income': financials.get('net_income', 'N/A'),
            'total_debt': financials.get('total_debt', 'N/A'),
            'total_cash': financials.get('total_cash', 'N/A'),
            'pe_ratio': financials.get('pe_ratio', 'N/A'),
            'pb_ratio': financials.get('pb_ratio', 'N/A'),
            'roe': financials.get('roe', 'N/A'),
            'debt_to_equity': financials.get('debt_to_equity', 'N/A'),
            'current_ratio': financials.get('current_ratio', 'N/A'),
            'technical_indicators': tech_indicators,
            'investment_summary': investment_summary
        }
    
    # Format Company 2 data (if available)
    if company2_data:
        financials2 = company2_data.get('financials', {})
        tech_indicators2 = company2_data.get('tech_indicators', {})
        investment_summary2 = company2_data.get('investment_summary', {})
        
        formatted_data['comparison_company'] = {
            'company_name': financials2.get('company_name', 'N/A'),
            'ticker': financials2.get('ticker', 'N/A'),
            'sector': financials2.get('sector', 'N/A'),
            'industry': financials2.get('industry', 'N/A'),
            'current_price': financials2.get('current_price', 'N/A'),
            'market_cap': financials2.get('market_cap', 'N/A'),
            'revenue': financials2.get('revenue', 'N/A'),
            'net_income': financials2.get('net_income', 'N/A'),
            'total_debt': financials2.get('total_debt', 'N/A'),
            'total_cash': financials2.get('total_cash', 'N/A'),
            'pe_ratio': financials2.get('pe_ratio', 'N/A'),
            'pb_ratio': financials2.get('pb_ratio', 'N/A'),
            'roe': financials2.get('roe', 'N/A'),
            'debt_to_equity': financials2.get('debt_to_equity', 'N/A'),
            'current_ratio': financials2.get('current_ratio', 'N/A'),
            'technical_indicators': tech_indicators2,
            'investment_summary': investment_summary2
        }
    
    return formatted_data

# =============================================================================
# LANGCHAIN FUNCTIONS
# =============================================================================
def create_llm():
    """Initialize the LangChain LLM"""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.7
    )

def create_system_prompt(financial_data):
    """Create the system prompt for the financial assistant"""
    return f"""You are a highly intelligent Financial Analysis Assistant for TechCorp India.

Your job is to analyze the provided financial data and respond to user questions with both text explanations and graph data when appropriate.

IMPORTANT: You must respond in a specific JSON format that allows for both text and graph responses.

Financial Data Available:
{financial_data}

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

def get_ai_response(user_question,financial_data, chat_history=None):
    """Get response from the AI model"""
    try:
        llm = create_llm()
        
        messages = [
            SystemMessage(content=create_system_prompt(financial_data)),
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
# CHATBOT UI FUNCTION
# =============================================================================

def render_financial_chatbot(company1_data, company2_data=None):
    """Render the financial chatbot interface"""
    
    # Initialize chatbot session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    
    # Format data for chatbot
    financial_data = format_financial_data_for_chatbot(company1_data, company2_data)
    
    # Chatbot header
    st.markdown("---")
    st.subheader("🤖 AI Financial Assistant")
    st.markdown("Ask questions about the analyzed financial data and get interactive visualizations!")
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
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
        "Ask about financial metrics, create charts, compare companies...",
        disabled=st.session_state.is_generating
    )
    
    # Handle user input
    if user_input and not st.session_state.is_generating:
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.is_generating = True
        st.rerun()
    
    # Process AI response
    if st.session_state.is_generating and st.session_state.chat_messages:
        last_message = st.session_state.chat_messages[-1]
        if last_message["role"] == "user":
            try:
                # Get AI response
                response = get_ai_response(
                    last_message["content"], 
                    financial_data,
                    st.session_state.chat_messages[:-1]
                )
                
                # Parse structured response
                content_blocks = parse_ai_response(response)
                
                # Add assistant response
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content_blocks": content_blocks
                })
                
            except Exception as e:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content_blocks": [{
                        "type": "text",
                        "data": f"**Error:** I encountered an issue processing your request: {str(e)}"
                    }]
                })
            
            st.session_state.is_generating = False
            st.rerun()
    
    # Clear chat button
    if st.session_state.chat_messages:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.chat_messages = []
                st.session_state.is_generating = False
                st.rerun()
    
    # Sample questions for new users
    if not st.session_state.chat_messages:
        st.markdown("### 💡 Try asking:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📈 Analysis Questions:**
            - "What are the key financial insights?"
            - "Show me a chart of financial ratios"
            - "Compare revenue vs expenses"
            """)
        
        with col2:
            st.markdown("""
            **📊 Visualization Requests:**
            - "Create a pie chart of asset breakdown"
            - "Show profit margins as a bar chart"
            - "Display technical indicators"
            """)