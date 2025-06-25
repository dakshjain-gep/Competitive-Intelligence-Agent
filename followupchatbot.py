import streamlit as st
import os
from dotenv import load_dotenv
from textwrap import dedent
from agno.agent import Agent
from agno.models.groq import Groq

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file.")
    st.stop()

# --- Embed Company Data ---
data = dedent("""
Company: TechCorp India
Financial Performance:
2020: Revenue = ₹500 Cr, Profit = ₹50 Cr
2021: Revenue = ₹750 Cr, Profit = ₹80 Cr
2022: Revenue = ₹900 Cr, Profit = ₹100 Cr
2023: Revenue = ₹1200 Cr, Profit = ₹130 Cr

Key Ratios:
EBITDA Margin (2020–2023): 15.0% → 18.0%
ROE: 18.5%
Debt-to-Equity Ratio: 0.3
Current Ratio: 2.1

Operational Highlights:
- Industry: IT Services
- Employees: 15,000
- Market Cap: ₹25,000 Cr
- HQ: Mumbai, India

Strategic Moves:
- Launched AI division in 2023
- Acquired 2 cybersecurity startups
- Expanded into 5 new countries
- Increased R&D spending by 40%
""")

financial_qa_agent = Agent(
    model=Groq(id="llama3-8b-8192", api_key="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"),

    description=dedent("""\
        You are a highly intelligent and context-aware Competitive Intelligence Assistant.

        You are part of a project that analyzes companies using preprocessed data from public sources like news articles,
        financial filings, and blogs. Your job is to act as a follow-up chatbot, answering user questions based on this data.

        The data is injected dynamically as a string variable named `data`. You must use that data as the only source of truth.
    """),

    instructions=dedent(f"""\
        You are given a variable named `data` containing structured insights.

        This is the data you will use:

        ```
        {data}
        ```

        --- ✅ Rules ---

        1. Use **only** the above data to answer questions.
           - If a question can't be answered from it, say:
             > "I'm sorry, I can't answer that based on the current dataset."

        2. Identify what the user wants (summary, SWOT, trends, etc.) and search `data` for relevant entries.

        3. Format all responses in **Markdown**:
           - Use headings (`##`) and bullet points
           - Use tables when comparing companies or metrics
           - Clearly label SWOT sections

        4. Explain calculations if data supports them.

        5. If chat history is available, maintain context.
    """),

    expected_output=dedent("""\
        ### ✅ Sample Outputs (in Markdown):

        **Q: Summary of Company X's Q1 earnings**

        ```markdown
        ## Company X – Q1 Earnings Summary (2025-03-15)

        - Reported 20% YoY growth
        - Driven by cloud service sales
        - Published in Economic Times
        ```

        **Q: SWOT for Company X**

        ```markdown
        ## SWOT Analysis – Company X

        **Strengths**
        - Strong cloud performance
        - Expanding R&D

        **Weaknesses**
        - Limited presence in Latin America

        **Opportunities**
        - Expansion in Asia
        - AI integration

        **Threats**
        - Regulatory pressure in Europe
        - Rising competitors
        ```

        **Q: CEO background?**

        ```markdown
        I'm sorry, I can't answer that based on the current dataset.
        ```
    """),

    markdown=True
)

# --- Page Config ---
st.set_page_config(page_title="🧠 Financial Q&A Agent", layout="centered")

# --- Styling ---
st.markdown("""
    <style>
    .chat-box {
        border-radius: 10px;
        padding: 12px;
        margin: 10px 0;
        width: fit-content;
        max-width: 80%;
        word-wrap: break-word;
        font-size: 1rem;
        line-height: 1.5;
    }
    .user-msg {
        background-color: #d1eaff;
        margin-left: auto;
        color: black;
    }
    .bot-msg {
        background-color: #eeeeee;
        margin-right: auto;
        color: black;
    }
    .scroll-anchor {
        height: 1px;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🧠 TechCorp India Financial Chatbot")
st.caption("Ask questions based on TechCorp India's dataset.")

# --- Initialize Chat State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Form ---
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Your question:", placeholder="e.g., What's the CAGR from 2020 to 2023?")
    submitted = st.form_submit_button("Send")

# --- Process Input ---
if submitted and user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.spinner("🤖 Thinking..."):
        response = financial_qa_agent.run(user_input)
    st.session_state.chat_history.append(("Agent", response.content))


# --- Display Chat History ---
st.markdown("## 💬 Chat")

for speaker, msg in st.session_state.chat_history:
    role_class = "user-msg" if speaker == "You" else "bot-msg"

    if speaker == "You":
        # Display user messages normally with HTML formatting
        st.markdown(
            f'<div class="chat-box {role_class}"><b>{speaker}:</b><br>{msg}</div>',
            unsafe_allow_html=True
        )
    else:
        # If it's a bot (agent), render the message as markdown in a box
        st.markdown(f'<div class="chat-box {role_class}"><b>{speaker}:</b></div>', unsafe_allow_html=True)
        st.markdown(msg)  # Render the actual message as Markdown (supports ## headers, lists, etc.)


# --- Auto-scroll (Streamlit-safe way) ---
st.markdown('<div class="scroll-anchor" id="end-of-chat"></div>', unsafe_allow_html=True)
st.markdown("""
    <script>
        const element = document.getElementById("end-of-chat");
        if (element) {
            element.scrollIntoView({ behavior: "smooth" });
        }
    </script>
""", unsafe_allow_html=True)


# --- Auto-scroll Anchor ---
scroll_anchor = st.empty()
scroll_anchor.markdown("""
    <div id="chat_end"></div>
    <script>
        const chatEnd = window.parent.document.getElementById("chat_end");
        if(chatEnd) {
            chatEnd.scrollIntoView({behavior: "smooth"});
        }
    </script>
""", unsafe_allow_html=True)

