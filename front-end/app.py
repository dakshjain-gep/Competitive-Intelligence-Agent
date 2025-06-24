import streamlit as st
import requests

# Streamlit page config
st.set_page_config(page_title="CI Chatbot", layout="centered")
st.title("💬 Competitive Intelligence Chatbot")

API_URL = "http://127.0.0.1:8001/chat"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask about a company...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Make API request
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(API_URL, json={"message": user_input})
                if response.status_code == 200:
                    reply = response.json().get("response", "No reply from backend.")
                else:
                    reply = f"Error {response.status_code}: {response.text}"
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"API call failed: {e}")