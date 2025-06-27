import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

# llm = ChatGroq(
#     model_name="llama3-8b-8192",
#     api_key=os.getenv("GROQ_API_KEY"),
# )

# llm = ChatOpenAI(
#     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#     openai_api_key=os.getenv("TOGETHER_API_KEY2"),
#     openai_api_base="https://api.together.xyz/v1"
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Or "gemini-1.5-flash", "gemini-1.5-pro", etc.
    google_api_key=os.getenv("GOOGLE_API_KEY1"),
    temperature=0.7
)