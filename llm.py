import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_key=os.getenv("TOGETHER_API_KEY"),
    openai_api_base="https://api.together.xyz/v1"
)