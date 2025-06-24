from langchain_openai import ChatOpenAI
from tools import analyze_competitor_text
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

load_dotenv()
llm=ChatOpenAI(model="mistralai/Mixtral-8x7B-Instruct-v0.1",
               openai_api_key=os.getenv("TOGETHER_API_KEY"),
               openai_api_base="https://api.together.xyz/v1",)

analyzer_agent=initialize_agent(
    tools=[analyze_competitor_text],llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,
    handle_parsing_errors=True
)