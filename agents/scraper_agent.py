from langchain.agents import initialize_agent,AgentType
from langchain_openai import ChatOpenAI
from tools import scrape_competitor_news
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatOpenAI(model="mistralai/Mixtral-8x7B-Instruct-v0.1",
               openai_api_key=os.getenv("TOGETHER_API_KEY"),
               openai_api_base="https://api.together.xyz/v1",)

scraper_agent=initialize_agent(tools=[scrape_competitor_news],llm=llm,
                               agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,
                               handle_parsing_errors=True,
                               return_direct=True)