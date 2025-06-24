from langchain.agents import initialize_agent,AgentType
from langchain_openai import ChatOpenAI
from tools import scrape_competitor_news
from dotenv import load_dotenv
from llm import llm

scraper_agent=initialize_agent(tools=[scrape_competitor_news],llm=llm,
                               agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,
                               handle_parsing_errors=True)