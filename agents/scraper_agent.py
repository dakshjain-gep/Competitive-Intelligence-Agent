from langchain.agents import initialize_agent,AgentType
from langchain_openai import ChatOpenAI
from tools import scrape_competitor_news
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")


llm = ChatOpenAI(
    model="mistralai/mixtral-8x7b-instruct",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    max_tokens=3000,
)


scraper_agent=initialize_agent(tools=[scrape_competitor_news],llm=llm,
                               agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,
                               handle_parsing_errors=True,
                               return_direct=True,
                               agent_kwargs={
                                   "system_message": (
                                       "You are a highly capable research analyst tasked with extracting and summarizing key information from recent news articles."
                                       "Only use the content explicitly provided — do not speculate, infer, or hallucinate missing details. "
                                       "Present clear, structured, and factual observations. Focus on relevance, accuracy, and clarity in each summary."
                                   )
                               }
                               )