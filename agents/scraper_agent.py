from langchain.agents import initialize_agent,AgentType
from langchain_openai import ChatOpenAI
from tools import scrape_competitor_news
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")


llm = ChatOpenAI(
    # model="mistralai/mixtral-8x7b-instruct",
    # openai_api_base="https://openrouter.ai/api/v1",
    # openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    openai_api_base="https://api.together.xyz/v1",
    openai_api_key=os.getenv("TOGETHER_API_KEY"),
    temperature=0.3,
    max_tokens=3000,
)


scraper_agent=initialize_agent(tools=[scrape_competitor_news],llm=llm,
                               agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,
                               handle_parsing_errors=True,
                               return_direct=True,
                               agent_kwargs={
                                   "system_message": (
                                       "You are a highly capable research analyst tasked with extracting and summarizing key information from recent news articles scraped."
                                       "Only use the content explicitly scraped and provided to you — do not speculate, infer, or hallucinate missing details. "
                                       "Present clear, structured, and factual observations. Focus on relevance, accuracy, and clarity in each summary."
                                       "Also do not ignore any information that is scraped and provided to you."
                                       "Cover all the information in the final verdict that the scraped text provided to you contains."
                                   )
                               }
                               )