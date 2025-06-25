from langchain_core.tools import tool
from scraper import scrape_news
from chain import generate_prompt_data, build_ci_chain
from llm import llm
@tool
def scrape_competitor_news(query:str)->str:
    """Scrape news about a competitor and return raw text."""
    raw_news=scrape_news(query)
    # for i, article in enumerate(raw_news, 1):
    #     print(f"{i}. [{article['source']}] {article['title']}")
    #     print(f"    Link: {article['link']}")
    #     print(f"    Snippet: {article['snippet']}")
    #     print("------")
    return "\n".join(raw_news)