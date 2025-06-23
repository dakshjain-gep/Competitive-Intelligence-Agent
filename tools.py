from langchain_core.tools import tool
from scraper import scrape_all_sources
from processor import clean_text
from analyzer import extract_entities

@tool
def scrape_competitor_news(query:str)->str:
    """Scrape news about a competitor and return raw text."""
    raw_news=scrape_all_sources(query)
    for i, article in enumerate(raw_news, 1):
        print(f"{i}. [{article['source']}] {article['title']}")
        print(f"    Link: {article['link']}")
        print(f"    Snippet: {article['snippet']}")
        print("------")
    return "\n".join(raw_news)

@tool
def analyze_competitor_text(raw_text:str)->str:
    """Analyze text to extract key insights and entities"""
    cleaned=clean_text(raw_text)
    insights=extract_entities([cleaned])
    return str(insights)
