from langchain_core.tools import tool
from scraper import scrape_news
from processor import clean_text
from analyzer import extract_entities

@tool
def scrape_competitor_news(query:str)->str:
    """Scrape news articles about a competitor and return raw combined text content."""
    raw_news = scrape_news(query)
    if not raw_news:
        return f"No news articles found for {query}."

    formatted_news = "\n\n".join([f"{i + 1}. {item.strip()}" for i, item in enumerate(raw_news)])
    return formatted_news

