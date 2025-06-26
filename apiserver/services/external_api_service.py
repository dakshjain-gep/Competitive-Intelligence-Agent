import requests
from schemas.data_schema import NewsItem
import os
from dotenv import load_dotenv

load_dotenv()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

def fetch_news(company: str) -> list[NewsItem]:
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": company,
        "lang": "en",
        "token": GNEWS_API_KEY,
        "max": 5
    }

    response = requests.get(url, params=params)
    data = response.json()

    return [
        NewsItem(
            title=article["title"],
            description=article["description"],
            content=article["content"],
            url=article["url"],
            published=article["publishedAt"],
            source=article["source"]["name"]
        )
        for article in data.get("articles", [])
    ]
