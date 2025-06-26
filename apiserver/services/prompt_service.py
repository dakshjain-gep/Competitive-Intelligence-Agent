from controllers.data_controller import fetch_news
from controllers.filings_controller import fetch_sec_filings

def generate_prompt(company: str) -> str:
    news_items = fetch_news(company)
    
    filings_texts = fetch_sec_filings(company)
    
    formatted_articles = "\n\n".join([
        f"Title: {article.title}\nSource: {article.source}\nPublished: {article.published} \nContent: {article.description + ' ' + article.content}"
        for article in news_items
    ])
    
    formatted_filings = "\n\n--- Filing Document ---\n\n".join([
        f"Document URL: {filing.document_url}\n\n{text[:5000]}"
        for filing in filings_texts if isinstance(filing.text, str)
        for text in [filing.text]
    ])

    return {
        "company": company,
        "articles": formatted_articles,
        "filings": formatted_filings
    }