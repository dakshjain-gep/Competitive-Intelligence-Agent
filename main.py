from agents.scraper_agent import scraper_agent


# app = FastAPI()
company="Facebook"

scraped_text=scraper_agent.invoke(f"Scrape news about {company}")

print(scraped_text["output"])


