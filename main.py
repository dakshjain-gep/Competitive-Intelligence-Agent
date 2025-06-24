from typing import Union
import os
from agents.analyzer_agent import analyzer_agent
from agents.scraper_agent import scraper_agent
from agents.swot_agent import swot_chain

company="Amazon"

scraped_text=scraper_agent.invoke(f"Scrape news about {company}")

analyzed_output=analyzer_agent.invoke(scraped_text)

swot_result=swot_chain.invoke(input=analyzed_output)

print(swot_result)

