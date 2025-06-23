from typing import Union
from fastapi import FastAPI,Request,Depends,HTTPException

from openai import OpenAI
from fastapi.responses import HTMLResponse
import os
from agents.analyzer_agent import analyzer_agent
from agents.scraper_agent import scraper_agent
from agents.swot_agent import swot_chain


# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates



# app = FastAPI()
company="GEP"

scraped_text=scraper_agent.invoke(f"Scrape news about {company}")

analyzed_output=analyzer_agent.invoke(scraped_text)

swot_result=swot_chain.invoke(input=analyzed_output)

print(swot_result)

