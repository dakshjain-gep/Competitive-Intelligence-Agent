
# from agents.scraper_agent import scraper_agent

# company="Amazon"

# scraped_text=scraper_agent.invoke(f"Scrape news about {company}")

from fastapi import FastAPI, Request
from chain import start_llm_chain
from pydantic import BaseModel

app = FastAPI()

class MessageInput(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(payload: MessageInput):
    user_msg = payload.message
    reply = start_llm_chain(user_msg)
    return {"reply": reply}


