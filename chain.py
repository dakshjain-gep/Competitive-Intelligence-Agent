from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableMap
import requests
from llm import llm
import re
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

class Tickers(BaseModel):
    companyticker: str = Field(description="The given company's Ticker")
    competitortickers: List[str] = Field(description="The competitor companies Tickers")

def ticker_extraction_prompt(data: dict):
    return f"""
        You are a financial data assistant.

        Given the following company name, news articles, and SEC filings:

        --- Company ---
        {data['company']}

        --- News Articles ---
        {data['articles']}

        --- SEC Filings ---
        {data['filings']}

        Your only task:

        Return a JSON object with the following keys:

        - "companyticker": string
        - "competitortickers": list of strings (up to 5 relevant competitors)

        ❌ Do not add any explanation or extra text.

        ✅ Only output raw valid JSON like this:

        {{
        "companyticker": "TSLA",
        "competitortickers": ["BYDDF", "VWAGY", "RIVN", "NIO", "LI"]
        }}
        """

def section_prompt_builder(section_title: str, instruction: str):
    def _make_prompt(data: dict):
        
        return f"""
            You are an advanced CI agent. Your task is: **{instruction}**
            
          --- Company ---
        {data['company']}

        --- News Articles ---
        {data['articles']}

        --- SEC Filings ---
        {data['filings']}

        Respond in Markdown format which is pleasing and engaging to read.
        Read the markdown again and remove any unformattable pieces, keep a consistent look,
        and clean up the document.
        """
            
    return RunnableLambda(_make_prompt) | llm
    

def start_llm_chain(prompt: str):
    
    company = prompt.strip()
    print(company)

    url = "http://127.0.0.1:8000/prepare-prompt"
    payload = {
        "company": company
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        
        if(data["articles"] == ""):
            return "Are you sure this company exists?"
        
        ticker_extraction_chain = RunnableLambda(lambda d: ticker_extraction_prompt(d)) | llm
        ticker_result_raw = ticker_extraction_chain.invoke(data)
        raw_text = ticker_result_raw.content
        print(raw_text)
        
        query = f"Parse this text into a JSON object: {raw_text}"
        parser = JsonOutputParser(pydantic_object=Tickers)
        
        json_prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        json_chain = json_prompt | llm | parser
        ticker_json = json_chain.invoke({"query": query})
        print(ticker_json)
        
        sections = {
            "summary": section_prompt_builder("Summary", "Summarize recent developments, key events, and any notable trends."),
            "sentiment": section_prompt_builder("Sentiment Analysis", "Conduct sentiment analysis based on tone and content."),
            "risks": section_prompt_builder("Risks & Legal", "Identify strategic risks, legal issues, or unusual activity."),
            "swot": section_prompt_builder("SWOT Analysis", "Generate a full SWOT analysis."),
            "competitive intelligence": section_prompt_builder("CI Insights", "Outline competitive intelligence insights that may affect the company. Focus the most on this section and make it as descriptive as possible"),
        }
        
        # Run all in parallel
        report_chain = RunnableMap(sections)
        
        results = report_chain.invoke(data)
        
        # Combine markdown
        full_markdown = f"# CI Brief: {company}\n\n"
        for key, value in results.items():
            full_markdown += f"## {key.replace('_', ' ').title()}\n{value.content}\n\n"

        return {
                "markdown": full_markdown,
                "companyticker": ticker_json["companyticker"],
                "competitortickers": ticker_json["competitortickers"]
            }         
    else:
        return f"Error: {response.status_code} {response.text}"
        

