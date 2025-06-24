from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnableMap
import requests
from llm import llm

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

        Respond in Markdown format.
        """
            
    return RunnableLambda(_make_prompt) | llm

def extract_company_llm(prompt: str) -> str:
    return llm.invoke(f"""
        Extract the company name from the following prompt.

        Prompt: {prompt}

        Only return the company name from the prompt, nothing else.
        
        Example -> Input: Generate a CI for Tesla 
                Output: Tesla
        """).content.strip()
    

def start_llm_chain(prompt: str):
    
    company = extract_company_llm(prompt)
    print(company)

    url = "http://127.0.0.1:8000/prepare-prompt"
    payload = {
        "company": company
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        
        sections = {
            "summary": section_prompt_builder("Summary", "Summarize recent developments, key events, and any notable trends."),
            "sentiment": section_prompt_builder("Sentiment Analysis", "Conduct sentiment analysis based on tone and content."),
            "risks": section_prompt_builder("Risks & Legal", "Identify strategic risks, legal issues, or unusual activity."),
            "swot": section_prompt_builder("SWOT", "Generate a full SWOT analysis."),
            "ci": section_prompt_builder("CI Insights", "Outline competitive intelligence insights that may affect the company."),
            "forward_look": section_prompt_builder("Forward Looking", "Highlight forward-looking statements or executive activity.")
        }
        
        # Run all in parallel
        report_chain = RunnableMap(sections)
        
        results = report_chain.invoke(data)
        
        # Combine markdown
        full_markdown = f"# CI Brief: {company}\n\n"
        for key, value in results.items():
            full_markdown += f"## {key.replace('_', ' ').title()}\n{value.content}\n\n"

        return full_markdown
        
    else:
        return f"Error: {response.status_code} {response.text}"
        

