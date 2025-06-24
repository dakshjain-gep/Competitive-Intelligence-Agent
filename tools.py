from langchain_core.tools import tool
from scraper import scrape_news
from chain import generate_prompt_data, build_ci_chain
from llm import llm
@tool
def scrape_competitor_news(query:str)->str:
    """Scrape news about a competitor and return raw text."""
    raw_news=scrape_news(query)
    # for i, article in enumerate(raw_news, 1):
    #     print(f"{i}. [{article['source']}] {article['title']}")
    #     print(f"    Link: {article['link']}")
    #     print(f"    Snippet: {article['snippet']}")
    #     print("------")
    return "\n".join(raw_news)

@tool
def generate_ci_brief(company: str) -> str:
    """
    Generate a Competitive Intelligence brief for the given company.
    The brief includes sections like recent developments, sentiment, risks, SWOT, and forward-looking insights.
    """
    data = generate_prompt_data(company)
    report_chain = build_ci_chain()
    results = report_chain.invoke(data)
    # Combine markdown
    full_markdown = f"# CI Brief: {company}\n\n"
    for key, value in results.items():
        full_markdown += f"## {key.replace('_', ' ').title()}\n{value.content}\n\n"

    return full_markdown

@tool
def ask_report_question(input: ReportQAInput) -> str:
    """
    Allow the user to ask follow up questions on a Competitive Intelligence brief that you previously generated for the user about a given company, else ask the user if he wants to generate a report.
    """
    question = input["question"]
    report_text = input["report_text"]
    
    return llm.invoke(f"""
        You are a CI expert. Based on the following report, answer the user's question.

        --- Report ---
        {report_text}

        --- Question ---
        {question}
        """).content
    
@tool
def compare_reports(input: compareQAInput) -> str:
    """
        If the user generated two reports for two given companies, then compare them and generate a descriptive report with all the data provided earlier.
    """
    
    report1 = input["report1"]
    report2 = input["report2"]
    
    return llm.invoke(f"""
        Compare the following two CI briefs and highlight key differences and competitive insights.

        --- Report 1 ---
        {report1}

        --- Report 2 ---
        {report2}
        """).content

