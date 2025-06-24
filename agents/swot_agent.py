from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm=ChatOpenAI(model="mistralai/Mixtral-8x7B-Instruct-v0.1",
               openai_api_key=os.getenv("TOGETHER_API_KEY"),
               openai_api_base="https://api.together.xyz/v1",)

swot_template=PromptTemplate.from_template("""
You are an expert business analyst specializing in strategic competitive intelligence.
Based on the following competitor insights, analyze and produce a comprehensive SWOT (Strengths, Weaknesses, Opportunities, and Threats) analysis for the company mentioned.

The analysis should:

Be formatted clearly using markdown with bold and bullet points where appropriate.

Include at least 4 points under each category: Strengths, Weaknesses, Opportunities, and Threats.

Consider both internal and external factors affecting the business.

Reflect recent trends, strategic moves, partnerships, market position, and technological or regulatory influences if present in the insights.

Competitor Insights:

{input}

Generate the SWOT analysis now.
""")

swot_chain=swot_template | llm