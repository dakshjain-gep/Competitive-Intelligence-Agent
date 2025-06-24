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
You are an expert business analyst.
Given the following competitor insights:
{input}

Generate a detailed SWOT analysis in markdown format.
""")

swot_chain = swot_template | llm