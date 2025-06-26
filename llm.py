import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


load_dotenv()

# llm = ChatGroq(
#     model_name="llama3-8b-8192",
#     api_key=os.getenv("GROQ_API_KEY"),
# )

# llm = ChatOpenAI(
#     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#     openai_api_key=os.getenv("TOGETHER_API_KEY2"),
#     openai_api_base="https://api.together.xyz/v1"
# )

llm = ChatOpenAI(    
                 model="mistralai/mixtral-8x7b-instruct",    
                 openai_api_base="https://openrouter.ai/api/v1",    
                 openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                 temperature=0.7  
                 )