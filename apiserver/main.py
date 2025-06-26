from fastapi import FastAPI
from controllers import data_controller, filings_controller, prompt_controller, finance_controller

app = FastAPI(title="Company Data API")
app.include_router(data_controller.router)
app.include_router(filings_controller.router)
app.include_router(prompt_controller.router)
app.include_router(finance_controller.router)
