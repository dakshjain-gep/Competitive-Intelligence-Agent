from pydantic import BaseModel
from typing import List

class DataRequest(BaseModel):
    company: str

class NewsItem(BaseModel):
    title: str
    description: str
    content: str
    url: str
    published: str
    source: str

class DataResponse(BaseModel):
    company: str
    news: List[NewsItem]
