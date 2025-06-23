import re
from bs4 import BeautifulSoup

def clean_text(text):
    text=BeautifulSoup(text, "html.parser").text
    text=re.sub(r'\s+','',text)
    return text.strip()