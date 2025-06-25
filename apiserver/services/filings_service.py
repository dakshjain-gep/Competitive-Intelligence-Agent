import requests
from schemas.filings_schema import FilingItem, FilingText
from typing import List
from bs4 import BeautifulSoup

def fetch_sec_filings(cik_or_name: str) -> List[FilingItem]:
    # CIK lookup via SEC API (if name is passed)
    if not cik_or_name.isdigit():
        cik = lookup_cik_from_name(cik_or_name)
        if(cik == "Company not found in SEC database."):
            return [
                FilingText(
                    text="Company name not found in SEC database",
                    document_url="No documents were found"
                )
            ]
    else:
        cik = cik_or_name.zfill(10)

    headers = { "User-Agent": "rohith0628@gmail.com" }
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    res = requests.get(url, headers=headers)
    
    if res.status_code != 200:
        raise Exception("Failed to fetch filings")

    data = res.json()
    recent = data.get("filings", {}).get("recent", {})
    count = min(len(recent.get("form", [])), 2)

    filings = [
        FilingItem(
            form_type=recent["form"][i],
            filed_at=recent["filingDate"][i],
            description=recent["primaryDocDescription"][i],
            document_url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{recent['accessionNumber'][i].replace('-', '')}/{recent['primaryDocument'][i]}"
        )
        for i in range(count)
    ]
        
    return [
        FilingText(
            text=scrape_filing(filing.document_url),
            document_url=filing.document_url   
        )
        for filing in filings
    ]
        
def lookup_cik_from_name(company: str) -> str:
    url = f"https://www.sec.gov/files/company_tickers.json"
    headers = { "User-Agent": "rohith0628@gmail.com" }
    res = requests.get(url, headers=headers)
    data = res.json()
    
    for entry in data.values():
        if company.lower() in entry["title"].lower():
            return str(entry["cik_str"]).zfill(10)

    return "Company not found in SEC database."


# to scrape the data from the filing document
def scrape_filing(filing_url: str) -> str:
    import requests
    from bs4 import BeautifulSoup

    headers = { "User-Agent": "rohith0628@gmail.com" }
    res = requests.get(filing_url, headers=headers)

    if res.status_code != 200:
        raise Exception(f"Failed to fetch filing: {filing_url}")

    soup = BeautifulSoup(res.content, "lxml-xml")

    # Remove <style> tags
    for meta_tag in soup.find_all("style"):
        meta_tag.decompose()

    # Extract visible text
    text = soup.get_text(separator="\n")

    # Final cleaning: strip blank lines and whitespace
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return cleaned or "[No readable text extracted]"


