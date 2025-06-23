# from selenium import webdriver
# from selenium.webdriver.common.by import By
# import time
#
# def scrape_news(query):
#     driver = webdriver.Chrome()
#
#     driver.get(f"https://www.google.com/search?q={query}+company+news&tbm=nws")
#
#     time.sleep(2)
#
#     elements = driver.find_elements(By.CSS_SELECTOR, 'div.BVG0Nb')
#
#     news = [ el.text for el in elements if el.text.strip() != "" ]
#
#     driver.quit()
#
#     return news


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time


def init_driver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=chrome_options)


def scrape_google_news(query, max_results=5):
    driver = init_driver()
    driver.get(f"https://www.google.com/search?q={query}+company+news&tbm=nws")
    time.sleep(3)

    articles = []
    elements = driver.find_elements(By.CSS_SELECTOR, 'div.dbsr')

    for el in elements[:max_results]:
        try:
            title = el.find_element(By.TAG_NAME, 'div').text
            link = el.find_element(By.TAG_NAME, 'a').get_attribute('href')
            snippet = el.find_element(By.CLASS_NAME, 'Y3v8qd').text
            articles.append({'source': 'Google News', 'title': title, 'link': link, 'snippet': snippet})
        except:
            continue

    driver.quit()
    return articles


def scrape_yahoo_finance_news(query, max_results=5):
    driver = init_driver()
    driver.get(f"https://finance.yahoo.com/quote/{query}/news")
    time.sleep(3)

    articles = []
    elements = driver.find_elements(By.CSS_SELECTOR, 'li.js-stream-content')

    for el in elements[:max_results]:
        try:
            title = el.find_element(By.TAG_NAME, 'h3').text
            link = el.find_element(By.TAG_NAME, 'a').get_attribute('href')
            snippet = el.find_element(By.TAG_NAME, 'p').text
            articles.append({'source': 'Yahoo Finance', 'title': title, 'link': link, 'snippet': snippet})
        except:
            continue

    driver.quit()
    return articles


def scrape_reuters_news(query, max_results=5):
    driver = init_driver()
    driver.get(f"https://www.reuters.com/site-search/?query={query}")
    time.sleep(3)

    articles = []
    elements = driver.find_elements(By.CSS_SELECTOR, 'div.search-result-content')

    for el in elements[:max_results]:
        try:
            title = el.find_element(By.TAG_NAME, 'h3').text
            link = el.find_element(By.TAG_NAME, 'a').get_attribute('href')
            snippet = el.find_element(By.CLASS_NAME, 'search-result-excerpt').text
            if not link.startswith("https://"):
                link = "https://www.reuters.com" + link
            articles.append({'source': 'Reuters', 'title': title, 'link': link, 'snippet': snippet})
        except:
            continue

    driver.quit()
    return articles


def scrape_businesswire(query, max_results=5):
    driver = init_driver()
    driver.get(f"https://www.businesswire.com/portal/site/home/search/?searchType=full&searchTerm={query}")
    time.sleep(3)

    articles = []
    elements = driver.find_elements(By.CSS_SELECTOR, 'div.search-result')[:max_results]

    for el in elements:
        try:
            title = el.find_element(By.CLASS_NAME, 'bwTitle').text
            link = el.find_element(By.TAG_NAME, 'a').get_attribute('href')
            snippet = el.find_element(By.CLASS_NAME, 'bwSummary').text
            articles.append({'source': 'BusinessWire', 'title': title, 'link': link, 'snippet': snippet})
        except:
            continue

    driver.quit()
    return articles


def scrape_all_sources(company, max_per_source=5):
    print(f"Scraping news for company: {company}")
    all_articles = []

    all_articles.extend(scrape_google_news(company, max_per_source))
    all_articles.extend(scrape_yahoo_finance_news(company, max_per_source))
    all_articles.extend(scrape_reuters_news(company, max_per_source))
    all_articles.extend(scrape_businesswire(company, max_per_source))

    return all_articles



# company = "Tesla"
# results = scrape_all_sources(company, max_per_source=3)

