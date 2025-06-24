# from selenium import webdriver
# from selenium.webdriver.common.by import By
# import time
# from selenium.webdriver.common.action_chains import ActionChains
#
# def scrape_news(query):
#     driver = webdriver.Chrome()
#
#     print(query)
#     driver.get(f"https://www.google.com/search?q={query}+company+news&tbm=nws")
#
#     time.sleep(10)
#
#     elements = driver.find_elements(By.XPATH, f'//div[contains(text(),"{query}")]')
#
#     news = [ el.text for el in elements if el.text.strip() != "" ]
#
#     driver.quit()
#
#     return news    1




# from selenium import webdriver
# from selenium.webdriver.common.by import By
# import time
#
#
# def scrape_news(query, max_articles=10):
#     driver = webdriver.Chrome()
#     driver.get(f"https://www.google.com/search?q={query}&tbm=nws")
#
#     time.sleep(5)  # wait for page load, can be replaced with explicit waits
#
#     # Find divs containing the query text
#     elements = driver.find_elements(By.XPATH, f'//div[contains(text(),"{query}")]')
#
#     urls = []
#     for div in elements:
#         try:
#             # Get nearest ancestor <a> tag's href attribute
#             parent_a = div.find_element(By.XPATH, './ancestor::a[1]')
#             url = parent_a.get_attribute("href")
#             if url and url not in urls:
#                 urls.append(url)
#             if len(urls) >= max_articles:
#                 break
#         except Exception as e:
#             print(f"Skipping a div due to error: {e}")
#             continue
#
#     scraped_articles = []
#     for i, url in enumerate(urls):
#         try:
#             driver.get(url)
#             time.sleep(5)  # wait for the detailed news page to load
#
#             # Collect all paragraphs text from the news article
#             paragraphs = driver.find_elements(By.TAG_NAME, "p")
#             article_text = " ".join([p.text for p in paragraphs if p.text.strip()])
#
#             if article_text:
#                 scraped_articles.append(article_text[:4000])  # limit length if needed
#             else:
#                 scraped_articles.append(f"[No text content found at] {url}")
#
#             print(f"Scraped article {i + 1} from {url}")
#
#         except Exception as e:
#             print(f"Failed to scrape article {i + 1} from {url}: {e}")
#             scraped_articles.append(f"[Failed to scrape] {url}")
#
#     driver.quit()
#
#     return scraped_articles   2




from selenium import webdriver
from selenium.webdriver.common.by import By
import time


def scroll_to_bottom(driver, pause_time=1.5, max_scrolls=10):
    """Scrolls the page to the bottom to load dynamic content."""
    last_height = driver.execute_script("return document.body.scrollHeight")

    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause_time)  # Wait for new content to load

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def scrape_news(query, max_articles=5):
    driver = webdriver.Chrome()
    driver.get(f"https://www.google.com/search?q={query}+company+and+its+competitors&tbm=nws")
    time.sleep(5)

    elements = driver.find_elements(By.XPATH, f'//div[contains(text(),"{query}")]')

    urls = []
    for div in elements:
        try:
            parent_a = div.find_element(By.XPATH, './ancestor::a[1]')
            url = parent_a.get_attribute("href")
            if url and url not in urls:
                urls.append(url)
            if len(urls) >= max_articles:
                break
        except Exception as e:
            print(f"Skipping a div due to error: {e}")
            continue

    scraped_articles = []
    for i, url in enumerate(urls):
        try:
            driver.get(url)
            time.sleep(3)

            # Scroll to bottom to ensure all content is loaded
            scroll_to_bottom(driver)

            paragraphs = driver.find_elements(By.TAG_NAME, "p")
            divs = driver.find_elements(By.TAG_NAME, "div")
            spans = driver.find_elements(By.TAG_NAME, "span")
            paragraphs_text = " ".join([p.text for p in paragraphs if p.text.strip()])
            divs_text = " ".join([div.text for div in divs if div.text.strip()])
            spans_text = " ".join([span.text for span in spans if span.text.strip()])

            if paragraphs_text:
                scraped_articles.append(paragraphs_text[:4000])
            if divs_text:
                scraped_articles.append(divs_text[:4000])
            if spans_text:
                scraped_articles.append(spans_text[:4000])
            else:
                scraped_articles.append(f"[No text content found at] {url}")

            print(f"Scraped article {i + 1} from {url}")

        except Exception as e:
            print(f"Failed to scrape article {i + 1} from {url}: {e}")
            scraped_articles.append(f"[Failed to scrape] {url}")

    driver.quit()
    return scraped_articles



