from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
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


def extract_urls(driver,urls,max_articles,query):
    elements = driver.find_elements(By.XPATH, f'//div[contains(text(),"{query}")]')
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

def scrape_news(query, max_articles=10):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    driver.get(f"https://www.google.com/search?q={query}+company+and+its+competitors&tbm=nws")
    time.sleep(5)

    urls = []

    extract_urls(driver,urls,max_articles,query)

    try:
        pages = driver.find_elements(By.XPATH, '//a[starts-with(@aria-label, "Page ") and number(substring-after(@aria-label, "Page ")) <= 3]')
        page_links = [a.get_attribute('href') for a in pages if a.get_attribute('href')]
    except Exception as e:
        print(f"Error finding pagination: {e}")


    for page_url in page_links:
        try:
            driver.get(page_url)
            time.sleep(5)

            extract_urls(driver,urls,max_articles,query)
            if len(urls) >= max_articles:
                break
        except Exception as e:
            print(f"Failed on page : {e}")


    scraped_articles = []
    for i, url in enumerate(urls):
        try:
            driver.get(url)
            time.sleep(3)


            scroll_to_bottom(driver)

            paragraphs = driver.find_elements(By.TAG_NAME, "p")
            paragraphs_text = " ".join([p.text for p in paragraphs if p.text.strip()])

            if paragraphs_text:
                scraped_articles.append(paragraphs_text[:4000])
            else:
                scraped_articles.append(f"[No text content found at] {url}")

            print(f"Scraped article {i + 1} from {url}")

        except Exception as e:
            print(f"Failed to scrape article {i + 1} from {url}: {e}")
            scraped_articles.append(f"[Failed to scrape] {url}")

    driver.quit()
    return scraped_articles



