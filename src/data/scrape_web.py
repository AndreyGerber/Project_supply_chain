import re
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
import json

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# -------------------------------
# Setup
# -------------------------------

BASE_RAW = "../data/raw/"
os.makedirs(BASE_RAW, exist_ok=True)

DATA_FILE = BASE_RAW + "trustpilot_autosave.json"
CHECKPOINT_FILE = BASE_RAW + "checkpoint.json"
COMPANIES_FILE = BASE_RAW + "companies.json"

# -------------------------------
# Driver Setup
# -------------------------------

def create_driver():
    edge_options = Options()
    edge_options.add_argument("--start-maximized")
    edge_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    service = Service(r"C:\WebDriver\msedgedriver.exe")
    driver = webdriver.Edge(service=service, options=edge_options)

    driver.execute_script("""
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
    })
    """)

    return driver


driver = create_driver()
wait = WebDriverWait(driver, 10)

# -------------------------------
# Human Behavior
# -------------------------------

def human_delay(a=2.5, b=6.5):
    time.sleep(random.uniform(a, b))


def human_scroll():
    total_height = driver.execute_script("return document.body.scrollHeight")
    current = 0

    while current < total_height:
        step = random.randint(300, 800)
        driver.execute_script(f"window.scrollBy(0, {step});")
        current += step
        time.sleep(random.uniform(0.5, 1.5))


# -------------------------------
# Restart Driver
# -------------------------------

def restart_driver():
    global driver, wait

    print("🔄 Restarting browser...")

    try:
        driver.quit()
    except:
        pass

    time.sleep(5)

    driver = create_driver()
    wait = WebDriverWait(driver, 10)


# -------------------------------
# Safe GET
# -------------------------------

def safe_get(url, retries=3):
    for i in range(retries):
        try:
            driver.get("https://www.google.com/")
            human_delay(2, 4)

            driver.get(url)
            human_delay(2, 5)

            if "<article" in driver.page_source.lower():
                return True
            

        except Exception as e:
            print(e)

        print(f"Retry {i+1} for {url}")
        time.sleep(random.uniform(5, 10))

    restart_driver()
    return False


def accept_cookies():
    try:
        btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='accept-all']"))
        )
        btn.click()
        human_delay(1, 2)
    except:
        pass


# -------------------------------
# Extract Functions
# -------------------------------

def extract_review_id(article):
    # Try primary attribute
    rid = article.get("data-service-review-id")

    if not rid:
        # Fallback 1
        rid = article.get("data-review-id")

    if not rid:
        # Fallback 2: hash (immer vorhanden)
        rid = str(hash(article.get_text()))

    return rid

def extract_rating(article):
    img = article.find("img", alt=lambda x: x and "Rated" in x)
    if img:
        match = re.search(r"(\d)", img.get("alt"))
        return int(match.group(1)) if match else None
    return None


def extract_location(article):
    tag = article.find("span", attrs={"data-consumer-country-typography": True})
    return tag.get_text(strip=True) if tag else None


def extract_supplier_response(article):
    tag = article.find("p", attrs={"data-service-review-business-reply-text-typography": True})
    return tag.get_text(" ", strip=True) if tag else None


def extract_verified(article):
    tag = article.find("span", string=lambda x: x and ("Verified" in x or "Invited" in x))
    return 1 if tag else 0


def extract_review_text(article): 
    tag = article.find("p", attrs={"data-service-review-text-typography": True})
    
    if not tag:
        # Fallback
        tag = article.find("p")
    
    return tag.get_text(" ", strip=True) if tag else None

def extract_response_date(article):
    try:
        reply_div = article.find("div", attrs={"data-service-review-business-reply-title-typography": True})
        if reply_div:
            time_tag = reply_div.find("time")
            if time_tag:
                return time_tag.get("datetime")
    except:
        pass
    return None


def extract_review(article, company):
    review_id = extract_review_id(article)

    date_tag = article.find("time")
    review_date = date_tag.get("datetime") if date_tag else None

    return {
        "review_id": review_id,
        "rating": extract_rating(article),
        "review_text": extract_review_text(article),
        "review_date": review_date,
        "location": extract_location(article),
        "supplier_response": extract_supplier_response(article),
        "response_date": extract_response_date(article),
        "verified": extract_verified(article),
        "company": company
    }


# -------------------------------
# Checkpoint Handling
# -------------------------------

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set(), set()

    with open(CHECKPOINT_FILE, "r") as f:
        data = json.load(f)

    return set(data["seen_ids"]), set(data["processed_companies"])


def save_checkpoint(seen_ids, processed_companies):
    data = {
        "seen_ids": list(seen_ids),
        "processed_companies": list(processed_companies)
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


# -------------------------------
# Load existing data
# -------------------------------

if os.path.exists(DATA_FILE):
    print("🔄 Loading existing data...")
    df_existing = pd.read_json(DATA_FILE)

    all_reviews = df_existing.to_dict("records")

    if "review_id" in df_existing.columns:
        seen_ids = set(df_existing["review_id"])
    else:
        print("⚠️ No review_id in old data → resetting")
        seen_ids = set()
else:
    all_reviews = []
    seen_ids = set()
_, processed_companies = load_checkpoint()

# -------------------------------
# Category Scraper
# -------------------------------

def scrape_category_urls(base_url, max_pages=5):
    company_urls = {}

    for page in range(1, max_pages + 1):
        url = f"{base_url}&page={page}"

        if not safe_get(url):
            continue

        accept_cookies()
        human_delay()

        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/review/" in href:
                name = href.split("/review/")[-1]
                company_urls[name] = "https://www.trustpilot.com" + href



    return company_urls


# -------------------------------
# Company Load / Save
# -------------------------------

category_url = "https://www.trustpilot.com/categories/auto_parts_wheels?country=DE&sort=reviews_count"

if os.path.exists(COMPANIES_FILE):
    print("📂 Loading existing company list...")
    with open(COMPANIES_FILE, "r") as f:
        companies = json.load(f)
else:
    print("🔎 Scraping company list...")
    companies = scrape_category_urls(category_url, max_pages=5)

    with open(COMPANIES_FILE, "w") as f:
        json.dump(companies, f)

# -------------------------------
# Scraper
# -------------------------------

SAVE_EVERY = 50

def autosave():
    pd.DataFrame(all_reviews).to_json(DATA_FILE, orient="records", indent=2)
    save_checkpoint(seen_ids, processed_companies)
    print(f"💾 Autosaved ({len(all_reviews)} reviews)")


def scrape_company(company, url, pages=10):
    print(f"\n🚀 {company}")

    for page in range(1, pages + 1):
        page_url = f"{url}?page={page}"

        if not safe_get(page_url):
            break

        accept_cookies()
        human_delay()
        human_scroll()

        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.find_all("article")
        print(f"Articles found: {len(articles)}")

        if len(articles) == 0:
            break

        for article in articles:
            
            review_id = extract_review_id(article)
            #print("ID:", review_id)
            if not review_id or review_id in seen_ids:
                continue

            seen_ids.add(review_id)

            review = extract_review(article, company)
            if review:
                all_reviews.append(review)

                if len(all_reviews) % SAVE_EVERY == 0:
                    autosave()
        
        print("Seen IDs:", len(seen_ids))
        print("Total reviews collected:", len(all_reviews))        
        human_delay(3, 7)


# -------------------------------
# RUN
# -------------------------------

items = list(companies.items())
random.shuffle(items)

for company, url in items:

    if company in processed_companies:
        print(f"⏭️ Skipping {company}")
        continue

    scrape_company(company, url)

    processed_companies.add(company)

    autosave()

    time.sleep(random.uniform(10, 20))

driver.quit()

# -------------------------------
# Final Save
# -------------------------------

assert len(all_reviews) > 0, "❌ No data collected! Scraper failed."
df = pd.DataFrame(all_reviews)

df["review_date"] = pd.to_datetime(df["review_date"])
df["response_date"] = pd.to_datetime(df["response_date"])

df["response_time_days"] = (
    (df["response_date"] - df["review_date"])
    .dt.total_seconds() / 86400
)

# zusätzliche sinnvolle Features
df["has_response"] = df["supplier_response"].notna()
df["response_time_days"] = df["response_time_days"].fillna(-1)


df.to_json(BASE_RAW + "trustpilot_final.json", orient="records", indent=2)

print("✅ DONE")

