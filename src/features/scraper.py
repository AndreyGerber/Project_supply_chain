import re
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

###
# OB 17.03.26
# web-scrape 
###

# -------------------------------
# Setup
# -------------------------------
BASE_RAW = "../data/raw/"
os.makedirs(BASE_RAW, exist_ok=True)

edge_options = Options()
edge_options.add_argument("--headless")
edge_options.add_argument("--window-size=1920,1080")

service = Service(r"C:\WebDriver\msedgedriver.exe")

driver = webdriver.Edge(service=service, options=edge_options)

wait = WebDriverWait(driver, 10)

# OB 17.03.26
#Best in Auto Parts & Wheels Germany
# Händler mit Bewertung >6000
companies1 = { 
    "autodoc_de": "https://www.trustpilot.com/review/autodoc.de",
    "mister-auto_de": "https://www.trustpilot.com/review/mister-auto.de",
    "atp-autoteile_de": "https://www.trustpilot.com/review/www.atp-autoteile.de",
    "motointegrator_de": "https://www.trustpilot.com/review/motointegrator.de",
    "stabilo-fachmarkt_de": "https://www.trustpilot.com/review/www.stabilo-fachmarkt.de",
    "autodoc_pt": "https://www.trustpilot.com/review/auto-doc.pt",
    "pkwteile_de": "https://www.trustpilot.com/review/pkwteile.de",
    "autodoc_be": "https://www.trustpilot.com/review/autodoc.be",
    "daparto_de": "https://www.trustpilot.com/review/www.daparto.de",
    "motointegrator_fr": "https://www.trustpilot.com/review/www.motointegrator.fr",
    "kfzteile24_de": "https://www.trustpilot.com/review/www.kfzteile24.de",
    "autoteile-markt_de": "https://www.trustpilot.com/review/autoteile-markt.de",
    "autoteiledirekt_de": "https://www.trustpilot.com/review/autoteiledirekt.de"
}
#Händler mit Bewertung <6000 >1000
companies2 = { 
    "aerosus_com": "https://www.trustpilot.com/review/aerosus.com",
    "scheibenwischer_com": "https://www.trustpilot.com/review/scheibenwischer.com",
    "bandel-online_de": "https://www.trustpilot.com/review/www.bandel-online.de",
    "trodo_com": "https://www.trustpilot.com/review/trodo.com",
    "rsu_de": "https://www.trustpilot.com/review/rsu.de",
    "motointegrator_it": "https://www.trustpilot.com/review/motointegrator.it",
    "motointegrator_nl": "https://www.trustpilot.com/review/motointegrator.nl",
    "kovvar_de": "https://www.trustpilot.com/review/www.onlinefussmatten.de",
    "wunschkennzeichen-reservieren_jetzt": "https://www.trustpilot.com/review/wunschkennzeichen-reservieren.jetzt",
    "autodoc_eu": "https://www.trustpilot.com/review/autodoc.eu",
    "motointegrator_es": "https://www.trustpilot.com/review/motointegrator.es",
    "autodoc_pl": "https://www.trustpilot.com/review/autodoc.pl",
    "profiteile_de": "https://www.trustpilot.com/review/profiteile.de"
}
companies3 = {
    "carglass_de": "https://www.trustpilot.com/review/carglass.de",
    "reifendirekt_de": "https://www.trustpilot.com/review/reifendirekt.de",
    "kennzeichen_express": "https://www.trustpilot.com/review/www.kennzeichen.express",
    "reifenleader_de": "https://www.trustpilot.com/review/reifenleader.de",
    "123pneus_fr": "https://www.trustpilot.com/review/123pneus.fr",
    "pitstop_de": "https://www.trustpilot.com/review/www.pitstop.de",
    "gommadiretto_it": "https://www.trustpilot.com/review/gommadiretto.it",
    "maciag-offroad_de": "https://www.trustpilot.com/review/maciag-offroad.de",
    "reifendiscount_de": "https://www.trustpilot.com/review/www.reifendiscount.de",
    "reifendirekt_ch": "https://www.trustpilot.com/review/reifendirekt.ch",
    "motea_com": "https://www.trustpilot.com/review/motea.com",
    "elektrovorteil_de": "https://www.trustpilot.com/review/elektrovorteil.de",
    "xlmoto_de": "https://www.trustpilot.com/review/xlmoto.de",
    "dackonline_se": "https://www.trustpilot.com/review/dackonline.se",
    "dækonline_dk": "https://www.trustpilot.com/review/d%C3%A6konline.dk" 
    
    }

companies4 = {
    "giga-reifen_de": "https://www.trustpilot.com/review/giga-reifen.de",
    "reifendirekt_at": "https://www.trustpilot.com/review/reifendirekt.at",
    "reifen-pneus-online_de": "https://www.trustpilot.com/review/reifen-pneus-online.de",
    "motorradreifendirekt_de": "https://www.trustpilot.com/review/www.motorradreifendirekt.de",
    "reifen-vor-ort_de": "https://www.trustpilot.com/review/www.reifen-vor-ort.de",
    "autobandenmarkt_nl": "https://www.trustpilot.com/review/autobandenmarkt.nl",
    "grip500_de": "https://www.trustpilot.com/review/grip500.de",
    "123pneus_ch": "https://www.trustpilot.com/review/123pneus.ch",
    "reifen24_de": "https://www.trustpilot.com/review/reifen24.de",
    "shop4tesla_com": "https://www.trustpilot.com/review/shop4tesla.com",
    "motorradmeistermilz_de": "https://www.trustpilot.com/review/motorradmeistermilz.de",
    "neumaticos-online_es": "https://www.trustpilot.com/review/neumaticos-online.es",
    "autobatterienbilliger_de": "https://www.trustpilot.com/review/autobatterienbilliger.de",
    "polo-motorrad_com": "https://www.trustpilot.com/review/polo-motorrad.com",
    "quotlix_de": "https://www.trustpilot.com/review/quotlix.de",
    "carbonify_de": "https://www.trustpilot.com/review/carbonify.de",
    "schilder-kaufen": "https://www.trustpilot.com/review/schilder.kaufen",
    "dekkonline_com": "https://www.trustpilot.com/review/dekkonline.com",
    "fussmatten-autoteppiche_de": "https://www.trustpilot.com/review/fussmatten-autoteppiche.de",
    "mvh-shop_de": "https://www.trustpilot.com/review/mvh-shop.de", 
    "www-autobutler_de": "https://www.trustpilot.com/review/www.autobutler.de",
    "123pneus_be": "https://www.trustpilot.com/review/123pneus.be",
    "oponeo_de": "https://www.trustpilot.com/review/oponeo.de",
    "ws-autoteile_com": "https://www.trustpilot.com/review/ws-autoteile.com"    
    }
all_companies = {**companies3, **companies4}
all_reviews = []

# -------------------------------
# Funktionen
# -------------------------------
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

#Trustpilot:
# There are two types of product review labels: 
# Those that indicate that the review has been collected via a Trustpilot 
# invitation, and those imported into our system from another product 
# review platform. -> "Verified" and "Invited" , both indicate that the review is genuine and collected through Trustpilot's invitation 
# while not have undergone the same verification process.

def extract_verified(article): 
    tag = article.find("span", string=lambda x: x and ("Verified" in x or "Invited" in x))
    return 1 if tag else 0

# review
def extract_review(article, company):
    try:
        text_tag = article.find("p")
        review_text = text_tag.get_text(strip=True) if text_tag else None

        date_tag = article.find("time")
        date = date_tag.get("datetime") if date_tag else None

        return {
            "review_text": review_text,
            "rating_svg": extract_rating(article),
            "date": date,
            "location": extract_location(article),
            "supplier_response": extract_supplier_response(article),
            "verified": extract_verified(article),
            "company": company
        }
    except Exception as e:
        print("Error parsing review:", e)
        return None

# -------------------------------
# Helper: Scroll
# -------------------------------

def scroll_page():
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)


# -------------------------------
# Scraper
# -------------------------------

def scrape_company(company, url, pages=20):
    print(f"\n🚀 Scraping {company}")

    for page in range(1, pages + 1):
        page_url = f"{url}?page={page}"
        print(f"→ Page {page}")

        driver.get(page_url)

        try:
            # Warten bis Reviews geladen sind
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))
        except:
            print("⚠️ Timeout – keine Reviews gefunden")
            break

        scroll_page()

        soup = BeautifulSoup(driver.page_source, "html.parser")
        articles = soup.find_all("article")

        print(f"   Found {len(articles)} reviews")

        if len(articles) == 0:
            print("⚠️ Keine weiteren Reviews → Stop")
            break

        for article in articles:
            review = extract_review(article, company)
            if review:
                all_reviews.append(review)

        # Anti-Blocking
        time.sleep(3)

# -------------------------------
# Run Scraper
# -------------------------------

for company, url in all_companies.items():
    scrape_company(company, url, pages=15)

driver.quit()

df = pd.DataFrame(all_reviews)

print("\n✅ Total reviews scraped:", len(df))

df.to_json(BASE_RAW + "trustpilot_reviews_production2.json", orient="records", indent=2)

#df.to_json(BASE_RAW + "trustpilot_raw_reviews2.json", orient="records", indent=2)

