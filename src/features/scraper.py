import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

###
# OB 17.03.26
# web-scrape 
###

BASE_DIR = "../data/raw/"
os.makedirs(BASE_DIR, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}
# OB 17.03.26
# ToDo: erweitern der Url um weitere companies -done
companies = {
    "autodoc": "https://www.trustpilot.com/review/autodoc.de",
    "mister-auto": "https://www.trustpilot.com/review/mister-auto.de",
    "atp-autoteile": "https://www.trustpilot.com/review/www.atp-autoteile.de",
    "motointegrator": "https://www.trustpilot.com/review/motointegrator.de"}

all_reviews = []
# rating 
def extract_rating(article):

    img = article.find("img", {"alt": lambda x: x and "Rated" in x})

    if img and img.get("src"):

        svg_url = img["src"]

        svg_name = svg_url.split("/")[-1]

        return svg_name

    return None
 
#location 
def extract_location(article):

    location = None

    spans = article.find_all("span")

    for span in spans:

        if span.has_attr("data-consumer-country-typography"):

            text = span.get_text(strip=True)

            # Nur gültige Ländercodes (z.B. DE, FR, ES)
            if text and len(text) == 2:
                location = text
                break

    return location   

#supplier_response    
def extract_supplier_response(article):

    response = None

    paragraphs = article.find_all("p")

    for p in paragraphs:

        # prüfe ob es ein Supplier Response ist
        if p.has_attr("data-service-review-business-reply-text-typography"):

            # HTML sauber in Text umwandeln (inkl. <br>)
            response = p.get_text(separator=" ", strip=True)

            break

    return response


# review
def extract_review(article):

    review_text = None
    rating_svg = None
    date = None
    location = None
    supplier_response = None

    text_tag = article.find("p")

    if text_tag:
        review_text = text_tag.text.strip()
    # wird wie gewünscht entnommen
    rating_svg = extract_rating(article)
    #wird  korrekt entnommen
    date_tag = article.find("time")
    if date_tag:
        date = date_tag.get("datetime")
    # OB 17.03.26    
    # ToDo: anpassen auf link entname z.B. 'de' -done
    # location_tag = article.find("span", {"data-consumer-country-typography": True})
    #if location_tag:
    #   location = location_tag.text.strip()
    location = extract_location(article)
    
    #hier klappt etwas nicht -> ToDo:prüfen wie der tag aussieht-done -muss noch testen
    #response_tag = article.find("p", {"data-service-review-business-reply-text-typography": True})
    #if response_tag:
    #    supplier_response = response_tag.text.strip()
    supplier_response = extract_supplier_response(article)

    return {
        "review_text": review_text,
        "rating_svg": rating_svg,
        "date": date,
        #"product": None, # wird nicht automatisch erwähnt -> text matching?
        "location": location,
        "supplier_response": supplier_response
    }


def scrape_company(company, url, pages=50):

    print("Scraping:", company)

    for page in range(1, pages+1):

        page_url = f"{url}?page={page}"

        response = requests.get(page_url, headers=headers)

        soup = BeautifulSoup(response.text, "html.parser")

        articles = soup.find_all("article")

        for article in articles:

            review = extract_review(article)

            review["company"] = company

            all_reviews.append(review)

        time.sleep(2)


for company, url in companies.items():

    scrape_company(company, url)

df = pd.DataFrame(all_reviews)

df.to_json(BASE_DIR + "trustpilot_raw_reviews.json", orient="records")

print("Saved:", len(df), "reviews")

###
# OB 17.03.26
# clean json Dataset
###
df = pd.read_json("../data/raw/trustpilot_raw_reviews.json")

#zerlegt in numerisches rating
def extract_numeric_rating(svg):
    number = svg.split('-')[1].split('.')[0]
    return number
    
# bereinigt supplier_response    
#ToDo

#neue Spalte rating
df["rating"] = df["rating_svg"].apply(extract_numeric_rating)

#weg mit Duplikaten
df = df.drop_duplicates()

#weg mit allen zeilen ohne Komentar
df = df.dropna(subset=["review_text"])



#speichern unter -> wichtig zum später aufrufen
df.to_csv("../data/clean/reviews_clean.csv", index=False)

print("Clean dataset:", len(df))

#df.head(20)
df_csv = pd.read_csv("../data/raw/reviews_clean.csv")

df_csv.head(20)


