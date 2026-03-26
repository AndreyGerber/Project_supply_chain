import pandas as pd
import os
import re
import numpy as np

from textblob import TextBlob
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_CLEAN = "../data/clean/"

os.makedirs(BASE_CLEAN, exist_ok=True)

df = pd.read_csv(BASE_CLEAN + "reviews_clean_test.csv")

#Datums-Engineering
# Datum konvertieren
df['date'] = pd.to_datetime(df['date'], errors='coerce') 
#habe ich schon in clean.py gemacht?
# Jahr, Monat, Wochentag
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.day_name()

# Jahreszeit
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df['season'] = df['month'].apply(get_season)

# Tageszeit
def get_daytime(hour):
    if pd.isna(hour):
        return "Unknown"
    elif 5 <= hour < 11:
        return "Morning"
    elif 11 <= hour < 16:
        return "Midday"
    elif 16 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df['hour'] = df['date'].dt.hour
df['daytime'] = df['hour'].apply(get_daytime)
#-> nach untersuchung der Daten, eine weitere Kategororische Variable hinzufügen?


# Kategorische Daten 
# Fehlwerte füllen
df['company_site'] = df['company_site'].fillna("Unknown")
df['company'] = df['company'].fillna("Unknown")
df['location'] = df['location'].fillna("Unknown")

# One-Hot-Encoding
df_encoded = pd.get_dummies(
    df,
    columns=['company_site', 'company', 'location'],
    drop_first=False
)



#Text-Vektorisierung (TF-IDF)

# Fehlwerte behandeln
df['review_text'] = df['review_text'].fillna("")
df['supplier_response'] = df['supplier_response'].fillna("")

# TF-IDF für Reviews
tfidf_review = TfidfVectorizer(max_features=500)
X_review = tfidf_review.fit_transform(df['review_text'])

# TF-IDF für Supplier Response
tfidf_response = TfidfVectorizer(max_features=300)
X_response = tfidf_response.fit_transform(df['supplier_response'])


#Sentiment Analyse




def get_sentiment(text):
    if not text:
        return 0
    return TextBlob(text).sentiment.polarity  # -1 bis +1

df['sentiment_review'] = df['review_text'].apply(get_sentiment)
df['sentiment_response'] = df['supplier_response'].apply(get_sentiment)

#Numerische Features
# Anzahl Issues (falls noch nicht vorhanden)
df['num_issues'] = df['issue_categories'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Länge des Reviews
df['review_length'] = df['review_text'].apply(lambda x: len(x.split()))

#Alles zusammenführen (Feature Matrix)


# Numerische Features
numerical_features = df[[
    'num_issues',
    'review_length',
    'sentiment_review',
    'sentiment_response'
]].values

# Sparse Matrix kombinieren
X = hstack([
    X_review,
    X_response,
    numerical_features
])
# Target Variable (Regression)
# rating vorbereiten (falls mein scraper oder cleaner sich ändert)
df['rating'] = df['rating_svg'].astype(float)

y = df['rating']