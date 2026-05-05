
import streamlit as st
from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# PAGE CONFIG
st.set_page_config(layout="wide")
st.title("📊 Feature Engineering")

# =========================
# 🔹 RAW DATA
# =========================
st.subheader("Raw Data")

file_path = Path("data/raw/trustpilot_reviews_production.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

df_raw = pd.DataFrame(data)

st.dataframe(df_raw.head(20))
st.write("""
The scraped raw data consists of eight columns:
review_text, rating_svg, date, location, supplier_response, verified, company
""")

# =========================
# 🔹 CLEAN DATA
# =========================
df = pd.read_csv("data/processed/reviews_processed.csv")

# =========================
# 🔹 EXTERNAL FEATURES
# =========================
st.subheader("First Part: External Features Analysis")


import numpy as np
from scipy.stats import chi2_contingency

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

st.title("📅 Feature Selection: Date Features")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("src/data/clean/reviews_processed.csv")

# =========================
# DATE CLEANING
# =========================
st.subheader("Date Preprocessing")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).reset_index(drop=True)

st.write("Cleaned dataset shape:", df.shape)

# =========================
# FEATURE EXTRACTION
# =========================
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day_of_week + 1

# =========================
# YEAR ANALYSIS
# =========================
st.subheader("📊 Year vs Rating")

ct_year = pd.crosstab(df["year"], df["rating"], normalize="index")
st.dataframe(ct_year)

chi2, p, _, _ = chi2_contingency(pd.crosstab(df["year"], df["rating"]))
n = df.shape[0]
V = np.sqrt(chi2 / (n * (min(pd.crosstab(df["year"], df["rating"]).shape) - 1)))

st.write(f"Chi²: {chi2:.2f} | p-value: {p:.5f} | Cramér’s V: {V:.2f}")

if p < 0.05:
    st.success("✔ Significant relationship between year and rating")
else:
    st.warning("⚠ No significant relationship")

st.info("➡️ Interpretation: statistically significant but weak relationship (~0.14)")

# =========================
# YEAR TREND
# =========================
st.subheader("📈 Rating Trend over Time")

fig, ax = plt.subplots()
df.groupby("year")["rating"].mean().plot(ax=ax)
ax.set_ylabel("Average Rating")
st.pyplot(fig)

# =========================
# YEAR TRANSFORMATION
# =========================
df["year"] = df["year"] - df["year"].min()

# =========================
# REVIEW AGE
# =========================
st.subheader("⏳ Review Age Feature")

df["review_age_days"] = (df["date"].max() - df["date"]).dt.days

fig, ax = plt.subplots()
sns.histplot(df["review_age_days"], bins=50, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(x="rating", y="review_age_days", data=df, ax=ax)
st.pyplot(fig)

# =========================
# AGE BUCKET
# =========================
df["age_bucket"] = pd.cut(
    df["review_age_days"],
    bins=[0, 30, 180, 365, 10000],
    labels=["0-30d", "1-6m", "6-12m", "1y+"]
)

st.subheader("📦 Age Bucket Analysis")

ct_age = pd.crosstab(df["age_bucket"], df["rating"], normalize="index")
st.dataframe(ct_age)

ct = pd.crosstab(df["age_bucket"], df["rating"])
chi2, p, _, _ = chi2_contingency(ct)

n = ct.values.sum()
V = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

st.write(f"Chi²: {chi2:.2f} | p-value: {p:.5f} | Cramér’s V: {V:.2f}")

# =========================
# TIME TREND MODEL
# =========================
st.subheader("📉 Time Trend (Regression)")

df["number_of_months"] = (
    (df["date"].dt.year - df["date"].dt.year.min()) * 12
    + df["date"].dt.month
)

model = smf.ols("rating ~ number_of_months", data=df).fit()

st.text(model.summary())

st.info("➡️ Weak but significant negative trend over time")

# =========================
# MONTH ANALYSIS
# =========================
st.subheader("📅 Month Analysis")

df["month_cat"] = df["month"].astype("category")

ct_month = pd.crosstab(df["month_cat"], df["rating"], normalize="index")
st.dataframe(ct_month)

chi2, p, _, _ = chi2_contingency(pd.crosstab(df["month_cat"], df["rating"]))

n = df.shape[0]
V = np.sqrt(chi2 / (n * (min(12, 5) - 1)))

st.write(f"Cramér’s V: {V:.2f}")

st.success("✔ Months show meaningful variation → useful feature")

# =========================
# SEASON FEATURE
# =========================
st.subheader("🌦 Season Feature")

def get_season(m):
    if m in [12,1,2]:
        return "winter"
    elif m in [3,4,5]:
        return "spring"
    elif m in [6,7,8]:
        return "summer"
    else:
        return "autumn"

df["season"] = df["month"].apply(get_season)

ct = pd.crosstab(df["season"], df["rating"])
chi2, p, _, _ = chi2_contingency(ct)

n = ct.values.sum()
V = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

st.write(f"Chi²: {chi2:.2f} | p-value: {p:.5f} | Cramér’s V: {V:.2f}")

# =========================
# SPECIAL FLAGS
# =========================
st.subheader("🎯 Special Time Features")

df["is_year_end"] = df["month"].isin([11,12]).astype(int)
df["is_march"] = (df["month"] == 3).astype(int)

for feature in ["is_year_end", "is_march"]:
    st.markdown(f"### {feature}")

    ct = pd.crosstab(df[feature], df["rating"])
    chi2, p, _, _ = chi2_contingency(ct)

    n = ct.values.sum()
    V = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

    st.write(f"Chi²: {chi2:.2f} | p-value: {p:.5f} | Cramér’s V: {V:.2f}")


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

st.subheader("📝 Review Length vs Rating")

# =========================
# 🔹 KORRELATION
# =========================
st.markdown("### Correlation")

corr = df[["review_length", "rating"]].corr()
st.write(corr)

st.info("""
Moderate negative correlation (~ -0.44):  
➡️ Longer reviews tend to be associated with lower ratings.
""")

# =========================
# 🔹 VISUALISIERUNG
# =========================
st.markdown("### Visualization")

fig, ax = plt.subplots()
sns.boxplot(x="rating", y="review_length", data=df, ax=ax)
ax.set_title("Review Length vs Rating")
st.pyplot(fig)

# =========================
# 🔹 LOG TRANSFORMATION
# =========================
st.markdown("### Log Transformation")

df["review_length_log"] = np.log1p(df["review_length"])

fig, ax = plt.subplots()
sns.boxplot(x="rating", y="review_length_log", data=df, ax=ax)
ax.set_title("Log Review Length vs Rating")
st.pyplot(fig)

# =========================
# 🔹 REGRESSION
# =========================
st.markdown("### Linear Regression (OLS)")

model = smf.ols("rating ~ review_length_log", data=df).fit()

st.text(model.summary())

# =========================
# 🔹 INTERPRETATION
# =========================
st.markdown("### 💡 Interpretation")

st.warning("""
There is a statistically significant relationship:
➡️ As review length increases, rating slightly decreases.

However:
- The effect size is very small  
- Review length alone is NOT a strong predictor  
""")

st.success("""
✔ Conclusion:
Review length should NOT be used alone,  
but can be useful in combination with other features.
""")

st.subheader("Verified vs Rating Analysis")

# Crosstab (normalisiert)
st.markdown("### Distribution (Normalized)")
ct_norm = pd.crosstab(df["verified"], df["rating"], normalize="index")
st.dataframe(ct_norm)

# Crosstab (absolut, für Test)
st.markdown("### Absolute Counts")
ct = pd.crosstab(df["verified"], df["rating"])
st.dataframe(ct)

# Chi²-Test
chi2, p, _, _ = chi2_contingency(ct)

# Cramér’s V
n = ct.values.sum()
V = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

# Ergebnisse anzeigen
st.markdown("### Statistical Test Results")

st.write(f"**Chi² Statistic:** {chi2:.2f}")
st.write(f"**p-value:** {p:.5f}")
st.write(f"**Cramér's V:** {V:.2f}")

# Interpretation
if p < 0.05:
    st.success("✔ Significant relationship between 'verified' and 'rating'")
else:
    st.warning("⚠ No significant relationship found")

# Effektstärke interpretieren
if V < 0.1:
    strength = "weak"
elif V < 0.3:
    strength = "moderate"
else:
    strength = "strong"

st.info(f"Effect size (Cramér's V): **{strength} association**")

# Optional: Business Insight
st.markdown("""
### 💡 Insight
The feature **verified** shows a meaningful relationship with ratings.  
This may indicate potential differences in behavior between verified and non-verified reviews (e.g., fake reviews or biased feedback).
""")
st.markdown("""
Summary: Only the feature **verified** shows a strong correlation with the rating.
""")

# =========================
# 🔹 TEXT FEATURES
# =========================
st.subheader("Second Part: Analysis of review_text")

# =========================
# 🔹 SENTIMENT
# =========================
st.markdown("### Sentiment Feature")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if pd.isna(text):
        return None
    return analyzer.polarity_scores(text)["compound"]

df["sentiment"] = df["review_text"].apply(get_sentiment)

st.write("Sample data:")
st.dataframe(df[["review_text", "rating", "sentiment"]].head(20))

st.write("Correlation between rating and sentiment:")
st.write(df[["rating", "sentiment"]].corr())

fig1, ax1 = plt.subplots()
sns.boxplot(x="rating", y="sentiment", data=df, ax=ax1)
st.pyplot(fig1)

st.info("""
Sentiment score ranges from:
- +1 → very positive  
- 0 → neutral  
- -1 → very negative  
""")

st.success("✔ Sentiment is a strong predictive feature (~ +0.65 correlation)")

# =========================
# 🔹 NEGATION FEATURE
# =========================
st.markdown("### Negation Feature")

df["has_negation"] = df["review_text"].str.contains(r"\b(not|no|never)\b", case=False)

st.write("Correlation between rating and negation:")
st.write(df[["rating", "has_negation"]].corr())

st.write("Distribution:")
st.write(pd.crosstab(df["has_negation"], df["rating"], normalize="index"))

st.success("✔ Negation is also a strong feature (~ -0.57 correlation)")

# =========================
# 🔹 FINAL SUMMARY
# =========================
st.subheader("Summary")

st.markdown("""
We identified several strong features for predicting customer ratings:

- Sentiment score  
- Presence of negation  
- English text filtering  
- TF-IDF (planned)  
- Embeddings (planned)  
   
    Bereinigt einen Text:
    - Entfernt HTML-Tags
    - Lowercase
    - Entfernt Emojis und Sonderzeichen
    - Entfernt Zahlen
    - Lemmatization
    - Entfernt Stopwords 
      aber behält Negationen wie "not", "no", "never" für sentiment analysis!
    
    EMBEDDING_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            
    tfidf_dim=5000,
    emb_dim=384,
    struct_dim=4

These features show strong correlation with the target variable and will be used in the modeling phase.
""")