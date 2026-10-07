import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import streamlit as st
from scipy.stats import chi2_contingency
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from _utils import load_processed, load_raw

st.set_page_config(layout="wide")
st.title("📊 Feature Engineering")


def chi2_test(feature: pd.Series, rating: pd.Series):
    """Chi² test plus Cramér's V for a categorical feature against the rating."""
    table = pd.crosstab(feature, rating)
    chi2, p, _, _ = chi2_contingency(table)
    v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
    return chi2, p, v


def show_chi2(feature, rating):
    chi2, p, v = chi2_test(feature, rating)
    st.write(f"Chi²: {chi2:.2f} | p-value: {p:.5f} | Cramér’s V: {v:.2f}")
    return chi2, p, v


def small_boxplot(data, x, y, title):
    fig, ax = plt.subplots(figsize=(4, 2.5))
    sns.boxplot(x=x, y=y, data=data, ax=ax)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    st.pyplot(fig, width="content")


@st.cache_data(show_spinner="Computing sentiment scores…")
def sentiment_scores(texts: pd.Series) -> pd.Series:
    analyzer = SentimentIntensityAnalyzer()
    return texts.apply(lambda t: None if pd.isna(t) else analyzer.polarity_scores(t)["compound"])


# =========================
# Raw data & methods
# =========================
st.subheader("Raw Data")
st.dataframe(load_raw().head(20))
st.write("""
The scraped raw data consists of eight columns:
review_text, rating_svg, date, location, supplier_response, verified, company
""")

st.subheader("Statistical Tests")
st.markdown("#### Correlation : Numerical variables")
st.write("Correlation gives a value between -1 and +1 and has direction")
st.table(pd.DataFrame({
    "Correlation value": ["0.00-0.10", "0.10-0.30", "0.30-0.50", "0.50-0.70", "0.70-1.00"],
    "Relationship": ["none", "weak", "moderate", "strong", "very strong"],
}))
st.write("For Feature Engineering: relevance in praxis")
st.table(pd.DataFrame({
    "Correlation value": ["<0.10", "0.10-0.30", "0.30-0.50", "0.50-1.00"],
    "Recommendation of usage": ["ignore", "maybe useful", "good feature", "strong feature"],
}))

st.markdown("#### Chi2 + Cramers V: Categorical variables")
st.write("Chi2 test: If p-value <0.05 --> significant relationship")
st.write("Cramers V: Measures how strong the relationship is")
st.table(pd.DataFrame({
    "Cramers V": ["0.00-0.10", "0.10-0.30", "0.30-0.50", "0.50-1.00"],
    "Strength of relationship": ["very weak", "weak to moderate", "moderate to strong", "strong"],
}))
st.write("Recommendation of usage: Useful feature if p < 0.05 and Cramers V > 0.2 , "
         "Strong feature if p < 0.05 and Cramers V > 0.5")

st.markdown("### TF-IDF, Embeddings, Sentiment and review_text_en: Text Features")
st.markdown("""
#### TF-IDF
- tfidf_dim = 5000
- Measures how important a word is in a review
- High score = frequent in one review, rare in all reviews
- Fast and simple, but does not understand meaning or context""")
with st.expander("TF-IDF"):
    st.markdown("""
    - Not every word has the same importance
    - A word gets a high TF-IDF value if:
        - it is frequent in the actual review
        - it is seldom in the whole dataset of reviews
    - Disadvantages:
        - does not understand meaning
        - no ordering of words
        - often creates a large sparse matrix
        - TF-IDF only understands words""")

st.markdown("""
#### Embeddings
- EMBEDDING_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2") with emb_dim = 384
- Convert text into numeric vectors with semantic meaning
- Capture context, similarity, and relationships between words
- More accurate and multilingual, but computationally expensive
""")
with st.expander("Embeddings"):
    st.markdown("""
    - Numeric vectors that represent the meaning of texts
    - Embeddings understand: meaning, context, similarity
    - Text gets embedded in a meaning space:
        - similar words are near each other
        - different words are far apart
    - We use Sentence Transformers with 384 dimensions
    - Advantages: much better semantics, more robust, multilingual, better performance
    - Disadvantages: output is harder to interpret, requires more computational power
    """)

st.markdown("""
#### Sentiment Analysis
- Detects emotional tone in text
- Classifies reviews as positive, negative, or neutral
- Useful for understanding customer opinions, but struggles with sarcasm and complex context
""")
with st.expander("Sentiment"):
    st.markdown("""
    - Detects the emotional tone of a text
    - Determines whether a review is positive, negative or neutral
    - Helps to understand customer opinions and satisfaction
    - Can identify patterns in feedback and trends in reviews
    - Often used together with TF-IDF or embeddings for better predictions
    - Advantages: easy to interpret, useful for customer feedback analysis, adds emotional context to text data
    - Disadvantages: may misunderstand sarcasm or irony, sensitive to context and wording,
      performance depends on the quality of the model
    """)

# =========================
# Time features
# =========================
st.subheader("External Feature Analysis")

df = load_processed().dropna(subset=["date"]).reset_index(drop=True)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

st.subheader("📊 Year vs Rating")
st.dataframe(pd.crosstab(df["year"], df["rating"], normalize="index"))
_, p, _ = show_chi2(df["year"], df["rating"])
if p < 0.05:
    st.success("✔ Significant relationship between year and rating")
else:
    st.warning("⚠ No significant relationship")
st.info("➡️ Interpretation: statistically significant but weak relationship (~0.14)")

with st.expander("📊 Time-based Feature Analysis"):
    st.subheader("📈 Rating Trend over Time")
    fig, ax = plt.subplots()
    df.groupby("year")["rating"].mean().plot(ax=ax)
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

    st.subheader("⏳ Review Age Feature")
    df["review_age_days"] = (df["date"].max() - df["date"]).dt.days
    fig, ax = plt.subplots()
    sns.histplot(df["review_age_days"], bins=50, ax=ax)
    st.pyplot(fig)
    fig, ax = plt.subplots()
    sns.boxplot(x="rating", y="review_age_days", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("📦 Age Bucket Analysis")
    df["age_bucket"] = pd.cut(df["review_age_days"], bins=[0, 30, 180, 365, 10000],
                              labels=["0-30d", "1-6m", "6-12m", "1y+"])
    st.dataframe(pd.crosstab(df["age_bucket"], df["rating"], normalize="index"))
    show_chi2(df["age_bucket"], df["rating"])

    st.subheader("📉 Time Trend (Regression)")
    df["number_of_months"] = (df["year"] - df["year"].min()) * 12 + df["month"]
    st.text(smf.ols("rating ~ number_of_months", data=df).fit().summary())
    st.info("➡️ Weak but significant negative trend over time")

    st.subheader("📅 Month Analysis")
    st.dataframe(pd.crosstab(df["month"], df["rating"], normalize="index"))
    st.write(f"Cramér’s V: {chi2_test(df['month'], df['rating'])[2]:.2f}")
    st.success("✔ Months show meaningful variation → useful feature")

    st.subheader("🌦 Season Feature")
    seasons = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
               6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    show_chi2(df["month"].map(seasons), df["rating"])

    st.subheader("🎯 Special Time Features")
    for feature, values in [("is_year_end", df["month"].isin([11, 12])), ("is_march", df["month"] == 3)]:
        st.markdown(f"### {feature}")
        show_chi2(values.astype(int), df["rating"])

# =========================
# Review length
# =========================
st.subheader("📝 Review Length vs Rating")
st.markdown("### Correlation")
st.write(df[["review_length", "rating"]].corr())
st.info("""
Moderate negative correlation (~ -0.44):
➡️ Longer reviews tend to be associated with lower ratings.
""")

st.markdown("### Visualization")
small_boxplot(df, "rating", "review_length", "review length vs rating")

st.markdown("### Log Transformation")
df["review_length_log"] = np.log1p(df["review_length"])
small_boxplot(df, "rating", "review_length_log", "Log Review Length vs Rating")

st.markdown("### Linear Regression (OLS)")
st.text(smf.ols("rating ~ review_length_log", data=df).fit().summary())

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

# =========================
# Verified
# =========================
st.subheader("Verified vs Rating Analysis")
st.markdown("### Distribution (Normalized)")
st.dataframe(pd.crosstab(df["verified"], df["rating"], normalize="index"))
st.markdown("### Absolute Counts")
st.dataframe(pd.crosstab(df["verified"], df["rating"]))

chi2, p, v = chi2_test(df["verified"], df["rating"])
st.markdown("### Statistical Test Results")
st.write(f"**Chi² Statistic:** {chi2:.2f}")
st.write(f"**p-value:** {p:.5f}")
st.write(f"**Cramér's V:** {v:.2f}")
if p < 0.05:
    st.success("✔ Significant relationship between 'verified' and 'rating'")
else:
    st.warning("⚠ No significant relationship found")
strength = "weak" if v < 0.1 else "moderate" if v < 0.3 else "strong"
st.info(f"Effect size (Cramér's V): **{strength} association**")

st.markdown("""
### 💡 Insight
The feature **verified** shows a meaningful relationship with ratings.
This may indicate potential differences in behavior between verified and non-verified reviews (e.g., fake reviews or biased feedback).

Summary: Only the feature **verified** shows a strong correlation with the rating.
""")

# =========================
# Text features
# =========================
st.subheader("Analysis of review_text")
st.markdown("### Sentiment Feature")
df["sentiment"] = sentiment_scores(df["review_text"])
st.write("Sample data:")
st.dataframe(df[["review_text", "rating", "sentiment"]].head(20))
st.write("Correlation between rating and sentiment:")
st.write(df[["rating", "sentiment"]].corr())
small_boxplot(df, "rating", "sentiment", "Sentiment vs Rating")
st.info("""
Sentiment score ranges from:
- +1 → very positive
- 0 → neutral
- -1 → very negative
""")
st.success("✔ Sentiment is a strong predictive feature (~ +0.65 correlation)")

st.markdown("### Negation Feature")
df["has_negation"] = df["review_text"].str.contains(r"\b(not|no|never)\b", case=False, na=False)
st.write("Correlation between rating and negation:")
st.write(df[["rating", "has_negation"]].corr())
st.write("Distribution:")
st.write(pd.crosstab(df["has_negation"], df["rating"], normalize="index"))
st.success("✔ Negation is also a strong feature (~ -0.57 correlation)")

st.subheader("Choice of features")
st.table(pd.DataFrame({
    "feature": ["sentiment", "has_negation", "review_length", "verified"],
    "correlation": ["+0.65", "-0.57", "-0.44", ""],
    "Cramers V": ["", "", "", "+0.28"],
    "feature strength": ["strong", "strong", "moderate", "moderate"],
}))
st.table(pd.DataFrame({
    "feature": ["embedding", "tf-idf", "review_text_en"],
    "task": ["meaning", "words", "helps sentiment, has_negation, Tf-idf"],
}))

st.subheader("➡️ How these features are used in the model")
st.markdown("""
- **verified** is part of every model (first, second and third try).
- **sentiment** and **has_negation** are added in the **third try** on the Modelling page, together with bi-grams,
  to measure how much they improve the prediction compared to plain TF-IDF.
- **review_length** and the time features are not used: their relationship with the rating is weak
  (small Cramér's V / effect size).
- **Embeddings** were not used, to keep the model fast enough to run live in this app.
""")
st.page_link("pages/04_Modelling.py", label="Continue to Modelling", icon="➡️")
