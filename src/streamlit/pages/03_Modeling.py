import streamlit as st
import pandas as pd
import os
from PIL import Image

# 1. Page Header & Image Analysis
st.title("Modeling Phase")
st.header("Fake-News")

# Display the analysis image from your static folder
img_path = "src/streamlit/static/Analysis.png"

if os.path.exists(img_path):
    img = Image.open(img_path)
    st.image(img, use_container_width=True)
else:
    st.warning(f"Analysis image not found at: {img_path}")

st.write("---")

# 2. Data Access from Session State
st.subheader("Data Integration")

if 'ml_data' in st.session_state:
    # Access the dataframe created in 02_Preprocessing.py
    df = st.session_state['ml_data']
    
    st.success("✅ Cleaned dataset successfully loaded from Session State!")
    
    # Display the top 15 rows as requested
    st.write("### Preview: Processed Data (Top 15 Rows)")
    st.dataframe(df.head(15), use_container_width=True)
    
    # Display dataset dimensions for confirmation
    st.info(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

else:
    # Error handling if the user skips the preprocessing step
    st.error("❌ No processed data found in memory!")
    st.info("Please run the **'02 Preprocessing'** page first to prepare the dataset for modeling.")
    
    # Optional: Fallback to load the CSV directly if session state is empty
    # if st.button("Load latest CSV as fallback"):
    #     df = pd.read_csv("src/data/clean/reviews_clean.csv")
    #     st.dataframe(df.head(15))





st.subheader("2. Duplicate Analysis")

# 1. Vorher-Wert berechnen (Exakte Duplikate)
raw_extra_rows = df.duplicated().sum()

# 2. Normalisierung durchführen (alles klein & Leerzeichen weg)
# Wir erstellen eine temporäre Spalte für den gründlichen Check
df['temp_clean'] = df['review_text'].str.lower().str.strip()
normalized_extra_rows = df.duplicated(subset=['temp_clean', 'rating']).sum()

if normalized_extra_rows > 0:
    # Dein gewünschter Satz mit den dynamischen Zahlen
    st.write(f"""
        Initially, we found **{raw_extra_rows}** exact duplicates. 
        Now, we **normalize the text** (converting everything to lowercase and removing extra spaces) and check again: 
        We found **{normalized_extra_rows}** rows that are potential duplicates.
    """)
    
    # Anzeige der Duplikate (basierend auf der sauberen Spalte)
    duplicates_df = df[df.duplicated(subset=['temp_clean', 'rating'], keep=False)]
    st.write("Preview of duplicate rows (after normalization):")
    st.dataframe(duplicates_df.head(10), use_container_width=True)
    
    # Reinigung: Wir behalten nur die einzigartigen Zeilen
    before_count = len(df)
    df = df.drop_duplicates(subset=['temp_clean', 'rating'], keep='first')
    df = df.drop(columns=['temp_clean']) # Hilfsspalte wieder löschen
    after_count = len(df)
    
    st.info(f"Cleanup finished: {before_count - after_count} rows removed. Unique rows remaining: {after_count}")
else:
    df = df.drop(columns=['temp_clean'])
    st.success("No duplicates found, even after normalization!")









# 3. Data Quality Checks & Grouping Logic
import plotly.express as px

st.title("Modeling Phase")

if 'ml_data' in st.session_state:
    df = st.session_state['ml_data'].copy()

    # --- STEP 1: DATA QUALITY CHECKS ---
    st.subheader("1. Data Quality & Cleaning")
    
    # Check for NaNs and Duplicates
    nan_count = df['review_text'].isna().sum()
    dup_count = df.duplicated().sum()
    
    # Check for "Reply from" (case insensitive)
    # We filter out rows that contain the string "reply from"
    reply_mask = df['review_text'].str.contains("reply from", case=False, na=False)
    reply_count = reply_mask.sum()
    
    # Cleaning execution
    df = df.dropna(subset=['review_text', 'rating'])
    df = df.drop_duplicates()
    df = df[~reply_mask]

    # Confirmation Message
    st.success(f"✅ Data verified: {nan_count} NaNs removed, {dup_count} duplicates deleted, and {reply_count} 'replies' filtered out.")

    # --- STEP 2: RATING PROPORTIONS & GROUPING ---
    st.subheader("2. Rating Distribution & Grouping")

    # Show Original Distribution
    fig_orig = px.histogram(df, x="rating", title="Original Rating Distribution", 
                            nbins=5, color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_orig, use_container_width=True)

    # Grouping logic: 1-2 -> Low, 3-4 -> Mid, 5 -> High
    def group_ratings(rating):
        if rating <= 2: return "Low (1-2)"
        elif rating <= 4: return "Mid (3-4)"
        else: return "High (5)"

    df['rating_group'] = df['rating'].apply(group_ratings)

    # Show Grouped Distribution
    fig_grouped = px.histogram(df, x="rating_group", title="Grouped Rating Distribution",
                               category_orders={"rating_group": ["Low (1-2)", "Mid (3-4)", "High (5)"]},
                               color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig_grouped, use_container_width=True)

    # Save cleaned and grouped DF back to session state for the next step
    st.session_state['df_modeling'] = df
    st.info(f"Dataset is ready for Sampling. Current shape: {df.shape[0]} rows.")

else:
    st.error("Please run the Preprocessing page first!")






# 4. Preventing Data Leakage: Target Creation & Train-Test Split
st.divider()
st.divider()
st.subheader("⚠️ Attention: Preventing Data Leakage")

# --- 1. DATA PREPARATION & GROUPING ---
def categorize_rating(r):
    if r <= 2: return "Low (1-2 ⭐)"
    elif r <= 4: return "Mid (3-4 ⭐)"
    else: return "High (5 ⭐)"

# Create the target column before splitting
df['target_group'] = df['rating'].apply(categorize_rating)

# --- 2. TRAIN-TEST SPLIT ---
from sklearn.model_selection import train_test_split

X = df[['review_text', 'verified']] 
y = df['target_group']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

st.success("Target groups created and data successfully split!")

# --- 3. VISUALIZATION: METRICS & DISTRIBUTIONS ---
st.write("### Train vs. Test Distribution")

# Define the correct order for the X-axis
category_order = ["Low (1-2 ⭐)", "Mid (3-4 ⭐)", "High (5 ⭐)"]

# Create two main columns for the entire layout
col1, col2 = st.columns(2)

with col1:
    # Metric at the top of the column
    st.metric("Training Samples (80%)", len(y_train))
    
    # Training distribution plot
    fig_train = px.histogram(
        pd.DataFrame(y_train), 
        x="target_group", 
        title="Training Set Distribution",
        category_orders={"target_group": category_order},
        color_discrete_sequence=['#00CC96']
    )
    fig_train.update_xaxes(categoryorder='array', categoryarray=category_order)
    st.plotly_chart(fig_train, use_container_width=True)

with col2:
    # Metric at the top of the column
    st.metric("Test Samples (20%)", len(y_test))
    
    # Test distribution plot
    fig_test = px.histogram(
        pd.DataFrame(y_test), 
        x="target_group", 
        title="Test Set Distribution",
        category_orders={"target_group": category_order},
        color_discrete_sequence=['#636EFA']
    )
    fig_test.update_xaxes(categoryorder='array', categoryarray=category_order)
    st.plotly_chart(fig_test, use_container_width=True)

st.info("💡 **Observation:** Notice how both sets maintain the same proportions. This is thanks to the 'stratify' parameter.")
