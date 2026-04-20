import streamlit as st
import pandas as pd
import os
from PIL import Image
import plotly.express as px
from sklearn.model_selection import train_test_split

# --- 1. HEADER & IMAGE ---
st.title("Modeling Phase")
st.header("Fake-News Analysis")

img_path = "src/streamlit/static/Analysis.png"
if os.path.exists(img_path):
    img = Image.open(img_path)
    st.image(img, use_container_width=True)
else:
    st.warning(f"Analysis image not found at: {img_path}")

st.write("---")

# --- 2. DATA INTEGRATION & CLEANING (Der korrigierte Block) ---
if 'ml_data' in st.session_state:
    # Daten einmalig laden
    df = st.session_state['ml_data'].copy()
    
    st.subheader("1. Data Integration & Advanced Cleaning")
    
    # A. Duplikat-Analyse (Normalisiert)
    raw_extra_rows = df.duplicated().sum()
    df['temp_clean'] = df['review_text'].astype(str).str.lower().str.strip()
    normalized_extra_rows = df.duplicated(subset=['temp_clean', 'rating']).sum()
    
    # Reinigung: Nur Unikate behalten
    df = df.drop_duplicates(subset=['temp_clean', 'rating'], keep='first')
    df = df.drop(columns=['temp_clean']) 
    
    # B. Qualitäts-Checks (NaNs & Replies)
    nan_count = df['review_text'].isna().sum()
    reply_mask = df['review_text'].str.contains("reply from", case=False, na=False)
    reply_count = reply_mask.sum()
    
    # Finale Reinigungsausführung
    df = df.dropna(subset=['review_text', 'rating'])
    df = df[~reply_mask]
    
    st.success(f"✅ Cleanup finished! Exact dups: {raw_extra_rows}, Normalized dups: {normalized_extra_rows}, NaNs: {nan_count}, Replies: {reply_count}")
    
    # Preview der sauberen Daten
    st.write("### Preview: Processed Data (Top 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)

    # --- 3. RATING DISTRIBUTION & GROUPING ---
    st.subheader("2. Rating Distribution & Grouping")

    # Gruppierungslogik
    def group_ratings(rating):
        if rating <= 2: return "Low (1-2 ⭐)"
        elif rating <= 4: return "Mid (3-4 ⭐)"
        else: return "High (5 ⭐)"

    df['target_group'] = df['rating'].apply(group_ratings)

    # Visualisierung der Verteilung
    category_order = ["Low (1-2 ⭐)", "Mid (3-4 ⭐)", "High (5 ⭐)"]
    fig_grouped = px.histogram(
        df, x="target_group", 
        title="Grouped Rating Distribution",
        category_orders={"target_group": category_order},
        color_discrete_sequence=['#00CC96']
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

    # --- 4. TRAIN-TEST SPLIT (Data Leakage Protection) ---
    st.divider()
    st.subheader("⚠️ Attention: Preventing Data Leakage")

    X = df[['review_text', 'verified']] 
    y = df['target_group']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )

    # Visualisierung Split
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples (80%)", len(y_train))
        fig_train = px.histogram(
            pd.DataFrame(y_train), x="target_group", title="Training Set",
            category_orders={"target_group": category_order}, color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig_train, use_container_width=True)

    with col2:
        st.metric("Test Samples (20%)", len(y_test))
        fig_test = px.histogram(
            pd.DataFrame(y_test), x="target_group", title="Test Set",
            category_orders={"target_group": category_order}, color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig_test, use_container_width=True)

    # WICHTIG: Alles im Session State für die nächste Seite speichern
    st.session_state['df_modeling_final'] = df
    st.session_state['train_test_split'] = {
        'X_train': X_train_raw, 
        'X_test': X_test_raw, 
        'y_train': y_train, 
        'y_test': y_test
    }
    st.info("💡 **Ready for Vektorization!** The split data is saved in session state.")

else:
    st.error("❌ No processed data found in memory!")
    st.info("Please run the **'02 Preprocessing'** page first.")




from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

st.divider()
st.header("4. Class Balancing with SMOTE")

# --- SCHRITT 1: Vektorisierung (Nötig für SMOTE) ---
# SMOTE braucht Zahlen. Wir wandeln den Text kurz in eine Matrix um.
tfidf_temp = TfidfVectorizer(max_features=1000) 
X_train_tfidf = tfidf_temp.fit_transform(X_train_raw['review_text'])

# --- SCHRITT 2: SMOTE anwenden ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

# --- SCHRITT 3: Visualisierung ---
st.write("### Comparison: Before vs. After SMOTE")

# Wir bereiten die Daten für Plotly vor
df_before = pd.DataFrame({'target_group': y_train, 'Status': 'Original (Imbalanced)'})
df_after = pd.DataFrame({'target_group': y_resampled, 'Status': 'After SMOTE (Balanced)'})
df_combined = pd.concat([df_before, df_after])

# Das kombinierte Histogramm
fig_smote = px.histogram(
    df_combined, 
    x="target_group", 
    color="Status", 
    barmode="group",
    title="Effect of SMOTE on Training Data",
    category_orders={"target_group": ["Low (1-2 ⭐)", "Mid (3-4 ⭐)", "High (5 ⭐)"]},
    color_discrete_map={'Original (Imbalanced)': '#636EFA', 'After SMOTE (Balanced)': '#00CC96'}
)

st.plotly_chart(fig_smote, use_container_width=True)

# --- SCHRITT 4: Metriken ---
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Samples Before", len(y_train))
with col2:
    st.metric("Total Samples After", len(y_resampled))

st.info("💡 **What happened?** SMOTE created synthetic examples for the 'Low' and 'Mid' classes so they now match the count of the 'High' class.")