import streamlit as st
import pandas as pd
import os
from PIL import Image
import plotly.express as px
from sklearn.model_selection import train_test_split

import sys
import subprocess
import os

# 1. Versuch: Paket im Standardpfad oder im User-Pfad suchen
user_site = os.path.expanduser("~/.local/lib/python3.12/site-packages")
if user_site not in sys.path:
    sys.path.append(user_site)

# 2. Versuch: Wenn Import scheitert, sofortige Installation erzwingen
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
    from imblearn.over_sampling import SMOTE

# Wenn wir hier ankommen, ist SMOTE bereit!
st.success("✅ Module 'imbalanced-learn' is ready!")


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

    # --- NEU: Implementierung von Strategie A (Ziel: ca. 2000 pro Gruppe) ---
    st.divider()
    st.subheader("⚖️ Balancing Training Data (Strategy A)")

    # Wir führen X und y kurz zusammen, um leichter zu filtern
    train_df = X_train_raw.copy()
    train_df['target_group'] = y_train

    target_count = 2000
    balanced_frames = []

    for group in train_df['target_group'].unique():
        subset = train_df[train_df['target_group'] == group]
        
        if group == 'High (5 ⭐)':
            # 1. Undersampling für High
            subset_balanced = subset.sample(n=min(len(subset), target_count), random_state=42)
        else:
            # 2. Oversampling für Mid und Low
            # (Hier kannst du später deine Synonym-Funktionen einbauen)
            how_many_to_add = target_count - len(subset)
            if how_many_to_add > 0:
                # Aktuell: Einfaches Kopieren (Random Oversampling)
                extras = subset.sample(n=how_many_to_add, replace=True, random_state=42)
                subset_balanced = pd.concat([subset, extras])
            else:
                subset_balanced = subset
        
        balanced_frames.append(subset_balanced)

    # Zusammenführen und neu mischen
    train_df_balanced = pd.concat(balanced_frames).sample(frac=1, random_state=42)
    
    # Zurück trennen in X_train und y_train
    X_train_final = train_df_balanced.drop(columns=['target_group'])
    y_train_final = train_df_balanced['target_group']

    # --- Visualisierung des neuen Status ---
    st.success(f"Resampling abgeschlossen! Jede Klasse hat nun ca. {target_count} Samples.")
    
    fig_balanced = px.histogram(
        train_df_balanced, x="target_group", 
        title="Balanced Training Set (New)",
        category_orders={"target_group": category_order},
        color_discrete_sequence=['#FF7F0E'] # Andere Farbe zur Unterscheidung
    )
    st.plotly_chart(fig_balanced, use_container_width=True)

    # WICHTIG: Die BALANCIERTEN Daten im Session State speichern
    st.session_state['train_test_split'] = {
        'X_train': X_train_final, 
        'X_test': X_test_raw, # Testdaten bleiben unangetastet!
        'y_train': y_train_final, 
        'y_test': y_test
    }



