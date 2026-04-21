import streamlit as st
import pandas as pd
import os
from PIL import Image
import plotly.express as px
from sklearn.model_selection import train_test_split



# --- 1. HEADER & IMAGE ---
st.title("Modeling Phase")
st.header("Fake-News")

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
    st.subheader("⚠️Step 1: Attention: Preventing Data Leakage")

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





import pandas as pd
import random
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

# --- 1. Helper Function for Text Augmentation ---
def simple_augment(text):
    """Swaps two random words to slightly vary the text without changing its meaning."""
    if not isinstance(text, str): return text
    words = text.split()
    if len(words) < 3: return text # Too short to swap
    
    # Select two random indices to swap
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return " ".join(words)

# --- 2. Initial Data Split ---
st.divider()

# Features and Target
X = df[['review_text', 'verified']] 
y = df['target_group']

# Splitting before Resampling to keep the test set "unseen"
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# --- 3. Implementation of Resampling Strategy A ---
st.divider()
st.subheader("⚖️Step 2: Balancing Training Data (word switching augmentation)")

# Combine X and y temporarily for easier filtering
train_df = X_train_raw.copy()
train_df['target_group'] = y_train

target_count = 2000
balanced_frames = []

# Define categories to ensure correct ordering
category_order = ["Low (1-2 ⭐)", "Mid (3-4 ⭐)", "High (5 ⭐)"]

for group in train_df['target_group'].unique():
    subset = train_df[train_df['target_group'] == group]
    
    if "5 ⭐" in group:
        # A) UNDERSAMPLING: Reduce the 5-Star group to 2000
        subset_balanced = subset.sample(n=min(len(subset), target_count), random_state=42)
        st.write(f"**{group}**: Reduced from {len(subset)} to {len(subset_balanced)} (Undersampling).")
        
    else:
        # B) OVERSAMPLING + AUGMENTATION for Mid & Low
        how_many_to_add = target_count - len(subset)
        if how_many_to_add > 0:
            # Draw random rows (keeping the 'verified' status linked to the text)
            extras = subset.sample(n=how_many_to_add, replace=True, random_state=42)
            
            # Slightly vary the text in the extra rows
            extras['review_text'] = extras['review_text'].apply(simple_augment)
            
            # Combine original subset with augmented extras
            subset_balanced = pd.concat([subset, extras])
            st.write(f"**{group}**: Increased from {len(subset)} to {len(subset_balanced)} (Augmentation).")
        else:
            subset_balanced = subset
            
    balanced_frames.append(subset_balanced)

# Recombine all balanced groups and shuffle them
train_df_balanced = pd.concat(balanced_frames).sample(frac=1, random_state=42).reset_index(drop=True)

# Separate back into X and y for training
X_train_final = train_df_balanced[['review_text', 'verified']]
y_train_final = train_df_balanced['target_group']

# --- 4. Visualization of Results ---
col1, col2 = st.columns(2)
with col1:
    st.metric("Final Training Samples", len(y_train_final))
    fig_balanced = px.histogram(
        train_df_balanced, x="target_group", title="Balanced Training Set",
        category_orders={"target_group": category_order}, 
        color_discrete_sequence=['#FF7F0E']
    )
    st.plotly_chart(fig_balanced, use_container_width=True)

with col2:
    st.metric("Test Samples (Original)", len(y_test))
    # Test set visualization (should still show original distribution)
    fig_test = px.histogram(
        pd.DataFrame(y_test), x="target_group", title="Original Test Set",
        category_orders={"target_group": category_order}, color_discrete_sequence=['#636EFA']
    )
    st.plotly_chart(fig_test, use_container_width=True)
    st.info("Note: The test set remains untouched to ensure honest evaluation.")

# --- 5. Saving to Session State ---
st.session_state['train_test_split'] = {
    'X_train': X_train_final, 
    'X_test': X_test_raw, 
    'y_train': y_train_final, 
    'y_test': y_test
}
st.success("💡 **Resampling successful!** Balanced training data is now saved and ready for removing stopwords.")







#Ab hier werden die Kommentare für das Machine Learning Modell vorbereitet.
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- 1. Vorbereitung (Downloads) ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab') # Für neuere NLTK Versionen oft nötig

def preprocess_text_full(text):
    if not isinstance(text, str):
        return ""
    
    # STEP 1: Lowercase
    text = text.lower()
    
    # STEP 2: Regex (Satzzeichen und Zahlen entfernen)
    # Wir machen das VOR den Stopwörtern, um "saubere" Wörter zu erhalten
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Behält nur Buchstaben und Leerzeichen
    text = re.sub(r'\d+', '', text)        # Entfernt restliche Zahlen
    
    # STEP 3: Tokenize
    tokens = word_tokenize(text)
    
    # STEP 4: Stopword Filtering
    stop_words = set(stopwords.words('english'))
    # Option: zusätzliche Stopwörter hinzufügen
    new_stop_words = [",", ".", "``", "@", "*", "(", ")", "...", "!", "?", "-", "_", ">", "<", ":", "/", "=", "--", "©", "~", ";", "\\", "\\\\"]
    stop_words.update(new_stop_words)
    
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # STEP 5: Re-Join (Zurückbauen zum String)
    # WICHTIG: TfidfVectorizer braucht einen String, keine Liste!
    return " ".join(filtered_tokens)

# --- 2. Anwendung auf die Daten ---
st.divider()
st.subheader("🧪Step 3 Advanced Text Preprocessing")

if 'train_test_split' in st.session_state:
    with st.spinner("Processing: Lowercase -> Regex -> Tokenize -> Filter -> Join..."):
        # Wir laden die Daten aus dem Session State
        X_train = st.session_state['train_test_split']['X_train']
        X_test = st.session_state['train_test_split']['X_test']
        
        # Anwendung der gesamten Kette
        X_train['clean_review'] = X_train['review_text'].apply(preprocess_text_full)
        X_test['clean_review'] = X_test['review_text'].apply(preprocess_text_full)
        
        # Ergebnisse zurück in den Session State
        st.session_state['train_test_split']['X_train'] = X_train
        st.session_state['train_test_split']['X_test'] = X_test

    st.success("All steps completed! Your text is now 'Vectorization-ready'.")
    
    # Vergleichs-Vorschau
    st.write("Comparison (Original vs. Processed):")
    st.dataframe(X_train[['review_text', 'clean_review']].head(5))
else:
    st.error("No data found. Please run the previous steps first.")





st.divider()

from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

st.divider()
st.subheader("🔢 Step 4: Vectorization & Feature Combination")

if 'train_test_split' in st.session_state:
    # 1. TF-IDF auf den Text anwenden
    tfidf = TfidfVectorizer(max_features=5000)
    
    # Text-Vektoren erzeugen (Sparse Matrizen)
    X_train_text = tfidf.fit_transform(X_train['clean_review'])
    X_test_text = tfidf.transform(X_test['clean_review'])

    # 2. Die 'verified'-Spalte vorbereiten
    # Wir müssen sie in ein Format bringen, das mit der Sparse-Matrix kompatibel ist
    X_train_verified = sp.csr_matrix(X_train[['verified']].values)
    X_test_verified = sp.csr_matrix(X_test[['verified']].values)

    # 3. Text-Vektoren und Verified-Status zusammenfügen (hstack)
    X_train_final_model = sp.hstack((X_train_text, X_train_verified))
    X_test_final_model = sp.hstack((X_test_text, X_test_verified))

    # 4. Speichern für das Modell
    st.session_state['tfidf_data'] = {
        'X_train': X_train_final_model,
        'X_test': X_test_final_model,
        'vectorizer': tfidf
    }
    # Anzeige der Matrix-Dimensionen (Zeilen x Spalten)
    rows_train = X_train_final_model.shape[0]
    cols_train = X_train_final_model.shape[1]

    st.success(f"✅ Vectorization complete!")
    st.write(f"**Matrix Dimensions:** {rows_train} samples (rows, each rating of 2000) × {cols_train} features (columns, words + verified).")









from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.figure_factory as ff
import time

st.divider()
st.subheader("🤖 Step 5: Model Training & Evaluation")

if 'tfidf_data' in st.session_state and 'train_test_split' in st.session_state:
    # 1. Daten aus dem Session State laden
    X_train = st.session_state['tfidf_data']['X_train']
    X_test = st.session_state['tfidf_data']['X_test']
    y_train = st.session_state['train_test_split']['y_train']
    y_test = st.session_state['train_test_split']['y_test']

    # 2. Modell initialisieren
    # Wir nehmen moderate Standardwerte für eine gute Balance zwischen Speed und Power
    model = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )

    # 3. Training mit Zeitmessung
    with st.spinner("Training Gradient Boosting Model... this might take a minute."):
        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time
    
    st.success(f"✅ Training finished in {duration:.2f} seconds!")

    # 4. Vorhersage (Prediction)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # 5. Metriken anzeigen
    st.metric("Model Accuracy", f"{accuracy:.2%}")

    # 6. Confusion Matrix (Visualisierung)
    st.write("### Confusion Matrix")

    # Definieren der gewünschten Reihenfolge
    ordered_labels = ["Low (1-2 ⭐)", "Mid (3-4 ⭐)", "High (5 ⭐)"]

    # Confusion Matrix mit expliziten Labels berechnen
    cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)

    # Heatmap erstellen
    fig_cm = ff.create_annotated_heatmap(
        z=cm[::-1],            # Matrix umkehren für korrekte Y-Anzeige
        x=ordered_labels, 
        y=ordered_labels[::-1], # Labels umkehren
        colorscale='Viridis', 
        showscale=True
    )

    # Achsenbeschriftung optimieren
    fig_cm.update_layout(
        xaxis_title="**Predicted rating**", 
        yaxis_title="**True rating**",
        xaxis=dict(side="top") # Schiebt die Labels der X-Achse nach oben für bessere Lesbarkeit
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    # 7. Detaillierter Report
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Modell im Session State für spätere Vorhersagen speichern
    st.session_state['final_model'] = model

else:
    st.error("No vectorized data found. Please complete the previous steps.")




st.markdown(f'<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
st.write("Something to optimize?")

col_code1, col_code2 = st.columns(2)

with col_code1:
    st.caption("Standard Model (Base)")
    st.code("""
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
    """, language="python")

with col_code2:
    st.caption("Optimized Model (Tuned)")
    st.code("""
model = GradientBoostingClassifier(
    n_estimators=150,   # More trees
    learning_rate=0.1,
    max_depth=5,        # Deeper trees
    subsample=0.8,      # Better generalization.
    random_state=42
)
    """, language="python")










#Optimierung des Modells mit Hyperparameter Tuning (GridSearchCV)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import plotly.figure_factory as ff
import time
import pandas as pd

st.divider()
st.subheader("🤖 Step 5 (second try): Optimized Model Training & Evaluation")

if 'tfidf_data' in st.session_state and 'train_test_split' in st.session_state:
    X_train = st.session_state['tfidf_data']['X_train']
    X_test = st.session_state['tfidf_data']['X_test']
    y_train = st.session_state['train_test_split']['y_train']
    y_test = st.session_state['train_test_split']['y_test']

    # --- OPTIMIERUNG 1: Class Weights berechnen ---
    # Da "Mid" oft schwer zu greifen ist, helfen Gewichte dem Modell, 
    # diese Klasse ernster zu nehmen.
    #weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # --- OPTIMIERUNG 2: Stärkeres Modell-Setup ---
    model = GradientBoostingClassifier(
        n_estimators=150,      # Mehr Bäume für feinere Nuancen (vorher 100)
        learning_rate=0.1, 
        max_depth=5,           # Tiefere Bäume, um komplexere Wort-Kombinationen zu verstehen (vorher 3)
        subsample=0.8,         # Nutzt nur 80% der Daten pro Baum (verhindert Overfitting)
        random_state=42
    )

    with st.spinner("Training Optimized Model... this may take a bit longer."):
        start_time = time.time()
        # Wir trainieren das Modell mit den berechneten Gewichten
        model.fit(X_train, y_train)
        duration = time.time() - start_time
    
    st.success(f"✅ Optimized Training finished in {duration:.2f} seconds!")

    # Vorhersage
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Model Accuracy", f"{accuracy:.2%}")

    # --- OPTIMIERUNG 3: Korrekte Matrix-Reihenfolge (Top-Down) ---
    st.write("### Confusion Matrix")
    ordered_labels = ["Low (1-2 ⭐)", "Mid (3-4 ⭐)", "High (5 ⭐)"]
    
    # Für die Heatmap drehen wir die Y-Achse um, damit 1-2 oben links steht
    cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)
    
    fig_cm = ff.create_annotated_heatmap(
        z=cm[::-1],            # Matrix umkehren für korrekte Y-Anzeige
        x=ordered_labels, 
        y=ordered_labels[::-1], # Labels umkehren
        colorscale='Viridis', 
        showscale=True
    )

    fig_cm.update_layout(
        xaxis_title="**Predicted rating**", 
        yaxis_title="**True rating**",
        xaxis=dict(side="top")
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.session_state['final_model'] = model