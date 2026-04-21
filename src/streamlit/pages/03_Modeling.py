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
st.subheader("⚠️ Attention: Preventing Data Leakage")

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
st.subheader("⚖️ Balancing Training Data (Strategy A)")

# Combine X and y temporarily for easier filtering
train_df = X_train_raw.copy()
train_df['target_group'] = y_train

target_count = 2000
balanced_frames = []

# Define categories to ensure correct ordering
category_order = ["High (5 ⭐)", "Mid (3-4 ⭐)", "Low (1-2 ⭐)"]

for group in train_df['target_group'].unique():
    subset = train_df[train_df['target_group'] == group]
    
    if "5 ⭐" in group:
        # A) UNDERSAMPLING: Reduce the 5-Star group to 2000
        subset_balanced = subset.sample(n=min(len(subset), target_count), random_state=42)
        st.write(f"✅ **{group}**: Reduced from {len(subset)} to {len(subset_balanced)} (Undersampling).")
        
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
            st.write(f"🚀 **{group}**: Increased from {len(subset)} to {len(subset_balanced)} (Augmentation).")
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
st.success("💡 **Resampling successful!** Balanced training data is now saved and ready for vectorization.")

