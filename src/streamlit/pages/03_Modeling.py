import streamlit as st

# 1. Load data from session state
if 'ml_data' in st.session_state:
    df = st.session_state['ml_data']
    st.success(f"✅ Dataset with {df.shape[0]} rows loaded successfully!")

    # 2. The Expander (as requested, in English)
    with st.expander("🔍 View Raw Data Columns"):
        st.write("Current columns in our dataset:")
        # We use 'df' here instead of 'df_processed'
        remaining_cols = list(df.columns)
        st.code(f"{remaining_cols}")

        # 3. Preview of the first 10 rows
        st.write("### 📋 Data Preview (Top 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

else:
    st.error("⚠️ No data found. Please run the Preprocessing first!")

st.markdown("---")







st.write("### 🔍 Text Preprocessing Comparison")

# Nutzer wählt eine Zeile (Index) aus
row_index = st.number_input("Select a Row Index to inspect:", min_value=0, max_value=len(df)-1, value=0)

selected_row = df.iloc[row_index]

# Darstellung in 3 Spalten
col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Original Text**")
    st.write(selected_row['review_text'])

with col2:
    st.success("**Basic Cleaned**")
    st.write(selected_row['review_text_clean'])

with col3:
    st.warning("**Advanced Cleaned**")
    st.write(selected_row['review_text_clean_advanced'])








    st.write("### 📊 Compression Statistics")

# Kurze Berechnung der Wortanzahl für die ausgewählte Zeile
orig_len = len(str(selected_row['review_text']).split())
clean_len = len(str(selected_row['review_text_clean']).split())
adv_len = len(str(selected_row['review_text_clean_advanced']).split())

m1, m2, m3 = st.columns(3)
m1.metric("Original Words", orig_len)
m2.metric("Basic Cleaned", clean_len, f"{clean_len - orig_len} words")
m3.metric("Advanced Cleaned", adv_len, f"{adv_len - orig_len} words")









st.markdown("### 🛠️ Data Quality Check (Missing Values)")

# 1. Berechnung der fehlenden Werte pro Spalte
missing_data = df.isnull().sum()
total_missing = missing_data.sum()

# 2. Bedingte Anzeige: Grün wenn alles okay ist, Gelb/Rot wenn nicht
if total_missing == 0:
    st.success("✨ **Perfect!** Your dataset has no missing values (0 NaNs).")
else:
    st.warning(f"⚠️ **Attention:** Found {total_missing} missing values in total!")
    
    # Tabelle mit den Spalten, die Probleme machen
    missing_df = missing_data[missing_data > 0].reset_index()
    missing_df.columns = ['Column Name', 'Number of NaNs']
    
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.table(missing_df)
    with col_right:
        st.info("💡 **Tip:** You might want to drop these rows or fill them with 'empty' before training.")

# 3. Kleiner Bonus: Check auf Duplikate
duplicates = df.duplicated().sum()
if duplicates > 0:
    st.info(f"👥 **Note:** There are {duplicates} exact duplicate rows in your data.")




st.markdown("### 🧹 Final Data Cleaning for Modeling")

# 1. Drop the problematic 'clean' columns
cols_to_drop = ['review_text_clean', 'review_text_clean_advanced']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 2. Remove rows where 'review_text' is NaN
initial_shape = df.shape[0]
df = df.dropna(subset=['review_text'])
removed_nans = initial_shape - df.shape[0]

# 3. Identify and display duplicates before removing them
duplicate_rows = df[df.duplicated(keep=False)].sort_values(by='review_text')

if not duplicate_rows.empty:
    st.warning(f"🔍 Found {len(duplicate_rows)} rows that are part of a duplicate set.")
    
    with st.expander("📋 View Duplicate Rows"):
        st.write("These rows have identical content across all columns:")
        st.dataframe(duplicate_rows, use_container_width=True)
    
    # 4. Remove Duplicates
    df = df.drop_duplicates()
    st.success(f"✅ Cleaning complete: Removed {removed_nans} NaNs and {len(duplicate_rows) // 2} duplicate pairs.")
else:
    st.success("✨ No duplicates found after NaN removal.")

# Update Session State with the cleaned data
st.session_state['ml_data'] = df

st.info(f"**Final Dataset Shape:** {df.shape[0]} rows and {df.shape[1]} columns.")


st.markdown("---")

