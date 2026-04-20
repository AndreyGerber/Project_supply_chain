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
