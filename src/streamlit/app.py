import streamlit as st

st.set_page_config(page_title="Introduction to Supply Chain Analytics", page_icon="📊", layout="wide")

st.title("🤖 Project Overview")
st.subheader("Introduction to the project: Supply Chain - Customer Satisfaction")
st.markdown("""
- Setup: The supply chain represents the stages of supply, the production process and the distribution of goods.
- Product Flow: Supplier → Factories → Warehouses → Outlets → Consumers
- Goal: To predict ratings from customer comments
- We chose German companies in the “Auto Parts Store” category.
""")

st.subheader("Business context")
st.markdown("This project aims to help a company anticipate dissatisfaction and improve operational efficiency.")

st.subheader("Technical context")
st.markdown("""
- Scraping Trustpilot websites to gain data
- Basic cleaning and basic analysis
- Preprocessing
- Feature Engineering
- Modelling and Optimization
- Live demo
""")

st.subheader("Economic context")
st.markdown("""
Improving customer satisfaction leads to improved business value, including:

- Increased customer retention
- Reduced operational costs (returns, complaints)
- Better brand reputation
""")

st.subheader("Scientific context")
st.markdown("""
This project falls under:

- Supervised Machine Learning (Classification)
- Imbalanced classification problem
- Use of data preprocessing and engineering techniques
""")

st.markdown("Use the **sidebar on the left** to navigate through the different phases of the project:")

PHASES = [
    ("Scraping", "Scraping Trustpilot websites and do basic cleaning of the scraped data"),
    ("Data Exploration", "Explore the data, give basic diagrams"),
    ("Preprocessing", "Prepare the data for feature engineering and model."),
    ("Feature Engineering", "Adding structured features to improve performance"),
    ("Modelling", "Comparison of various ML algorithms regarding accuracy, performance, and training time."),
    ("Live Demo", "Interactive Prediction: Enter your own comment and let the AI predict the rating!"),
]
for i, (col, (name, text)) in enumerate(zip(st.columns(len(PHASES)), PHASES), start=1):
    with col:
        st.info(f"**Phase {i}**")
        st.write(f"**{name}**")
        st.caption(text)

st.divider()
st.success("💡 **Ready to Start:** then let's move to Data Exploration.")
st.page_link("pages/01_Data_Exploration.py", label="Go to Data Exploration", icon="➡️")
