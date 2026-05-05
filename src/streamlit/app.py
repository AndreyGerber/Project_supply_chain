import streamlit as st

# 1. Konfiguration der Seite
st.set_page_config(
    page_title="Introduction to Supply Chain Analytics",
    page_icon="📊",
    layout="wide"
)

# 2. Titel und Einleitung
st.title("🤖 Project Overview")
st.subheader("Introduction to the project: Supply Chain - Customer Satisfaction")

st.markdown("""

- Setup: The supply chain represents the stages of supply, the production process and the distribution of goods.

- Product Flow: Supplier → Factories → Warehouses → Outlets → Consumers

- Goal: To predict ratings from customer comments
            
- We chose German companies in the “Auto Parts Store” category.
            """)
st.subheader("Business context")
st.markdown("""
This project aims to help a company anticipate dissatisfaction and improve operational efficiency.
""")

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
### 🚀 Workflow & Navigation
st.markdown("""
Use the **sidebar on the left** to navigate through the different phases of the project:
""", unsafe_allow_html=True)

# 3. Project Phases visualized as Columns
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.info("**Phase 1**")
    st.write("**Scraping**")
    st.caption("Scraping Trustpilot websites and do basic cleaning of the scraped data")

with col2:
    st.info("**Phase 2**")
    st.write("**Data Exploration**")
    st.caption("Explore the data, give basic diagrams")

with col3:
    st.info("**Phase 3**")
    st.write("**Preprocessing**")
    st.caption("Prepare the data for feature engineering and model.")


with col4:
    st.info("**Phase4**")
    st.write("**Feature Engineering**")
    st.caption("Adding structured features to improve performance")

with col5:
    st.info("**Phase5**")
    st.write("**Modelling**")
    st.caption("Comparison of various ML algorithms regarding accuracy, performance, and training time.")

with col6:
    st.info("**Phase 6**")
    st.write("**Live Demo**")
    st.caption("Interactive Prediction: Enter your own comment and let the AI predict the rating!")

st.markdown("---")

st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

font_size = "20px"
st.markdown(
    f"""
    <div style="
        background-color: #d4edda; 
        color: #155724; 
        padding: 15px; 
        border-radius: 5px; 
        border: 1px solid #c3e6cb;
        font-size: {font_size};
        display: flex;
        align-items: center;
    ">
        💡 <span style="margin-left: 10px;">
            <b>Ready to Start:</b> then let's move to Data Exploration.
        </span>
    </div>
    """, 
    unsafe_allow_html=True
)