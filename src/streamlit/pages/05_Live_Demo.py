import pandas as pd
import plotly.express as px
import streamlit as st

from _utils import LABEL_COLORS, train_model, vectorize_review

st.set_page_config(layout="wide")
st.title("🚀 Live Model Demo")
st.info("💡 **Note:** This model was trained on **English** text. Please enter your reviews in English.")
st.write("Type a review below to see how the model classifies it!")

MODEL = "enhanced"  # third try: bi-grams + verified + sentiment + negation

# The model is trained once and then shared with the Modelling page (and all visitors)
with st.spinner("Loading the model… on the very first start it is trained once, this can take a few minutes ☕"):
    model = train_model(MODEL)["model"]

with st.form("prediction_form"):
    user_text = st.text_area("Enter Review Text:", placeholder="e.g., This was a great experience, I love it!")
    is_verified = st.checkbox("Is the user verified?", value=True)
    submit = st.form_submit_button("Classify Review")

if submit:
    if not user_text.strip():
        st.warning("Please enter some text first!")
        st.stop()

    cleaned_text, features = vectorize_review(user_text, is_verified, MODEL)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    st.divider()
    color = "green" if "High" in prediction else "violet" if "Mid" in prediction else "red"
    st.subheader(f"Result: :{color}[{prediction}]")

    prob_df = pd.DataFrame({"Rating Group": model.classes_, "Probability": probabilities})
    fig = px.bar(prob_df, x="Rating Group", y="Probability", title="Model Confidence", color="Rating Group",
                 color_discrete_map=LABEL_COLORS, text_auto=".2%")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")

    st.info(f"**Processed text used for prediction:** *{cleaned_text or '[Text was empty after cleaning]'}*")
