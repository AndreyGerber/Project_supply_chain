import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from _utils import FEATURE_SETS, LABELS, MODEL_PARAMS, STATIC_DIR, TARGET_PER_CLASS, build_dataset, build_features, train_model

st.set_page_config(layout="wide")


def group_histogram(values: pd.Series, title: str, color: str):
    fig = px.histogram(values.to_frame("target_group"), x="target_group", title=title,
                       category_orders={"target_group": LABELS}, color_discrete_sequence=[color])
    st.plotly_chart(fig, width="stretch")


def confusion_heatmap(y_true, y_pred, key: str):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig = px.imshow(cm, x=LABELS, y=LABELS, text_auto=True, color_continuous_scale="Viridis",
                    labels=dict(x="Predicted rating", y="True rating", color="Count"))
    fig.update_layout(xaxis=dict(side="top"))
    st.plotly_chart(fig, width="stretch", key=key)


def params_code(params: dict) -> str:
    args = ",\n".join(f"    {k}={v}" for k, v in params.items())
    return f"model = GradientBoostingClassifier(\n{args}\n)"


# =========================
# Header
# =========================
st.title("Modeling")
st.image(str(STATIC_DIR / "Analysis.png"), width="stretch")
st.divider()

data = build_dataset()
stats = data["stats"]

# =========================
# Step 0: Cleaning & grouping
# =========================
st.subheader("Step 0-1. Data Integration & Advanced Cleaning")
st.success(f"✅ Cleanup finished! Exact dups: {stats['exact_dups']}, Normalized dups: {stats['normalized_dups']}, "
           f"NaNs: {stats['nans']}, Replies: {stats['replies']}")
st.write("### Preview: Processed Data (Top 10 Rows)")
st.dataframe(data["df"].drop(columns=["target_group"]).head(10), width="stretch")

st.subheader("Step 0-2: Rating Distribution & Grouping")
group_histogram(data["df"]["target_group"], "Grouped Rating Distribution", "#00CC96")

# =========================
# Step 1: Train-test split
# =========================
st.divider()
st.subheader("⚠️Step 1: Attention: Preventing Data Leakage")
st.write("The data is split **before** balancing and text processing, so the test set stays unseen.")
col1, col2 = st.columns(2)
with col1:
    st.metric("Training Samples (80%)", len(data["y_train_split"]))
    group_histogram(data["y_train_split"], "Training Set", "#00CC96")
with col2:
    st.metric("Test Samples (20%)", len(data["y_test"]))
    group_histogram(data["y_test"], "Test Set", "#636EFA")

# =========================
# Step 2: Balancing
# =========================
st.divider()
st.subheader("⚖️Step 2: Balancing Training Data (word switching augmentation)")
for group, before, after, method in data["balance_log"]:
    direction = "Reduced" if method == "Undersampling" else "Increased"
    st.write(f"**{group}**: {direction} from {before} to {after} ({method}).")

col1, col2 = st.columns(2)
with col1:
    st.metric("Final Training Samples", len(data["y_train"]))
    group_histogram(data["y_train"], "Balanced Training Set", "#FF7F0E")
with col2:
    st.metric("Test Samples (Original)", len(data["y_test"]))
    group_histogram(data["y_test"], "Original Test Set", "#636EFA")
    st.info("Note: The test set remains untouched to ensure honest evaluation.")
st.success(f"💡 **Resampling successful!** Each class now has {TARGET_PER_CLASS} training samples.")

# =========================
# Step 3: Text preprocessing
# =========================
st.divider()
st.subheader("🧪Step 3: Advanced Text Preprocessing")
st.write("Lowercase → Regex → Tokenize → Remove stopwords (negations are kept) → Join")
st.success("All steps completed! Your text is now 'Vectorization-ready'.")
st.write("Comparison (Original vs. Processed):")
st.dataframe(data["X_train_text"][["review_text", "clean_review"]].head(5))

# =========================
# Step 4: Vectorization
# =========================
st.divider()
st.subheader("🔢 Step 4: Vectorization & Feature Combination")
rows, cols = build_features("tfidf")["X_train"].shape
st.success("✅ Vectorization complete!")
st.write(f"**Matrix Dimensions:** {rows} samples (rows, each rating of {TARGET_PER_CLASS}) × "
         f"{cols} features (columns, 5000 TF-IDF words + verified).")

# =========================
# Step 5: Training
# =========================
st.divider()
st.subheader("🤖 Step 5: Base Model Training & Evaluation")
y_test = data["y_test"]


def show_results(name: str, label: str):
    result = train_model(name)
    st.success(f"✅ {label} Training finished in {result['duration']:.2f}s")
    st.metric(f"{label} Accuracy", f"{accuracy_score(y_test, result['y_pred']):.2%}")
    confusion_heatmap(y_test, result["y_pred"], key=f"cm_{name}")
    return result


if st.button("🚀 Run Base Model Training"):
    with st.spinner("It will take a moment... you can grab a coffee ☕"):
        train_model("base")
    st.session_state["base_done"] = True

if st.session_state.get("base_done"):
    show_results("base", "Base")

    st.write("#### 💡 Something to optimize?")
    col1, col2 = st.columns(2)
    col1.caption("Standard Model (Base)")
    col1.code(params_code(MODEL_PARAMS["base"]), language="python")
    col2.caption("Optimized Model (Tuned)")
    col2.code(params_code(MODEL_PARAMS["tuned"]), language="python")

    st.divider()
    st.subheader("🤖 Step 5 (Second Try): Optimized Model")
    if st.button("🚀 Run Optimized Training"):
        with st.spinner("It will take a moment... you can grab one more coffee ☕"):
            train_model("tuned")
        st.session_state["tuned_done"] = True
        st.balloons()

    if st.session_state.get("tuned_done"):
        result = show_results("tuned", "Optimized")
        st.write("### Classification Report (Optimized)")
        st.dataframe(pd.DataFrame(classification_report(y_test, result["y_pred"], output_dict=True)).transpose())

        # --- Third try: better features instead of better hyperparameters ---
        st.divider()
        st.subheader("🤖 Step 5 (Third Try): Enhanced Features")
        enhanced = FEATURE_SETS["enhanced"]
        st.markdown(f"""
The second try changed the **model**. The third try changes the **features** and keeps the optimized hyperparameters,
so the difference in accuracy comes from the features alone:

- **Bi-grams:** word pairs like *"no problem"* or *"never arrived"* become features of their own, so a negation
  stays connected to the word it refers to (vocabulary: {enhanced['max_features']} uni- and bi-grams).
- **Sentiment score** (VADER, -1 to +1) and **negation flag** (not / no / never) from the Feature Engineering page,
  where both showed the strongest relationship with the rating.
""")
        st.code("""TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
features = TF-IDF + verified + sentiment + has_negation""", language="python")

        if st.button("🚀 Run Enhanced Training"):
            with st.spinner("It will take a moment... last coffee, promised ☕"):
                train_model("enhanced")
            st.session_state["enhanced_done"] = True
            st.balloons()

        if st.session_state.get("enhanced_done"):
            result = show_results("enhanced", "Enhanced")
            st.write("### Classification Report (Enhanced)")
            st.dataframe(pd.DataFrame(classification_report(y_test, result["y_pred"], output_dict=True)).transpose())

            st.write("### 🏁 Comparison of all three tries")
            rows = []
            for name, label in [("base", "1. Base"), ("tuned", "2. Optimized hyperparameters"),
                                ("enhanced", "3. Enhanced features")]:
                res = train_model(name)
                report = classification_report(y_test, res["y_pred"], output_dict=True)
                rows.append({"Model": label, "Accuracy": accuracy_score(y_test, res["y_pred"]),
                             "F1 Mid (3-4 ⭐)": report[LABELS[1]]["f1-score"],
                             "Macro F1": report["macro avg"]["f1-score"],
                             "Training time (s)": res["duration"]})
            st.dataframe(pd.DataFrame(rows).style.format({"Accuracy": "{:.2%}", "F1 Mid (3-4 ⭐)": "{:.2f}",
                                                          "Macro F1": "{:.2f}", "Training time (s)": "{:.1f}"}),
                         hide_index=True, width="stretch")
            st.page_link("pages/05_Live_Demo.py", label="Try the enhanced model in the Live Demo", icon="🚀")

# =========================
# Outlook
# =========================
st.divider()
st.subheader("🚀 Future Improvements & Outlook")
st.write("To further increase the model's performance (especially for the 'Mid' class), "
         "the following strategies could be implemented in the next iteration:")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 1. Advanced Balancing")
    st.write("Instead of physical resampling (shrinking/growing data), we could use the entire dataset "
             "and handle the imbalance mathematically via **Class Weights**.")
    st.code("# weights = compute_sample_weight(class_weight='balanced', y=y_train)\n"
            "# model.fit(X_train, y_train, sample_weight=weights)", language="python")
with col2:
    st.markdown("### 2. Hyperparameter Tuning")
    st.write("Using **GridSearchCV** or **RandomizedSearchCV** to systematically iterate "
             "through different combinations of:")
    st.info("n_estimators, max_depth, learning_rate & subsample")

st.divider()
st.markdown("### 3. Cross-Validation (K-Fold)")
st.write("Instead of a single Train-Test-Split, we could use **Cross-Validation**. "
         "This means splitting the training data into e.g., 5 smaller areas (folds). "
         "The model is trained 5 times, each time using a different area as the 'mini-test-set'.")
st.success("🎯 **Benefit:** This ensures the results are stable and not just a 'lucky punch' from one specific split.")

st.markdown("### 4. Linguistic & Semantic Refinement")
st.write("Bi-grams are already part of the third try. A next step would be **Lemmatization**.")
st.info("*   **Lemmatization:** Reducing words to their base form (e.g., 'running' → 'run') to unify the vocabulary.")

st.markdown("### 5. Transition to Deep Learning Frameworks")
st.write("Implementing the model using specialized Deep Learning libraries like **Keras (TensorFlow)** or **PyTorch**.")
st.info("""
*   **Keras/TensorFlow:** Ideal for rapid prototyping of Dense Neural Networks to capture non-linear relationships between words.
*   **PyTorch:** Offers great flexibility for implementing advanced architectures like LSTMs or Transformers (the tech behind ChatGPT), which understand the **context** and **order** of words far better than traditional models.
""")

st.markdown("### 6. Alternative Classical Models")
st.write("Gradient Boosting is not the typical first choice for sparse text features. "
         "These models are worth comparing on the same features:")
st.info("""
*   **Logistic Regression & Linear SVM (LinearSVC):** The standard baselines for TF-IDF. They handle thousands of sparse features very well, train in seconds instead of minutes and are easy to interpret.
*   **Multinomial / Complement Naive Bayes:** Very fast and robust on small text datasets; Complement NB is designed for imbalanced classes like our 'Mid' group.
*   **LightGBM / XGBoost:** Faster and usually stronger implementations of gradient boosting, with built-in class weights.
*   **Fine-tuned Transformer (e.g. DistilBERT):** Reads the whole sentence in context, so negation and sarcasm are captured much better – at the cost of a GPU and longer training.
""")
