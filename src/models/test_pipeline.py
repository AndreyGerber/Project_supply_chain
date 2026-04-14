if __name__ == "__main__":
    from src.data.load_data import get_data
    from src.features.build_features import preprocess_dataframe, generate_tfidf, generate_embeddings
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from src.evaluation.evaluate import evaluate_classification
    import torch, numpy as np, random

    # ------------- Global Seed -------------
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1️⃣ Daten laden
    df = get_data(use_processed=True)
    df = df.head(100)  # klein für Test

    # 2️⃣ Ziel
    y = (df["rating"] >= 4).astype(int)

    # 3️⃣ TF-IDF Pipeline Test
    preprocessor, df = generate_tfidf(df)
    pipeline_tfidf = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=10, random_state=SEED))
    ])
    pipeline_tfidf.fit(df, y)
    preds = pipeline_tfidf.predict(df)
    print("\nTF-IDF Test Evaluation:")
    evaluate_classification(y, preds)

    # 4️⃣ Embeddings Pipeline Test
    X_emb = generate_embeddings(df)
    model_emb = RandomForestClassifier(n_estimators=10, random_state=SEED)
    model_emb.fit(X_emb, y)
    preds_emb = model_emb.predict(X_emb)
    print("\nEmbeddings Test Evaluation:")
    evaluate_classification(y, preds_emb)

    print("\n✅ Mini-Testpipeline erfolgreich durchlaufen (Reproduzierbar).")