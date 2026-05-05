import numpy as np
import pandas as pd


def get_top_tfidf_words(tfidf_vector, tfidf_pipeline, n=15):
    """
    Extract top contributing TF-IDF words for a single document.
    """

    feature_names = tfidf_pipeline.get_feature_names_out()

    vector = tfidf_vector.toarray().flatten()

    indices = np.argsort(vector)[::-1][:n]

    return pd.DataFrame({
        "word": [feature_names[i] for i in indices],
        "score": vector[indices]
    })