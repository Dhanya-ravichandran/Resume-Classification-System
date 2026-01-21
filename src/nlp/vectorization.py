from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_vectorizer(
    max_features=3000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
):
    """
    Returns a configured TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        ngram_range=ngram_range,
        stop_words="english"
    )
    return vectorizer


    # max_features=5000,
    # ngram_range=(1, 2)