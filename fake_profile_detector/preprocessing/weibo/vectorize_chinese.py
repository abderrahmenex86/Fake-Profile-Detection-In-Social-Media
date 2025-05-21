from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_chinese(texts):
    vectorizer = TfidfVectorizer(max_features=512)
    X = vectorizer.fit_transform(texts)

    return X, vectorizer
