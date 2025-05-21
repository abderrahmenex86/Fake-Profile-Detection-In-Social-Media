import joblib

from ...preprocessing.weibo.normalize_chinese import normalize_chinese
from ...preprocessing.weibo.remove_stopwords import remove_stopwords
from ...preprocessing.weibo.segment_chinese import segment_chinese
from ...preprocessing.weibo.vectorize_chinese import vectorize_chinese


def preprocess_text(raw, cache_file="preprocessed_weibo_data.joblib", force=False):
    path = os.path.join("data", cache_file)
    if os.path.exists(path) and not force:
        print(f"Loading preprocessed data from {path}")
        cached_data = joblib.load(path)
        return cached_data["X"], cached_data["vectorizer"]

    preprocess_texts = []
    for text in tqdm(raw):
        try:
            text = normalize_chinese(text)
            text = segment_chinese(text)
            text = remove_stopwords(text)
            preprocess_texts.append(text)
        except Exception as e:
            preprocess_texts.append("")
    X, vectorizer = vectorize_chinese(preprocess_texts)
    print(f"Saving preprocessed data to {cache_file}")
    joblib.dump({"X": X, "vectorizer": vectorizer}, path)
    return X, vectorizer
