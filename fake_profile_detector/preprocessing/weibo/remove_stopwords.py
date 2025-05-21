import os

import jieba


def remove_stopwords(text, stopwords_file="chinese_stopwords.txt"):
    file_path = os.path.join("data", stopwords_file)
    if not os.path.exists(file_path):
        url = "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt"
        try:
            os.system(f"wget {url} -O {file_path}")
        except Exception as e:
            print(f"Error downloading stopwords file: {e}")
            return text

    with open(file_path, "r", encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f])

    words = jieba.cut(text)
    return " ".join([word for word in words if word not in stopwords])
