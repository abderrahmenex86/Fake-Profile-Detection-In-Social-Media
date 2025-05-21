import jieba


def segment_chinese(text):
    return " ".join(jieba.cut(text))
