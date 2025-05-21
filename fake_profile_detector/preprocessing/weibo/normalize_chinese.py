import opencc


def normalize_chinese(text, convert_to_simplified=True):
    if convert_to_simplified:
        converter = opencc.opencc("t2s")
        text = converter.convert(text)

    punctuation = "！，。、：；？" '""【】（）《》〈〉…—～·'
    for p in punctuation:
        text = text.replace(p, " ")

    return text
