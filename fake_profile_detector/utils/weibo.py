import glob
import json
import os
import re

import jieba
import joblib
import kagglehub
import numpy as np
import opencc
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def vectorize_chinese(texts):
    vectorizer = TfidfVectorizer(max_features=512)
    X = vectorizer.fit_transform(texts)

    return X, vectorizer


def normalize_chinese(text, convert_to_simplified=True):
    if convert_to_simplified:
        converter = opencc.OpenCC("t2s")
        text = converter.convert(text)

    punctuation = "！，。、：；？" '""【】（）《》〈〉…—～·'
    for p in punctuation:
        text = text.replace(p, " ")

    return text


def segment_chinese(text):
    return " ".join(jieba.cut(text))


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


def extract_config_data(html_content):
    config_pattern = re.compile(r"\$CONFIG\s*=\s*\{\s*(.*?)\}\s*;", re.DOTALL)
    match = config_pattern.search(html_content)

    config_data = {}
    if match:
        config_text = match.group(1)

        lines = config_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and ":" in line:
                # Handle the line to extract key and value
                if line.endswith(";"):
                    line = line[:-1]

                key_val = line.split(":", 1)
                if len(key_val) == 2:
                    key, value = key_val

                    # Clean up the key
                    key = (
                        key.strip()
                        .replace("$CONFIG[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .replace('"', "")
                    )

                    # Clean up the value
                    value = value.strip().strip("'").strip(";").strip("'").strip('"')

                    config_data[key] = value

    return config_data


def extract_post_content(soup):
    post_divs = soup.find_all("div", class_="WB_text W_f14")

    all_content = []
    for div in post_divs:
        text = div.get_text(strip=True)
        if text:
            all_content.append(text)

    return " ".join(all_content)


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


def extract_profile_info(soup):
    profile_info = {}

    try:
        counter_div = soup.find("div", class_="PCD_counter")
        if counter_div:
            counts = counter_div.find_all("strong", class_="W_f18")
            if len(counts) >= 3:
                profile_info["following_count"] = counts[0].get_text(strip=True)
                profile_info["followers_count"] = counts[1].get_text(strip=True)
                profile_info["posts_count"] = counts[2].get_text(strip=True)
    except:
        pass

    try:
        description = soup.find("div", class_="pf_intro")
        if description:
            profile_info["profile_description"] = description.get_text(strip=True)
    except:
        pass

    try:
        info_items = soup.find_all("li", class_="item S_line2 clearfix")
        for item in info_items:
            icon = item.find("em", class_="W_ficon")
            if icon and icon.get_text(strip=True):
                icon_type = icon.get_text(strip=True)
                text = item.find("span", class_="item_text W_fl").get_text(strip=True)

                if "地" in icon_type or "place" in str(icon.get("class", [])):
                    profile_info["location"] = text
                elif "学" in icon_type or "edu" in str(icon.get("class", [])):
                    profile_info["education"] = text
                elif "职" in icon_type or "bag" in str(icon.get("class", [])):
                    profile_info["occupation"] = text
                elif "邮" in icon_type or "email" in str(icon.get("class", [])):
                    profile_info["email"] = text
    except:
        pass

    return profile_info


def process_html_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        config_data = extract_config_data(html_content)
        post_content = extract_post_content(soup)
        profile_info = extract_profile_info(soup)

        all_data = {
            "file_name": os.path.basename(file_path),
            "account_content": post_content,
            **config_data,
            **profile_info,
        }

        return all_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {"file_name": os.path.basename(file_path), "error": str(e)}


def process_all_files(directory):
    all_data = []

    html_files = glob.glob(os.path.join(directory, "*.html"))
    for file_path in tqdm(html_files, desc="Processing HTML files"):
        file_data = process_html_file(file_path)
        if file_data:
            all_data.append(file_data)

    return pd.DataFrame(all_data)


def load_data(force=False):
    kaggle_dir = kagglehub.dataset_download(
        "bitandatom/social-network-fake-account-dataset"
    )
    legit_file = os.path.join(kaggle_dir, "legitimate_account.csv")
    fake_file = os.path.join(kaggle_dir, "fake_account.csv")

    if not os.path.exists("fake.csv") or force:
        fake_dir = os.path.join(kaggle_dir, "fake_account_raw", "fake_account_raw")
        fake_df = process_all_files(fake_dir)
        fake_df["is_fake"] = 1
        fake_df.replace("", np.nan, inplace=True)
        fake_df.to_csv("fake.csv", encoding="utf-8", index=False)
    else:
        fake_df = pd.read_csv("fake.csv", encoding="utf-8")

    fake_clean = fake_df[["account_content", "is_fake"]]
    fake_clean = fake_clean.dropna(subset=["account_content"])
    fake_from_disk = pd.read_csv(
        fake_file, encoding="utf-8", sep="\t", names=["userindex", "account_content"]
    )
    fake_from_disk["is_fake"] = 1
    fake_from_disk = fake_from_disk[["account_content", "is_fake"]]
    legit_df = pd.read_csv(
        legit_file,
        encoding="utf-8",
        sep="\t",
        names=[
            "userindex",
            "postdate",
            "retweet_count",
            "comment_count",
            "like_count",
            "account_content",
        ],
    )
    legit_df["is_fake"] = 0
    legit_clean = legit_df[["account_content", "is_fake"]]
    dataframe = pd.concat([fake_clean, fake_from_disk, legit_clean], ignore_index=True)

    y = dataframe["is_fake"]
    X, vectorizer = preprocess_text(dataframe["account_content"])

    return X, y, vectorizer
