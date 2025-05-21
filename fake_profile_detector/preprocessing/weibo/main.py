import os

import kagglehub
import numpy as np
import pandas as pd

from ...preprocessing.weibo.preprocess_text import preprocess_text
from ...preprocessing.weibo.process_all_files import process_all_files


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
