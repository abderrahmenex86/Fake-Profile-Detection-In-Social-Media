import os

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments.config import RANDOM_SEED, TEST_SIZE, VAL_SIZE
from utils import instagram, weibo, x_fake_profile_detection


def load_data(name):
    assert name in ["instagram", "weibo", "x"], "Dataset not found"
    if name == "instagram":
        X, y = instagram.load_data()
    elif name == "weibo":
        X, y, vectorizer = weibo.load_data()
    else:
        X, y = x_fake_profile_detection.load_data()

    X_temp, X_final_test, y_temp, y_final_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    X_final_train, X_opt, y_final_train, y_opt = train_test_split(
        X_temp, y_temp, test_size=0.05, random_state=RANDOM_SEED, stratify=y_temp
    )

    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_opt, y_opt, test_size=0.25, stratify=y_opt
    )

    return (
        X_final_train,
        y_final_train,
        X_opt_train,
        y_opt_train,
        X_opt_val,
        y_opt_val,
        X_final_test,
        y_final_test,
    )
