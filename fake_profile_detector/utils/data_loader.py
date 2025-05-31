from sklearn.model_selection import train_test_split

from ..configs.general import RANDOM_SEED, TEST_SIZE
from ..preprocessing.instagram import main as instagram
from ..preprocessing.weibo import main as weibo
from ..preprocessing.x import main as x


def load_data(name):
    assert name in ["instagram", "weibo", "x"], "Dataset not found"
    print(name)
    if name == "instagram":
        X, y = instagram.load_data()
    elif name == "weibo":
        X, y, vectorizer = weibo.load_data()
    else:
        print("x")
        X, y = x.load_data()

    X_temp, X_final_test, y_temp, y_final_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    X_final_train, X_opt, y_final_train, y_opt = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=RANDOM_SEED, stratify=y_temp
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
