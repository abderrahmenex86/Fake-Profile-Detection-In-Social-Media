from ...preprocessing.x.prepare import prepare


def load_data():
    X, y = prepare()
    return X, y
