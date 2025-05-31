import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

from fake_profile_detector.configs.general import RANDOM_SEED


def evaluate_features(X, y, feature_name, extractor=None):
    print(f"\nEvaluating {feature_name}...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # Apply PCA only to training data if extractor has PCA enabled
    if extractor is not None and extractor.apply_pca:
        X_train, X_test = extractor.fit_pca_and_transform(X_train, X_test)

    classifiers = {
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, random_state=RANDOM_SEED
        ),
    }

    results = {}

    # Add progress bar for classifier evaluation
    for clf_name, classifier in tqdm(
        classifiers.items(), desc=f"Evaluating classifiers for {feature_name}"
    ):
        print(f"  Training {clf_name}...")
        start_time = time.time()

        try:
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            train_time = time.time() - start_time

            results[clf_name] = {
                "accuracy": accuracy,
                "train_time": train_time,
                "feature_dim": X_train.shape[1],  # Use actual feature dimension after PCA
            }

            print(f"    Accuracy: {accuracy:.4f}, Time: {train_time:.2f}s")

        except Exception as e:
            print(f"    Error with {clf_name}: {e}")
            results[clf_name] = {
                "accuracy": 0.0,
                "train_time": 0.0,
                "feature_dim": X_train.shape[1],
            }

    return results
