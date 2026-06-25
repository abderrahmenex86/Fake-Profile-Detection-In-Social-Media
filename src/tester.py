from sklearn.utils import shuffle

from src.dataset import load_image_paths
from src.factory import build_extractor, build_model


def run_test(args):
    print(f"\n[INFO] Running Sanity Check for {args.model.upper()} with {args.feature.upper()} features...")

    paths, labels = load_image_paths("dataset", split="train")

    paths, labels = shuffle(paths, labels, random_state=42)
    paths, labels = paths[:20], labels[:20]  # Aggressive slice

    print("\n[TEST 1] Instantiating Extractor...")
    extractor = build_extractor(args)

    print("\n[TEST 2] Fitting Extractor (Sanity shape check)...")
    X = extractor.fit_transform(paths)
    print(f"  -> Features Shape: {X.shape}")

    print("\n[TEST 3] Instantiating and Fitting Model...")
    model = build_model(args.model)
    model.fit(X, labels)

    print("\n[TEST 4] Predicting...")
    preds = model.predict(X)
    print(f"  -> Predictions Shape: {preds.shape}")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] All sanity checks passed! Ready to train.")
    print("=" * 60 + "\n")
