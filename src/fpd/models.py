try:
    from cuml.ensemble import RandomForestClassifier as CuRandomForest
    from cuml.linear_model import LogisticRegression as CuLogisticRegression
    from cuml.svm import SVC as CuSVC

    CUML_AVAILABLE = True
except Exception:
    CUML_AVAILABLE = False


class MLPModel:
    def __init__(self, input_dim, hidden, dropout, lr, weight_decay, seed, epochs, batch_size, device):
        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def _build(self):
        from torch import nn

        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden // 2, 4),
        )

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.seed)
        self.model = self._build().to(self.device)
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            self.model.train()
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(features), targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self

    def predict(self, X):
        import torch

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float().to(self.device))
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).float().to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()


def build_model(name, params, seed, backend, input_dim=None, mlp_epochs=30, mlp_batch_size=128):
    use_gpu = backend == "gpu"
    if name == "logreg":
        if use_gpu and CUML_AVAILABLE:
            return CuLogisticRegression(C=params["C"], max_iter=2000)
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=params["C"], max_iter=2000, random_state=seed)
    if name == "linear_svm":
        if use_gpu and CUML_AVAILABLE:
            return CuSVC(kernel="linear", C=params["C"])
        from sklearn.svm import SVC

        return SVC(kernel="linear", C=params["C"], cache_size=2000, random_state=seed)
    if name == "rbf_svm":
        if use_gpu and CUML_AVAILABLE:
            return CuSVC(kernel="rbf", C=params["C"], gamma=params["gamma"])
        from sklearn.svm import SVC

        return SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], cache_size=2000, random_state=seed)
    if name == "rf":
        if use_gpu and CUML_AVAILABLE:
            return CuRandomForest(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                max_features=params["max_features"],
                random_state=seed,
            )
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            n_jobs=-1,
            random_state=seed,
            eval_metric="mlogloss",
            verbosity=0,
        )
    if name == "mlp":
        import torch

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        return MLPModel(
            input_dim=input_dim,
            hidden=params["hidden"],
            dropout=params["dropout"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            seed=seed,
            epochs=mlp_epochs,
            batch_size=mlp_batch_size,
            device=device,
        )
    raise ValueError(name)
