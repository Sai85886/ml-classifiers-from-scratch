import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

ROOT = os.path.dirname(os.path.abspath(__file__))
for folder in ["KNN", "RandomForests", "NeuralNetworks"]:
    full_path = os.path.join(ROOT, folder)
    if full_path not in sys.path:
        sys.path.append(full_path)

from knn import kNN  # type: ignore
from decision_tree import DecisionTreeClassifier  # type: ignore
from random_forest import RandomForestClassifier  # type: ignore
from neural_network import NeuralNetwork  # type: ignore


def make_folds(y, k=10, seed=589):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for i, row_idx in enumerate(idx):
            folds[i % k].append(int(row_idx))
    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def zscore_by_train(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).replace(0, 1.0)
    return (X_train - mean) / std, (X_test - mean) / std


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def f1_binary(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def f1_macro(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    scores = []
    for label in labels:
        yt = (y_true == label).astype(int)
        yp = (y_pred == label).astype(int)
        scores.append(f1_binary(yt, yp, positive_label=1))
    return float(np.mean(scores)) if scores else 0.0


def fit_predict(model_name, params, X_train, y_train, X_test):
    if model_name == "knn":
        model = kNN()
        model.features_train = X_train.reset_index(drop=True)
        model.labels_train = y_train.reset_index(drop=True)
        preds = []
        for i in range(len(X_test)):
            preds.append(model.predict_using_knn(X_test.iloc[i], params["k"]))
        return np.asarray(preds)

    if model_name == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=params.get("max_depth"),
            min_size_for_split=params.get("min_size_for_split", 2),
            min_gain=params.get("min_gain", 0.0),
            random_state=589,
        )
        model.fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True))
        return model.predict(X_test.reset_index(drop=True))

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_trees=params["n_trees"],
            m_try=params.get("m_try"),
            max_depth=params.get("max_depth"),
            min_size_for_split=params.get("min_size_for_split", 2),
            min_gain=params.get("min_gain", 0.0),
            random_state=589,
        )
        model.fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True))
        return model.predict(X_test.reset_index(drop=True))

    if model_name == "neural_network":
        classes = sorted(y_train.unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(classes)}

        x = X_train.to_numpy(dtype=np.float64).T
        if len(classes) == 2:
            y = y_train.to_numpy(dtype=np.float64).reshape(1, -1)
            layers = [x.shape[0]] + params["layer_sizes"] + [1]
        else:
            y = np.zeros((len(classes), len(y_train)), dtype=np.float64)
            for col, val in enumerate(y_train.tolist()):
                y[class_to_idx[val], col] = 1.0
            layers = [x.shape[0]] + params["layer_sizes"] + [len(classes)]

        nn = NeuralNetwork(layers, rng=np.random.default_rng(589))
        nn.train(
            x=x,
            y=y,
            lam=params.get("lam", 0.01),
            alpha=params.get("alpha", 0.5),
            max_epochs=params.get("max_epochs", 100),
            batch_size=min(params.get("batch_size", 64), x.shape[1]),
        )

        x_test = X_test.to_numpy(dtype=np.float64).T
        probs = nn.predict_proba(x_test)
        if len(classes) == 2:
            return (probs.ravel() >= params.get("threshold", 0.5)).astype(int)
        best = np.argmax(probs, axis=0)
        return np.asarray([classes[i] for i in best])

    raise ValueError(f"Unknown model name: {model_name}")


def run_cv(X, y, model_name, param_grid, normalize=False, multiclass=False, positive_label=1):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    folds = make_folds(y, k=10, seed=589)
    rows = []

    for params in param_grid:
        fold_acc = []
        fold_f1 = []
        for i in range(10):
            test_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(10) if j != i])

            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_test = y.iloc[test_idx].reset_index(drop=True)

            if normalize:
                X_train, X_test = zscore_by_train(X_train, X_test)

            y_pred = fit_predict(model_name, params, X_train, y_train, X_test)
            fold_acc.append(accuracy(y_test.to_numpy(), y_pred))
            if multiclass:
                fold_f1.append(f1_macro(y_test.to_numpy(), y_pred))
            else:
                fold_f1.append(f1_binary(y_test.to_numpy(), y_pred, positive_label=positive_label))

        rows.append(
            {
                "model": model_name,
                "params": str(params),
                "mean_accuracy": float(np.mean(fold_acc)),
                "std_accuracy": float(np.std(fold_acc)),
                "mean_f1": float(np.mean(fold_f1)),
                "std_f1": float(np.std(fold_f1)),
            }
        )

    return pd.DataFrame(rows)


def save_basic_plot(df, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    x = list(range(1, len(df) + 1))
    plt.figure(figsize=(8, 4))
    plt.plot(x, df["mean_f1"], marker="o")
    plt.xlabel("config id")
    plt.ylabel("mean f1")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_digits_data():
    data = load_digits()
    X = pd.DataFrame(data.data, columns=[f"pixel_{i}" for i in range(data.data.shape[1])])
    y = pd.Series(data.target.astype(int))
    return X, y


def load_parkinsons_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].copy()
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(0).astype(int)
    return X, y


def load_rice_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].copy()
    y_text = df.iloc[:, -1].astype(str)
    labels = sorted(y_text.unique())
    mapping = {label: i for i, label in enumerate(labels)}
    y = y_text.map(mapping).astype(int)
    return X, y


def load_credit_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].copy()
    cat_cols = [c for c in X.columns if c.endswith("_cat")]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)
    y = pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(0).astype(int)
    return X, y
