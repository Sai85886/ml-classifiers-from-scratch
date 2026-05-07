import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, load_digits

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
            criterion=params.get("criterion", "info_gain"),
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
            criterion=params.get("criterion", "info_gain"),
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


def cv_score(X, y, model_name, params, normalize=False, multiclass=False, positive_label=1, k=10):
    """Run stratified k-fold CV for one hyperparameter setting."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    folds = make_folds(y, k=k, seed=589)

    fold_acc = []
    fold_f1 = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])

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

    return {
        "mean_acc": float(np.mean(fold_acc)),
        "std_acc": float(np.std(fold_acc)),
        "mean_f1": float(np.mean(fold_f1)),
        "std_f1": float(np.std(fold_f1)),
    }


def plot_metric_vs_param(df, x_col, title, out_dir, prefix, x_label=None, log_x=False):
    """Save Accuracy and F1 (with std error bars) as two separate PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    x_values = df[x_col].tolist()
    x_label = x_label or x_col

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x_values, df["mean_acc"], yerr=df["std_acc"], marker="o", capsize=3)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} - Accuracy vs {x_label}")
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_accuracy.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x_values, df["mean_f1"], yerr=df["std_f1"], marker="o", capsize=3, color="orange")
    ax.set_xlabel(x_label)
    ax.set_ylabel("F1 score")
    ax.set_title(f"{title} - F1 vs {x_label}")
    ax.grid(True, alpha=0.3)
    if log_x:
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_f1.png"), dpi=150)
    plt.close(fig)


def plot_metric_vs_param_grouped(df, x_col, group_col, title, out_dir, prefix, x_label=None):
    """Same as plot_metric_vs_param but with one curve per group (e.g. criterion).
    Saves Accuracy and F1 as two separate PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    x_label = x_label or x_col

    fig, ax = plt.subplots(figsize=(7, 4))
    for group_name, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(x_col)
        ax.errorbar(
            group_df[x_col], group_df["mean_acc"], yerr=group_df["std_acc"],
            marker="o", capsize=3, label=str(group_name),
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title} - Accuracy vs {x_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(title=group_col)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_accuracy.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for group_name, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(x_col)
        ax.errorbar(
            group_df[x_col], group_df["mean_f1"], yerr=group_df["std_f1"],
            marker="o", capsize=3, label=str(group_name),
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("F1 score")
    ax.set_title(f"{title} - F1 vs {x_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(title=group_col)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_f1.png"), dpi=150)
    plt.close(fig)


def cv_fold_accuracies(X, y, model_name, params, n_runs=10, k=10,
                       normalize=False, base_seed=589):
    """Repeat stratified k-fold CV `n_runs` times with different fold seeds
    and return (train_accs, test_accs) - one entry per fold per run.
    With the defaults this gives 10 * 10 = 100 fold accuracies, which is
    enough to draw the HW1-style accuracy histogram while still using CV."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    train_accs = []
    test_accs = []
    for run_idx in range(n_runs):
        folds = make_folds(y, k=k, seed=base_seed + run_idx)
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])

            X_tr = X.iloc[train_idx].reset_index(drop=True)
            y_tr = y.iloc[train_idx].reset_index(drop=True)
            X_te = X.iloc[test_idx].reset_index(drop=True)
            y_te = y.iloc[test_idx].reset_index(drop=True)

            if normalize:
                X_tr, X_te = zscore_by_train(X_tr, X_te)

            # Predicting back on the training fold is what the HW1 question
            # used to expose overfitting (train accuracy >> test accuracy).
            y_pred_te = fit_predict(model_name, params, X_tr, y_tr, X_te)
            y_pred_tr = fit_predict(model_name, params, X_tr, y_tr, X_tr)

            train_accs.append(accuracy(y_tr.to_numpy(), y_pred_tr))
            test_accs.append(accuracy(y_te.to_numpy(), y_pred_te))

    return train_accs, test_accs


def plot_accuracy_histogram(accuracies, title, output_path, bins=20):
    """Histogram of test accuracies (HW3 decision-tree style)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(accuracies, bins=bins)
    ax.set_xlabel("Test accuracy")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def nn_learning_curve(X, y, params, multiclass=False, normalize=True,
                      sizes=None, seed=589):
    """Train the NN with the given config on increasing training sizes,
    record the cost J on a held-out test split. Mirrors the HW4 plot."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    folds = make_folds(y, k=5, seed=seed)
    test_idx = folds[0]
    pool_idx = np.hstack([folds[j] for j in range(1, 5)])

    rng = np.random.default_rng(seed)
    rng.shuffle(pool_idx)

    X_pool = X.iloc[pool_idx].reset_index(drop=True)
    y_pool = y.iloc[pool_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    if normalize:
        X_pool, X_test = zscore_by_train(X_pool, X_test)

    n = len(y_pool)
    if sizes is None:
        sizes = sorted({int(n * f) for f in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0] if int(n * f) >= 5})

    classes = sorted(pd.unique(y_pool).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    x_test_np = X_test.to_numpy(dtype=np.float64).T
    if multiclass:
        y_test_oh = np.zeros((len(classes), len(y_test)), dtype=np.float64)
        for col, val in enumerate(y_test.tolist()):
            y_test_oh[class_to_idx[val], col] = 1.0
    else:
        y_test_oh = y_test.to_numpy(dtype=np.float64).reshape(1, -1)

    rows = []
    for n_train in sizes:
        X_train = X_pool.iloc[:n_train].reset_index(drop=True)
        y_train = y_pool.iloc[:n_train].reset_index(drop=True)

        x = X_train.to_numpy(dtype=np.float64).T
        if multiclass:
            y_oh = np.zeros((len(classes), len(y_train)), dtype=np.float64)
            for col, val in enumerate(y_train.tolist()):
                y_oh[class_to_idx[val], col] = 1.0
            layers = [x.shape[0]] + params["layer_sizes"] + [len(classes)]
        else:
            y_oh = y_train.to_numpy(dtype=np.float64).reshape(1, -1)
            layers = [x.shape[0]] + params["layer_sizes"] + [1]

        nn = NeuralNetwork(layers, rng=np.random.default_rng(seed))
        nn.train(
            x=x, y=y_oh,
            lam=params.get("lam", 0.01),
            alpha=params.get("alpha", 0.5),
            max_epochs=params.get("max_epochs", 100),
            batch_size=min(params.get("batch_size", 64), x.shape[1]),
        )

        # J on held-out test set; we drop the regularization term so the
        # curve reflects how well the model fits the unseen data.
        j_test = nn.cost(x_test_np, y_test_oh, lam=0.0, include_reg=False)
        rows.append({"n_train": n_train, "J_test": j_test})

    return pd.DataFrame(rows)


def plot_learning_curve(df, title, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["n_train"], df["J_test"], marker="o")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Test cost J")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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


def load_zoo_data():
    """Load the Zoo dataset (mixed attributes, 7 classes)."""
    data = fetch_openml(name="zoo", version=1, as_frame=True)
    df = data.frame.copy()

    y_text = df.iloc[:, -1].astype(str)
    X = df.iloc[:, :-1].copy()

    # The first column is typically an animal name; remove non-informative ID-like text.
    if X.columns[0].lower() in {"animal_name", "name"}:
        X = X.iloc[:, 1:]

    # One-hot any categorical columns; keep numeric columns as-is.
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)

    labels = sorted(y_text.unique())
    mapping = {label: i for i, label in enumerate(labels)}
    y = y_text.map(mapping).astype(int)
    return X, y
