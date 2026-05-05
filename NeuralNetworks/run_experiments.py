"""
Stratified 5-fold CV experiments for HW4 (WDBC + Loan).
Also generates learning curves for the best configuration per dataset.

Run from this directory:
  python3 run_experiments.py

Stopping criterion: fixed number of training epochs (m) per assignment.
Hyperparameters (edit TRAINING below if needed): alpha, max_epochs, mini-batch size.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_loan, load_wdbc, standardize, stratified_kfold_indices
from neural_network import NeuralNetwork, compute_metrics_binary

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "results")

TRAINING = {
    "alpha": 1.0,
    "max_epochs": 100,
    "batch_size": 64,
    "k_folds": 5,
    "random_seed": 589,
}

# At least 6 architectures (hidden layer sizes only)
ARCHITECTURES: list[list[int]] = [
    [8],
    [16],
    [8, 8],
    [16, 8],
    [32],
    [4, 4, 4],
]

LAMBDAS = [0.0, 0.01, 0.1]


@dataclass
class CVResult:
    name: str
    hidden: list[int]
    lam: float
    mean_acc: float
    std_acc: float
    mean_f1: float
    std_f1: float


def _layer_sizes(n_in: int, hidden: list[int]) -> list[int]:
    return [n_in] + hidden + [1]


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    hidden: list[int],
    lam: float,
    rng_base: int,
) -> tuple[float, float, float, float]:
    k = TRAINING["k_folds"]
    splits = stratified_kfold_indices(y, k, np.random.default_rng(rng_base))
    accs: list[float] = []
    f1s: list[float] = []
    n_in = X.shape[0]
    layers = _layer_sizes(n_in, hidden)
    for fold, (tr, va) in enumerate(splits):
        rng = np.random.default_rng(rng_base * 1000 + fold * 17 + len(hidden) * 3)
        Xtr, Xva = X[:, tr], X[:, va]
        ytr, yva = y[:, tr], y[:, va]
        Xtr_s, Xva_s = standardize(Xtr, Xva)
        net = NeuralNetwork(layers, rng=rng)
        net.train(
            Xtr_s,
            ytr,
            lam,
            TRAINING["alpha"],
            TRAINING["max_epochs"],
            TRAINING["batch_size"],
        )
        probs = net.predict_proba(Xva_s)
        pred = (probs >= 0.5).astype(np.int32)
        acc, f1 = compute_metrics_binary(yva, pred)
        accs.append(acc)
        f1s.append(f1)
    return (
        float(np.mean(accs)),
        float(np.std(accs)),
        float(np.mean(f1s)),
        float(np.std(f1s)),
    )


def run_all_experiments(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
) -> list[CVResult]:
    results: list[CVResult] = []
    for hidden in ARCHITECTURES:
        for lam in LAMBDAS:
            arch_name = str(hidden)
            mean_acc, std_acc, mean_f1, std_f1 = cross_validate(
                X, y, hidden, lam, TRAINING["random_seed"]
            )
            results.append(
                CVResult(
                    name=name,
                    hidden=hidden,
                    lam=lam,
                    mean_acc=mean_acc,
                    std_acc=std_acc,
                    mean_f1=mean_f1,
                    std_f1=std_f1,
                )
            )
            print(
                f"[{name}] h={hidden} lam={lam:.3g} "
                f"acc={mean_acc:.4f}+-{std_acc:.4f} f1={mean_f1:.4f}+-{std_f1:.4f}"
            )
    return results


def save_results_table(results: list[CVResult], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "hidden_layers",
                "lambda",
                "mean_accuracy",
                "std_accuracy",
                "mean_f1",
                "std_f1",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.name,
                    json.dumps(r.hidden),
                    r.lam,
                    r.mean_acc,
                    r.std_acc,
                    r.mean_f1,
                    r.std_f1,
                ]
            )


def learning_curve(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    best_hidden: list[int],
    best_lam: float,
    test_fraction: float = 0.2,
) -> None:
    """Train on increasing subsets of a training split; plot J on held-out test."""
    rng = np.random.default_rng(TRAINING["random_seed"] + 7)
    m = X.shape[1]
    idx = np.arange(m)
    rng.shuffle(idx)
    n_test = max(1, int(round(test_fraction * m)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    X_test, y_test = X[:, test_idx], y[:, test_idx]
    X_pool, y_pool = X[:, train_idx], y[:, train_idx]

    sizes = sorted(
        set(
            [5, 10, 20, 30, 50, 75, 100, 150, 200, X_pool.shape[1]]
        )
    )
    sizes = [s for s in sizes if s <= X_pool.shape[1]]
    if sizes[-1] != X_pool.shape[1]:
        sizes.append(X_pool.shape[1])
    sizes = sorted(set(sizes))

    Js: list[float] = []
    used: list[int] = []
    layers = _layer_sizes(X.shape[0], best_hidden)
    for n in sizes:
        sub = np.arange(n)
        Xn_raw = X_pool[:, sub]
        yn = y_pool[:, sub]
        Xn, Xte_s = standardize(Xn_raw, X_test)
        net = NeuralNetwork(layers, rng=np.random.default_rng(TRAINING["random_seed"] + n))
        net.train(
            Xn,
            yn,
            best_lam,
            TRAINING["alpha"],
            TRAINING["max_epochs"],
            min(TRAINING["batch_size"], n),
        )
        j_te = net.cost(Xte_s, y_test, best_lam, include_reg=False)
        Js.append(j_te)
        used.append(n)
        print(f"[LC {name}] n_train={n} J_test={j_te:.5f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(used, Js, marker="o")
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("Test cost J (logistic, no regularization term)")
    ax.set_title(
        f"{name} learning curve — best arch {best_hidden}, lambda={best_lam}, "
        f"alpha={TRAINING['alpha']}, epochs={TRAINING['max_epochs']}"
    )
    ax.grid(True, alpha=0.3)
    path = os.path.join(OUT_DIR, f"learning_curve_{name.lower()}.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def pick_best(results: list[CVResult]) -> CVResult:
    return max(results, key=lambda r: (r.mean_f1, r.mean_acc))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "stopping_criterion": "fixed_number_of_epochs",
        "m_epochs": TRAINING["max_epochs"],
        "alpha": TRAINING["alpha"],
        "batch_size": TRAINING["batch_size"],
        "k_folds": TRAINING["k_folds"],
        "architectures": ARCHITECTURES,
        "lambdas": LAMBDAS,
    }
    with open(os.path.join(OUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Config:", json.dumps(meta, indent=2))

    Xw, yw = load_wdbc(os.path.join(BASE, "datasets", "wdbc.csv"))
    Xl, yl = load_loan(os.path.join(BASE, "datasets", "loan.csv"))

    res_w = run_all_experiments("WDBC", Xw, yw)
    save_results_table(res_w, os.path.join(OUT_DIR, "cv_wdbc.csv"))

    res_l = run_all_experiments("Loan", Xl, yl)
    save_results_table(res_l, os.path.join(OUT_DIR, "cv_loan.csv"))

    bw = pick_best(res_w)
    bl = pick_best(res_l)
    print("\nBest WDBC (by mean F1):", bw.hidden, "lambda=", bw.lam, bw.mean_f1)
    print("Best Loan:", bl.hidden, "lambda=", bl.lam, bl.mean_f1)

    learning_curve("WDBC", Xw, yw, bw.hidden, bw.lam)
    learning_curve("Loan", Xl, yl, bl.hidden, bl.lam)


if __name__ == "__main__":
    main()
