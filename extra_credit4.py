"""
Extra Credit Question #4: non-trivial variant of a studied algorithm.

Implements distance-weighted k-NN (inverse-distance^p voting) and evaluates it on
all four required datasets with stratified 10-fold CV (same protocol as main project).

Run from repo root:
    python3 extra_credit4.py
"""
import os

import pandas as pd

from util import (
    cv_score,
    load_credit_data,
    load_digits_data,
    load_parkinsons_data,
    load_rice_data,
    plot_metric_vs_param,
)


def _run_one_dataset(name, X, y, multiclass, out_root):
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    # At least 6 hyperparameter settings: sweep k (and fixed power=2).
    for k in [1, 3, 5, 7, 9, 11, 13, 15]:
        result = cv_score(
            X,
            y,
            "weighted_knn",
            {"k": k, "power": 2},
            normalize=True,
            multiclass=multiclass,
        )
        result.update({"dataset": name, "model": "weighted_knn", "k": k, "power": 2})
        rows.append(result)
    df = pd.DataFrame(rows)
    title = f"{name.replace('_', ' ').title()} - Weighted k-NN (EC4)"
    plot_metric_vs_param(df, "k", title, out_dir, "weighted_knn", x_label="k (neighbors)")
    df.to_csv(os.path.join(out_dir, f"{name}_weighted_knn_cv_results.csv"), index=False)
    return df


def run():
    root = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(root, "outputs", "extra_credit4")
    os.makedirs(out_root, exist_ok=True)

    parts = []

    X, y = load_digits_data()
    parts.append(_run_one_dataset("digits", X, y, multiclass=True, out_root=out_root))

    X, y = load_parkinsons_data(os.path.join(root, "parkinsons.csv"))
    parts.append(_run_one_dataset("parkinsons", X, y, multiclass=False, out_root=out_root))

    X, y = load_rice_data(os.path.join(root, "rice.csv"))
    parts.append(_run_one_dataset("rice", X, y, multiclass=False, out_root=out_root))

    X, y = load_credit_data(os.path.join(root, "credit_approval.csv"))
    parts.append(_run_one_dataset("credit_approval", X, y, multiclass=False, out_root=out_root))

    all_df = pd.concat(parts, ignore_index=True)
    all_df.to_csv(os.path.join(out_root, "extra_credit4_cv_results.csv"), index=False)
    print(f"Saved Extra Credit #4 results under {out_root}")


if __name__ == "__main__":
    run()
