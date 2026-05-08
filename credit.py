# Credit approval: one-hot categoricals + numeric. DT, RF, k-NN + DT accuracy histograms.
import os

import numpy as np
import pandas as pd

from util import (
    cv_fold_accuracies,
    cv_score,
    load_credit_data,
    plot_accuracy_histogram,
    plot_metric_vs_param,
    plot_metric_vs_param_grouped,
    print_cv_result,
)

DATASET_TAG = "credit"


def run():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "outputs", "credit_approval")
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_credit_data(os.path.join(root, "credit_approval.csv"))
    multiclass = False
    all_rows = []

    print(
        f"[{DATASET_TAG}] Loaded {len(X)} rows, {X.shape[1]} features after one-hot (binary approval).",
        flush=True,
    )

    print(f"\n[{DATASET_TAG}] === Decision Tree: depth × criterion ===", flush=True)
    dt_rows = []
    for criterion in ["info_gain", "gini"]:
        for max_depth in [3, 5, 7, 10, 15, None]:
            params = {"max_depth": max_depth, "criterion": criterion, "min_size_for_split": 2}
            result = cv_score(X, y, "decision_tree", params, multiclass=multiclass)
            depth_label = -1 if max_depth is None else max_depth
            result.update({
                "model": "decision_tree",
                "max_depth": depth_label,
                "criterion": criterion,
            })
            dt_rows.append(result)
            all_rows.append(result)
            print_cv_result(
                DATASET_TAG,
                "DecisionTree",
                f"max_depth={depth_label}, criterion={criterion}",
                result,
                multiclass=multiclass,
            )
    dt_df = pd.DataFrame(dt_rows)
    plot_metric_vs_param_grouped(
        dt_df, "max_depth", "criterion", "Credit - Decision Tree",
        out_dir, "dt", x_label="max_depth (-1 = unlimited)",
    )
    print(f"[{DATASET_TAG}] Wrote DT plots → {out_dir}/dt_*.png", flush=True)

    # Many repeated CV splits → distribution of train vs test fold accuracy (overfitting check).
    print(f"\n[{DATASET_TAG}] === Best decision tree: repeated CV for accuracy histograms ===", flush=True)
    best_dt_row = dt_df.sort_values(["mean_f1", "mean_acc"], ascending=False).iloc[0]
    best_dt_params = {
        "max_depth": None if int(best_dt_row["max_depth"]) == -1 else int(best_dt_row["max_depth"]),
        "criterion": best_dt_row["criterion"],
        "min_size_for_split": 2,
    }
    train_accs, test_accs = cv_fold_accuracies(
        X,
        y,
        "decision_tree",
        best_dt_params,
        n_runs=10,
        k=10,
        progress_tag=DATASET_TAG,
    )
    best_label = f"depth={best_dt_row['max_depth']}, {best_dt_row['criterion']}"
    plot_accuracy_histogram(
        test_accs,
        f"Credit - DT test accuracy distribution ({best_label})",
        os.path.join(out_dir, "dt_test_accuracy_hist.png"),
    )
    plot_accuracy_histogram(
        train_accs,
        f"Credit - DT train accuracy distribution ({best_label})",
        os.path.join(out_dir, "dt_train_accuracy_hist.png"),
    )
    pd.DataFrame({"train_acc": train_accs, "test_acc": test_accs}).to_csv(
        os.path.join(out_dir, "dt_fold_accuracies.csv"), index=False,
    )
    print(
        f"[{DATASET_TAG}] Best DT {best_label}: train acc {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}, "
        f"test acc {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}",
        flush=True,
    )
    print(f"[{DATASET_TAG}] Wrote histograms → dt_*_accuracy_hist.png", flush=True)

    print(f"\n[{DATASET_TAG}] === Random Forest ===", flush=True)
    rf_rows = []
    for criterion in ["info_gain", "gini"]:
        for n_trees in [10, 30, 50, 75, 100, 150]:
            params = {"n_trees": n_trees, "criterion": criterion}
            result = cv_score(X, y, "random_forest", params, multiclass=multiclass)
            result.update({"model": "random_forest", "n_trees": n_trees, "criterion": criterion})
            rf_rows.append(result)
            all_rows.append(result)
            print_cv_result(
                DATASET_TAG,
                "RandomForest",
                f"ntree={n_trees}, criterion={criterion}",
                result,
                multiclass=multiclass,
            )
    rf_df = pd.DataFrame(rf_rows)
    plot_metric_vs_param_grouped(
        rf_df, "n_trees", "criterion", "Credit - Random Forest",
        out_dir, "rf", x_label="ntree",
    )
    print(f"[{DATASET_TAG}] Wrote RF plots → {out_dir}/rf_*.png", flush=True)

    print(f"\n[{DATASET_TAG}] === k-NN on expanded feature matrix (z-score per fold) ===", flush=True)
    knn_rows = []
    for k in [1, 3, 5, 7, 9, 11, 15]:
        result = cv_score(X, y, "knn", {"k": k}, normalize=True, multiclass=multiclass)
        result.update({"model": "knn", "k": k})
        knn_rows.append(result)
        all_rows.append(result)
        print_cv_result(DATASET_TAG, "k-NN", f"k={k}", result, multiclass=multiclass)
    knn_df = pd.DataFrame(knn_rows)
    plot_metric_vs_param(knn_df, "k", "Credit - KNN", out_dir, "knn", x_label="k (neighbors)")
    print(f"[{DATASET_TAG}] Wrote k-NN plots → {out_dir}/knn_*.png", flush=True)

    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, "credit_approval_cv_results.csv"), index=False)
    print(f"\n[{DATASET_TAG}] Done. Saved → {out_dir}/credit_approval_cv_results.csv", flush=True)


if __name__ == "__main__":
    run()
