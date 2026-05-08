# Extra Credit 2: OpenML Zoo (multiclass, mixed types). k-NN, DT, RF, NN.
import os

import pandas as pd

from util import (
    cv_score,
    load_zoo_data,
    nn_learning_curve,
    plot_learning_curve,
    plot_metric_vs_param,
    plot_metric_vs_param_grouped,
    print_cv_result,
)

DATASET_TAG = "zoo (EC2)"


def run():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "outputs", "extra_credit2_zoo")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{DATASET_TAG}] Fetching/caching Zoo via OpenML (network on first run)...", flush=True)
    X, y = load_zoo_data()
    multiclass = True
    all_rows = []

    print(
        f"[{DATASET_TAG}] Loaded {len(X)} examples, {X.shape[1]} features, {y.nunique()} classes.",
        flush=True,
    )

    print(f"\n[{DATASET_TAG}] === k-NN ===", flush=True)
    knn_rows = []
    for k in [1, 3, 5, 7, 9, 11]:
        result = cv_score(X, y, "knn", {"k": k}, normalize=True, multiclass=multiclass)
        result.update({"model": "knn", "k": k})
        knn_rows.append(result)
        all_rows.append(result)
        print_cv_result(DATASET_TAG, "k-NN", f"k={k}", result, multiclass=multiclass)
    knn_df = pd.DataFrame(knn_rows)
    plot_metric_vs_param(knn_df, "k", "Zoo - KNN", out_dir, "knn", x_label="k (neighbors)")
    print(f"[{DATASET_TAG}] Wrote k-NN plots → {out_dir}/knn_*.png", flush=True)

    print(f"\n[{DATASET_TAG}] === Decision Tree ===", flush=True)
    dt_rows = []
    for criterion in ["info_gain", "gini"]:
        for max_depth in [5, 10, 15, 20, 30, None]:
            params = {"max_depth": max_depth, "criterion": criterion, "min_size_for_split": 2}
            result = cv_score(X, y, "decision_tree", params, multiclass=multiclass)
            depth_label = -1 if max_depth is None else max_depth
            result.update({"model": "decision_tree", "max_depth": depth_label, "criterion": criterion})
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
        dt_df, "max_depth", "criterion", "Zoo - Decision Tree",
        out_dir, "dt", x_label="max_depth (-1 = unlimited)",
    )
    print(f"[{DATASET_TAG}] Wrote DT plots → {out_dir}/dt_*.png", flush=True)

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
        rf_df, "n_trees", "criterion", "Zoo - Random Forest",
        out_dir, "rf", x_label="ntree",
    )
    print(f"[{DATASET_TAG}] Wrote RF plots → {out_dir}/rf_*.png", flush=True)

    print(f"\n[{DATASET_TAG}] === Neural network grid ===", flush=True)
    nn_grid = [
        {"layer_sizes": [16], "lam": 0.001, "alpha": 0.5, "max_epochs": 120, "batch_size": 32},
        {"layer_sizes": [16], "lam": 0.01, "alpha": 0.5, "max_epochs": 120, "batch_size": 32},
        {"layer_sizes": [32], "lam": 0.001, "alpha": 0.5, "max_epochs": 120, "batch_size": 32},
        {"layer_sizes": [32], "lam": 0.01, "alpha": 0.5, "max_epochs": 120, "batch_size": 32},
        {"layer_sizes": [32, 16], "lam": 0.001, "alpha": 0.5, "max_epochs": 120, "batch_size": 32},
        {"layer_sizes": [32, 16], "lam": 0.01, "alpha": 0.5, "max_epochs": 120, "batch_size": 32},
    ]
    nn_rows = []
    for params in nn_grid:
        result = cv_score(X, y, "neural_network", params, normalize=True, multiclass=multiclass)
        label = f"{params['layer_sizes']}, lam={params['lam']}"
        result.update({
            "model": "neural_network",
            "config": label,
            "layer_sizes": str(params["layer_sizes"]),
            "lam": params["lam"],
        })
        nn_rows.append(result)
        all_rows.append(result)
        print_cv_result(DATASET_TAG, "NeuralNet", label, result, multiclass=multiclass)
    nn_df = pd.DataFrame(nn_rows).sort_values("mean_f1", ascending=False).reset_index(drop=True)
    nn_df.to_csv(os.path.join(out_dir, "nn_configs.csv"), index=False)

    best_label = nn_df.iloc[0]["config"]
    best_params = nn_grid[[i for i, p in enumerate(nn_grid)
                           if f"{p['layer_sizes']}, lam={p['lam']}" == best_label][0]]
    print(f"[{DATASET_TAG}] Best NN config by mean macro-F1: {best_label}", flush=True)

    print(f"[{DATASET_TAG}] NN learning curve...", flush=True)
    lc_df = nn_learning_curve(
        X, y, best_params, multiclass=multiclass, normalize=True, progress_tag=DATASET_TAG,
    )
    lc_df.to_csv(os.path.join(out_dir, "nn_learning_curve.csv"), index=False)
    plot_learning_curve(
        lc_df, f"Zoo - NN learning curve ({best_label})",
        os.path.join(out_dir, "nn_learning_curve.png"),
    )
    print(f"[{DATASET_TAG}] Wrote NN learning curve → {out_dir}/nn_learning_curve.png", flush=True)

    results_df = pd.DataFrame(all_rows).sort_values(["mean_f1", "mean_acc"], ascending=False)
    results_df.to_csv(os.path.join(out_dir, "extra_credit2_zoo_cv_results.csv"), index=False)
    print(f"\n[{DATASET_TAG}] Done. Saved → {out_dir}/extra_credit2_zoo_cv_results.csv", flush=True)


if __name__ == "__main__":
    run()
