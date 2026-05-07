import os

import pandas as pd

from util import (
    cv_score,
    load_rice_data,
    nn_learning_curve,
    plot_learning_curve,
    plot_metric_vs_param,
    plot_metric_vs_param_grouped,
)


def run():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "outputs", "rice")
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_rice_data(os.path.join(root, "rice.csv"))
    multiclass = False
    all_rows = []

    knn_rows = []
    for k in [1, 3, 5, 7, 9, 11, 15]:
        result = cv_score(X, y, "knn", {"k": k}, normalize=True, multiclass=multiclass)
        result.update({"model": "knn", "k": k})
        knn_rows.append(result)
        all_rows.append(result)
    knn_df = pd.DataFrame(knn_rows)
    plot_metric_vs_param(knn_df, "k", "Rice - KNN", out_dir, "knn", x_label="k (neighbors)")

    rf_rows = []
    for criterion in ["info_gain", "gini"]:
        for n_trees in [10, 30, 50, 75, 100, 150]:
            params = {"n_trees": n_trees, "criterion": criterion}
            result = cv_score(X, y, "random_forest", params, multiclass=multiclass)
            result.update({"model": "random_forest", "n_trees": n_trees, "criterion": criterion})
            rf_rows.append(result)
            all_rows.append(result)
    rf_df = pd.DataFrame(rf_rows)
    plot_metric_vs_param_grouped(
        rf_df, "n_trees", "criterion", "Rice - Random Forest",
        out_dir, "rf", x_label="ntree",
    )

    nn_grid = [
        {"layer_sizes": [8], "lam": 0.0, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [8], "lam": 0.01, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [16], "lam": 0.01, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [16], "lam": 0.1, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [16, 8], "lam": 0.01, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [8, 8], "lam": 0.1, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
    ]
    nn_rows = []
    for params in nn_grid:
        result = cv_score(X, y, "neural_network", params, normalize=True, multiclass=multiclass)
        label = f"{params['layer_sizes']}, lam={params['lam']}"
        result.update({"model": "neural_network", "config": label, "layer_sizes": str(params["layer_sizes"]), "lam": params["lam"]})
        nn_rows.append(result)
        all_rows.append(result)
    nn_df = pd.DataFrame(nn_rows).sort_values("mean_f1", ascending=False).reset_index(drop=True)
    nn_df.to_csv(os.path.join(out_dir, "nn_configs.csv"), index=False)

    best_label = nn_df.iloc[0]["config"]
    best_params = nn_grid[[i for i, p in enumerate(nn_grid)
                           if f"{p['layer_sizes']}, lam={p['lam']}" == best_label][0]]

    lc_df = nn_learning_curve(X, y, best_params, multiclass=multiclass, normalize=True)
    lc_df.to_csv(os.path.join(out_dir, "nn_learning_curve.csv"), index=False)
    plot_learning_curve(lc_df, f"Rice - NN learning curve ({best_label})",
                        os.path.join(out_dir, "nn_learning_curve.png"))

    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, "rice_cv_results.csv"), index=False)
    print(f"Saved rice results in {out_dir}")


if __name__ == "__main__":
    run()
