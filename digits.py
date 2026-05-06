import os

import pandas as pd

from util import load_digits_data, run_cv, save_basic_plot


def run():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "outputs", "digits")
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_digits_data()

    all_results = []

    knn_grid = [{"k": k} for k in [1, 3, 5, 7, 9, 11]]
    knn_df = run_cv(X, y, "knn", knn_grid, normalize=True, multiclass=True)
    all_results.append(knn_df)
    save_basic_plot(knn_df, "Digits - KNN configs", os.path.join(out_dir, "knn_f1_curve.png"))

    rf_grid = [{"n_trees": n} for n in [10, 30, 50, 75, 100, 150]]
    rf_df = run_cv(X, y, "random_forest", rf_grid, normalize=False, multiclass=True)
    all_results.append(rf_df)
    save_basic_plot(rf_df, "Digits - Random Forest configs", os.path.join(out_dir, "rf_f1_curve.png"))

    nn_grid = [
        {"layer_sizes": [32], "lam": 0.0, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [32], "lam": 0.01, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [64], "lam": 0.01, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [64], "lam": 0.1, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [64, 32], "lam": 0.01, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
        {"layer_sizes": [32, 32], "lam": 0.1, "alpha": 0.5, "max_epochs": 80, "batch_size": 128},
    ]
    nn_df = run_cv(X, y, "neural_network", nn_grid, normalize=True, multiclass=True)
    all_results.append(nn_df)
    save_basic_plot(nn_df, "Digits - Neural Network configs", os.path.join(out_dir, "nn_f1_curve.png"))

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(os.path.join(out_dir, "digits_cv_results.csv"), index=False)
    print("Saved digits results in outputs/digits")


if __name__ == "__main__":
    run()
