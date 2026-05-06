import os

import pandas as pd

from util import load_credit_data, run_cv, save_basic_plot


def run():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "outputs", "credit_approval")
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_credit_data(os.path.join(root, "credit_approval.csv"))

    dt_grid = [
        {"max_depth": None, "min_size_for_split": 2, "min_gain": 0.0},
        {"max_depth": 4, "min_size_for_split": 2, "min_gain": 0.0},
        {"max_depth": 6, "min_size_for_split": 2, "min_gain": 0.0},
        {"max_depth": 8, "min_size_for_split": 5, "min_gain": 0.0},
        {"max_depth": 10, "min_size_for_split": 5, "min_gain": 0.001},
        {"max_depth": 12, "min_size_for_split": 10, "min_gain": 0.001},
    ]
    rf_grid = [{"n_trees": n} for n in [10, 30, 50, 75, 100, 150]]
    knn_grid = [{"k": k} for k in [1, 3, 5, 7, 9, 11]]

    dt_df = run_cv(X, y, "decision_tree", dt_grid, normalize=False, multiclass=False, positive_label=1)
    rf_df = run_cv(X, y, "random_forest", rf_grid, normalize=False, multiclass=False, positive_label=1)
    knn_df = run_cv(X, y, "knn", knn_grid, normalize=True, multiclass=False, positive_label=1)

    save_basic_plot(dt_df, "Credit - Decision Tree configs", os.path.join(out_dir, "dt_f1_curve.png"))
    save_basic_plot(rf_df, "Credit - Random Forest configs", os.path.join(out_dir, "rf_f1_curve.png"))
    save_basic_plot(knn_df, "Credit - KNN configs", os.path.join(out_dir, "knn_f1_curve.png"))

    final_df = pd.concat([dt_df, rf_df, knn_df], ignore_index=True)
    final_df.to_csv(os.path.join(out_dir, "credit_approval_cv_results.csv"), index=False)
    print("Saved credit results in outputs/credit_approval")


if __name__ == "__main__":
    run()
