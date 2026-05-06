import os

import pandas as pd

from util import load_parkinsons_data, run_cv, save_basic_plot


def run():
    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "outputs", "parkinsons")
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_parkinsons_data(os.path.join(root, "parkinsons.csv"))

    knn_grid = [{"k": k} for k in [1, 3, 5, 7, 9, 11]]
    rf_grid = [{"n_trees": n} for n in [10, 30, 50, 75, 100, 150]]
    nn_grid = [
        {"layer_sizes": [8], "lam": 0.0, "alpha": 0.7, "max_epochs": 120, "batch_size": 64},
        {"layer_sizes": [8], "lam": 0.01, "alpha": 0.7, "max_epochs": 120, "batch_size": 64},
        {"layer_sizes": [16], "lam": 0.0, "alpha": 0.7, "max_epochs": 120, "batch_size": 64},
        {"layer_sizes": [16], "lam": 0.01, "alpha": 0.7, "max_epochs": 120, "batch_size": 64},
        {"layer_sizes": [8, 8], "lam": 0.01, "alpha": 0.7, "max_epochs": 120, "batch_size": 64},
        {"layer_sizes": [16, 8], "lam": 0.1, "alpha": 0.7, "max_epochs": 120, "batch_size": 64},
    ]

    knn_df = run_cv(X, y, "knn", knn_grid, normalize=True, multiclass=False, positive_label=1)
    rf_df = run_cv(X, y, "random_forest", rf_grid, normalize=False, multiclass=False, positive_label=1)
    nn_df = run_cv(X, y, "neural_network", nn_grid, normalize=True, multiclass=False, positive_label=1)

    save_basic_plot(knn_df, "Parkinsons - KNN configs", os.path.join(out_dir, "knn_f1_curve.png"))
    save_basic_plot(rf_df, "Parkinsons - Random Forest configs", os.path.join(out_dir, "rf_f1_curve.png"))
    save_basic_plot(nn_df, "Parkinsons - Neural Network configs", os.path.join(out_dir, "nn_f1_curve.png"))

    final_df = pd.concat([knn_df, rf_df, nn_df], ignore_index=True)
    final_df.to_csv(os.path.join(out_dir, "parkinsons_cv_results.csv"), index=False)
    print("Saved parkinsons results in outputs/parkinsons")


if __name__ == "__main__":
    run()
