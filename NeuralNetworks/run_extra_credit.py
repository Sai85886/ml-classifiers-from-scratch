# Extra credit: Raisin + Titanic; same protocol as run_experiments.py.
from __future__ import annotations

import json
import os

from data_utils import load_raisin, load_titanic
from run_experiments import (
    OUT_DIR,
    TRAINING,
    learning_curve,
    pick_best,
    run_all_experiments,
    save_results_table,
)

BASE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    # Alternate seed for an additional valid rerun.
    TRAINING["random_seed"] = 777
    meta = {
        "extra_credit": True,
        "stopping_criterion": "fixed_number_of_epochs",
        "m_epochs": TRAINING["max_epochs"],
        "alpha": TRAINING["alpha"],
        "batch_size": TRAINING["batch_size"],
        "k_folds": TRAINING["k_folds"],
    }
    with open(
        os.path.join(OUT_DIR, "experiment_config_extra_credit.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, indent=2)

    Xr, yr = load_raisin(os.path.join(BASE, "datasets", "raisin.csv"))
    Xt, yt = load_titanic(os.path.join(BASE, "datasets", "titanic.csv"))

    res_r = run_all_experiments("Raisin", Xr, yr)
    save_results_table(res_r, os.path.join(OUT_DIR, "cv_raisin.csv"))

    res_t = run_all_experiments("Titanic", Xt, yt)
    save_results_table(res_t, os.path.join(OUT_DIR, "cv_titanic.csv"))

    br = pick_best(res_r)
    bt = pick_best(res_t)
    print("\nBest Raisin (by mean F1):", br.hidden, "lambda=", br.lam, br.mean_f1)
    print("Best Titanic:", bt.hidden, "lambda=", bt.lam, bt.mean_f1)

    learning_curve("Raisin", Xr, yr, br.hidden, br.lam)
    learning_curve("Titanic", Xt, yt, bt.hidden, bt.lam)


if __name__ == "__main__":
    main()
