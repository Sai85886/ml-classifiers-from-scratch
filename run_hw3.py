from __future__ import annotations

# run stratified 5-fold CV for each ntree and show plots on screen.

from typing import Dict, List, Optional, Tuple

try:
    import matplotlib

    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    # You can still print metrics if matplotlib is missing; plots need a separate install.
    plt = None
import numpy as np
import pandas as pd

from random_forest import cross_validate_random_forest

# Values of ntree the assignment asks you to compare.
NTREE_VALUES = [1, 5, 10, 20, 30, 40, 50]

# Toggle plot behavior:
# - If True, each plot is displayed with plt.show() (no saving to files).
# - If False, only metrics are printed.
SHOW_PLOTS = True


class DatasetSpec:
    # Holds one dataset's name, file path, and label column for the loop in main().
    def __init__(
        self,
        name: str,
        csv_path: str,
        label_col: str = "label",
        positive_label: int = 1,
    ) -> None:
        self.name = name
        self.csv_path = csv_path
        self.label_col = label_col
        # Which class counts as "positive" for precision / recall / F1 (binary uses 0 and 1).
        self.positive_label = positive_label


def load_dataset(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(spec.csv_path)
    if spec.label_col not in df.columns:
        raise ValueError(f"Label column '{spec.label_col}' not found in {spec.csv_path}")
    y = df[spec.label_col]
    X = df.drop(columns=[spec.label_col])

    # Our CSVs name numeric columns with _num; everything else we treat as categorical.
    for col in X.columns:
        if col.endswith("_num"):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].astype(object)

    # Simple cleanup if something could not be parsed as a number.
    if X.isnull().any().any():
        X = X.fillna(X.mean(numeric_only=True))
        for col in X.columns:
            if not col.endswith("_num"):
                X[col] = X[col].fillna("MISSING")

    return X, y


def plot_metric_vs_ntree(
    dataset_name: str,
    metric_name: str,
    ntree_values: List[int],
    metric_values: List[float],
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it (e.g., `python3 -m pip install matplotlib`) "
            "or run metrics-only and generate plots elsewhere."
        )
    # One plot for one dataset + one metric.
    plt.figure(figsize=(7, 4.5))
    plt.plot(ntree_values, metric_values, marker="o")
    plt.xlabel("ntree")
    plt.ylabel(metric_name)
    plt.title(f"{dataset_name}: {metric_name} vs ntree (5-fold stratified CV)")
    plt.xticks(ntree_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def run_one_dataset(
    spec: DatasetSpec,
    *,
    k: int = 5,
    random_state: int = 0,
    m_try: Optional[int] = None,
    max_depth: Optional[int] = None,
    min_size_for_split: int = 2,
    min_gain: float = 0.0,
) -> Dict[str, List[float]]:
    X, y = load_dataset(spec)

    # Collect one list per metric; index lines up with NTREE_VALUES in order.
    results: Dict[str, List[float]] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for ntree in NTREE_VALUES:
        metrics = cross_validate_random_forest(
            X,
            y,
            k=k,
            ntree=ntree,
            m_try=m_try,
            max_depth=max_depth,
            min_size_for_split=min_size_for_split,
            min_gain=min_gain,
            random_state=random_state,
            positive_label=spec.positive_label,
        )
        for key in results:
            results[key].append(metrics[key])
        print(f"[{spec.name}] ntree={ntree} -> {metrics}")

    return results


def pick_best_stopping_params_for_f1(
    spec: DatasetSpec,
    *,
    k: int,
    random_state: int,
    m_try: Optional[int],
) -> Tuple[Optional[int], int, float]:
    # We tune the tree stopping rules by looking at cross-validated F1 for one "middle" ntree value.
    # This is a practical shortcut so we don't grid-search for every ntree.
    X, y = load_dataset(spec)

    # Grid you asked for: max_depth, min_size_for_split, min_gain.
    max_depth_grid = [None, 5, 10]
    min_size_grid = [2, 5, 10]
    min_gain_grid = [0.0, 0.001, 0.01]

    # Faster tuning: fewer trees + fewer folds. We still use stratified CV.
    k_tuning = min(3, k)
    ntree_for_tuning = 10

    best_f1 = -1.0
    best_params = (None, 2, 0.0)

    print(f"[{spec.name}] tuning stopping params for F1...", flush=True)
    for max_depth in max_depth_grid:
        for min_size_for_split in min_size_grid:
            for min_gain in min_gain_grid:
                metrics = cross_validate_random_forest(
                    X,
                    y,
                    k=k_tuning,
                    ntree=ntree_for_tuning,
                    m_try=m_try,
                    max_depth=max_depth,
                    min_size_for_split=min_size_for_split,
                    min_gain=min_gain,
                    random_state=random_state,
                    positive_label=spec.positive_label,
                )
                f1 = metrics["f1"]
                print(
                    f"[{spec.name}] try max_depth={max_depth}, min_size={min_size_for_split}, min_gain={min_gain} -> f1={f1:.4f}",
                    flush=True,
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (max_depth, min_size_for_split, min_gain)

    return best_params


def main() -> None:
    datasets = [
        DatasetSpec(name="WDBC", csv_path="wdbc.csv", label_col="label", positive_label=1),
        DatasetSpec(name="Loan", csv_path="loan.csv", label_col="label", positive_label=1),
        # Extra credit datasets from the HW3 PDF:
        DatasetSpec(name="Raisin", csv_path="raisin.csv", label_col="label", positive_label=1),
        DatasetSpec(name="Titanic", csv_path="titanic.csv", label_col="label", positive_label=1),
    ]

    # Tree stopping rules and randomness: we'll tune these by F1 per dataset.
    k = 5
    random_state = 0
    # None -> random_forest uses sqrt(number of features) at each split (assignment suggestion).
    m_try = None

    for spec in datasets:
        max_depth, min_size_for_split, min_gain = pick_best_stopping_params_for_f1(
            spec,
            k=k,
            random_state=random_state,
            m_try=m_try,
        )
        print(
            f"[{spec.name}] best stopping params by F1: "
            f"max_depth={max_depth}, min_size_for_split={min_size_for_split}, min_gain={min_gain}",
            flush=True,
        )

        results = run_one_dataset(
            spec,
            k=k,
            random_state=random_state,
            m_try=m_try,
            max_depth=max_depth,
            min_size_for_split=min_size_for_split,
            min_gain=min_gain,
        )

        for metric_name, metric_values in results.items():
            if plt is not None:
                plot_metric_vs_ntree(spec.name, metric_name, NTREE_VALUES, metric_values)
            else:
                # Fallback if you run without matplotlib: copy these lists into your report by hand.
                print(f"[no-plot] {spec.name} {metric_name}: {metric_values}")

    print("Done.")


if __name__ == "__main__":
    main()

