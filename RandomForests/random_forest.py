from __future__ import annotations

# Random forest: many trees on bootstrap samples, each split only considers m_try features.
# Cross-validation averages metrics over folds so one unlucky split does not dominate.

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from decision_tree import DecisionTreeClassifier, _majority_class


def bootstrap_sample(X: pd.DataFrame, y: pd.Series, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.Series]:
    # Same size as the training set, but rows are picked with replacement (classic bagging).
    n = len(y)
    idx = rng.integers(0, n, size=n)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def majority_vote(preds: Sequence[Any]) -> Any:
    # Each tree votes for a class; we take whichever label shows up most often.
    return _majority_class(np.asarray(preds))


class RandomForestClassifier:
    def __init__(
        self,
        n_trees: int = 10,
        m_try: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_size_for_split: int = 2,
        min_gain: float = 0.0,
        criterion: str = "info_gain",
        random_state: Optional[int] = None,
    ) -> None:
        self.n_trees = n_trees
        # How many features to consider at each split; None means sqrt(#features), per HW3 hint.
        self.m_try = m_try
        self.max_depth = max_depth
        self.min_size_for_split = min_size_for_split
        self.min_gain = min_gain
        # Same as the underlying decision tree: "info_gain" (entropy) or "gini".
        self.criterion = criterion
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.trees: List[DecisionTreeClassifier] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestClassifier":
        self.trees = []

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        m_try = self.m_try
        if m_try is None:
            m_try = max(1, int(np.sqrt(X.shape[1])))

        for _ in range(int(self.n_trees)):
            # Fresh bootstrap sample for this tree so trees are not identical.
            Xb, yb = bootstrap_sample(X, y, self._rng)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_size_for_split=self.min_size_for_split,
                min_gain=self.min_gain,
                criterion=self.criterion,
                # Give each tree its own RNG so they differ even with the same forest seed.
                random_state=int(self._rng.integers(0, 2**31 - 1)),
            )
            tree.fit(Xb, yb, m_try=m_try)
            self.trees.append(tree)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.trees:
            raise ValueError("RandomForestClassifier is not fitted.")

        # Rows are test examples; columns are trees. Then we vote per example.
        all_tree_preds = np.vstack([t.predict(X) for t in self.trees])
        out: List[Any] = []
        for j in range(all_tree_preds.shape[1]):
            out.append(majority_vote(all_tree_preds[:, j]))
        return np.asarray(out)


def stratified_kfold_indices(y: Sequence[Any], k: int, *, rng: np.random.Generator) -> List[np.ndarray]:
    # Build k folds so each class is spread across folds in roughly the same ratio as the full data.
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)

    per_class_indices: Dict[Any, np.ndarray] = {}
    for c in classes:
        idx = np.where(y_arr == c)[0]
        rng.shuffle(idx)
        per_class_indices[c] = idx

    folds: List[List[int]] = [[] for _ in range(k)]
    for c in classes:
        idx = per_class_indices[c]
        # Round-robin: send the i-th example of this class to fold (i mod k).
        for i, sample_idx in enumerate(idx):
            folds[i % k].append(int(sample_idx))

    return [np.asarray(sorted(f), dtype=int) for f in folds]


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int], positive_label: int = 1) -> Tuple[int, int, int, int]:
    # Standard 2x2 table for binary problems (we treat one label as "positive" for precision/recall).
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    pos = positive_label
    neg = 0 if pos != 0 else 1

    tp = int(np.sum((yt == pos) & (yp == pos)))
    tn = int(np.sum((yt == neg) & (yp == neg)))
    fp = int(np.sum((yt == neg) & (yp == pos)))
    fn = int(np.sum((yt == pos) & (yp == neg)))
    return tp, tn, fp, fn


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], positive_label: int = 1) -> Dict[str, float]:
    tp, tn, fp, fn = confusion_counts(y_true, y_pred, positive_label=positive_label)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def cross_validate_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    k: int = 5,
    ntree: int,
    m_try: Optional[int] = None,
    max_depth: Optional[int] = None,
    min_size_for_split: int = 2,
    min_gain: float = 0.0,
    random_state: int = 0,
    positive_label: int = 1,
) -> Dict[str, float]:
    # Train a full forest on k-1 folds, test on the held-out fold; repeat k times and average.
    rng = np.random.default_rng(random_state)
    folds = stratified_kfold_indices(y, k, rng=rng)

    metrics_per_fold: List[Dict[str, float]] = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        rf = RandomForestClassifier(
            n_trees=ntree,
            m_try=m_try,
            max_depth=max_depth,
            min_size_for_split=min_size_for_split,
            min_gain=min_gain,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        metrics_per_fold.append(classification_metrics(y_test.to_numpy(), preds, positive_label=positive_label))

    # One number per metric: mean over the k folds (what HW3 asks for).
    out: Dict[str, float] = {}
    for key in ("accuracy", "precision", "recall", "f1"):
        out[key] = float(np.mean([m[key] for m in metrics_per_fold]))
    return out

