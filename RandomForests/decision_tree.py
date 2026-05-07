from __future__ import annotations

# Decision tree built from HW1-style entropy + information gain, but split into a class
# so the random forest can call fit() / predict() on pandas tables.

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def entropy(y: Sequence[Any]) -> float:
    # How mixed the labels are: high when classes are balanced, low when one class dominates.
    values, counts = np.unique(np.asarray(y), return_counts=True)
    probs = counts / max(len(y), 1)
    # Tiny epsilon avoids log(0) if something weird slips through.
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def gini(y: Sequence[Any]) -> float:
    # Same idea as entropy but uses 1 - sum(p^2). Used as an alternative split criterion.
    values, counts = np.unique(np.asarray(y), return_counts=True)
    probs = counts / max(len(y), 1)
    return float(1.0 - np.sum(probs ** 2))


def _impurity_fn(criterion: str):
    if criterion == "gini":
        return gini
    return entropy


def _majority_class(y: Sequence[Any]) -> Any:
    # When we stop splitting, guess the most common label in this bucket.
    values, counts = np.unique(np.asarray(y), return_counts=True)
    return values[int(np.argmax(counts))]


def _is_numeric_series(s: pd.Series) -> bool:
    # Pandas tells us if the column is stored as numbers (we use that for split rules).
    return pd.api.types.is_numeric_dtype(s)


def _split_gain_categorical(x_col: pd.Series, y: pd.Series, parent_impurity: float, impurity_fn) -> float:
    # Split quality for a categorical column: branch one child per category value.
    values, counts = np.unique(x_col.to_numpy(), return_counts=True)
    weighted = 0.0
    n = len(y)
    for v, c in zip(values, counts):
        subset_y = y[x_col == v]
        # Weight each branch by how many rows go there.
        weighted += (c / n) * impurity_fn(subset_y)
    return float(parent_impurity - weighted)


def _split_gain_numeric(
    x_col: pd.Series, y: pd.Series, threshold: float, parent_impurity: float, impurity_fn
) -> float:
    # Split quality for a number: rows <= threshold go left, the rest go right.
    left_mask = x_col <= threshold
    right_mask = ~left_mask

    n = len(y)
    if n == 0:
        return 0.0

    y_left = y[left_mask]
    y_right = y[right_mask]

    # No real split if everyone lands on one side.
    if len(y_left) == 0 or len(y_right) == 0:
        return 0.0

    weighted = (len(y_left) / n) * impurity_fn(y_left) + (len(y_right) / n) * impurity_fn(y_right)
    return float(parent_impurity - weighted)


def information_gain_categorical(x_col: pd.Series, y: pd.Series) -> float:
    # Backwards-compat wrapper: entropy-based information gain.
    return _split_gain_categorical(x_col, y, entropy(y), entropy)


def information_gain_numeric(x_col: pd.Series, y: pd.Series, threshold: float) -> float:
    # Backwards-compat wrapper: entropy-based information gain.
    return _split_gain_numeric(x_col, y, threshold, entropy(y), entropy)


class _Leaf:
    # End of a branch: we output one class label.
    def __init__(self, prediction: Any) -> None:
        self.prediction = prediction


class _CatNode:
    # Branch on one categorical feature: each value gets its own subtree.
    # If we see a category at test time that never appeared in training, use fallback.
    def __init__(self, feature: str, children: Dict[Any, Any], fallback: Any) -> None:
        self.feature = feature
        self.children = children
        self.fallback = fallback


class _NumNode:
    # Branch on one numeric feature: compare to a threshold (we use the mean at training time).
    def __init__(self, feature: str, threshold: float, left: Any, right: Any, fallback: Any) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.fallback = fallback


class DecisionTreeClassifier:
    # Decision tree using Information Gain.
    #
    # - Supports categorical and numerical features.
    # - For numeric features, the split threshold is the mean value at the node.
    # - Supports per-split feature subsampling (m_try), used by Random Forests.

    def __init__(
        self,
        *,
        max_depth: Optional[int] = None,
        min_size_for_split: int = 2,
        min_gain: float = 0.0,
        criterion: str = "info_gain",
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_size_for_split = int(min_size_for_split)
        self.min_gain = float(min_gain)
        # "info_gain" uses entropy (HW1 default); "gini" uses 1 - sum(p^2).
        self.criterion = criterion
        self.random_state = random_state
        # Randomness only matters when we subsample features at each split (random forest).
        self._rng = np.random.default_rng(random_state)
        self._root: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series, *, m_try: Optional[int] = None) -> "DecisionTreeClassifier":
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self._root = self._build_tree(X, y, depth=0, m_try=m_try)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._root is None:
            raise ValueError("DecisionTreeClassifier is not fitted.")
        preds: List[Any] = []
        # Walk each row down the tree from the root.
        for _, row in X.iterrows():
            preds.append(self._predict_one(self._root, row))
        return np.asarray(preds)

    def _predict_one(self, node: Any, x: pd.Series) -> Any:
        if isinstance(node, _Leaf):
            return node.prediction
        if isinstance(node, _CatNode):
            v = x.get(node.feature)
            child = node.children.get(v)
            if child is None:
                return node.fallback
            return self._predict_one(child, x)
        if isinstance(node, _NumNode):
            v = x.get(node.feature)
            try:
                v_float = float(v)
            except Exception:
                # Bad or missing value: safest is to use the majority vote we stored at this node.
                return node.fallback
            if v_float <= node.threshold:
                return self._predict_one(node.left, x)
            return self._predict_one(node.right, x)
        raise TypeError(f"Unknown node type: {type(node)}")

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, *, depth: int, m_try: Optional[int]) -> Any:
        # If every row here has the same label, we are done.
        if len(np.unique(y.to_numpy())) == 1:
            return _Leaf(_majority_class(y))

        # Stop if we hit a depth limit, too few rows, or no features left.
        if self.max_depth is not None and depth >= self.max_depth:
            return _Leaf(_majority_class(y))
        if len(y) < self.min_size_for_split:
            return _Leaf(_majority_class(y))
        if X.shape[1] == 0:
            return _Leaf(_majority_class(y))

        features = list(X.columns)
        # Random forest: only look at m_try random features when choosing the best split.
        if m_try is not None and m_try < len(features):
            features = list(self._rng.choice(features, size=m_try, replace=False))

        impurity = _impurity_fn(self.criterion)
        parent_impurity = impurity(y)

        best_gain = -1.0
        best_feature: Optional[str] = None
        best_is_numeric = False
        best_threshold: Optional[float] = None

        for feature in features:
            col = X[feature]
            if _is_numeric_series(col):
                # HW3 allows using the mean as the split point for numeric attributes.
                thr = float(col.mean())
                gain = _split_gain_numeric(col.astype(float), y, thr, parent_impurity, impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_is_numeric = True
                    best_threshold = thr
            else:
                gain = _split_gain_categorical(col.astype(object), y, parent_impurity, impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_is_numeric = False
                    best_threshold = None

        # Nothing useful to split on (or gain too small).
        if best_feature is None or best_gain < self.min_gain:
            return _Leaf(_majority_class(y))

        # Used for unknown categories at prediction time.
        fallback = _majority_class(y)

        if best_is_numeric:
            assert best_threshold is not None
            col = X[best_feature].astype(float)
            left_mask = col <= best_threshold
            right_mask = ~left_mask
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                return _Leaf(fallback)

            left = self._build_tree(X[left_mask], y[left_mask], depth=depth + 1, m_try=m_try)
            right = self._build_tree(X[right_mask], y[right_mask], depth=depth + 1, m_try=m_try)
            return _NumNode(feature=best_feature, threshold=float(best_threshold), left=left, right=right, fallback=fallback)

        # Categorical: one branch per value; drop that column in children so we do not reuse it.
        children: Dict[Any, Any] = {}
        for v in X[best_feature].unique():
            mask = X[best_feature] == v
            child = self._build_tree(
                X[mask].drop(columns=[best_feature]),
                y[mask],
                depth=depth + 1,
                m_try=m_try,
            )
            children[v] = child

        return _CatNode(feature=best_feature, children=children, fallback=fallback)

