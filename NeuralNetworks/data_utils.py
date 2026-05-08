# Load WDBC / Loan (etc.) CSVs; numpy-only feature transforms.
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np


def _read_csv(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if len(row) == len(header)]
    return header, rows


def load_wdbc(path: str) -> tuple[np.ndarray, np.ndarray]:
    header, rows = _read_csv(path)
    feat_idx = [i for i, h in enumerate(header) if h != "label"]
    label_idx = header.index("label")
    X = np.array([[float(r[i]) for i in feat_idx] for r in rows], dtype=np.float64).T
    y = np.array([[float(r[label_idx])] for r in rows], dtype=np.float64).T
    return X, y


def load_raisin(path: str) -> tuple[np.ndarray, np.ndarray]:
    header, rows = _read_csv(path)
    feat_idx = [i for i, h in enumerate(header) if h != "label"]
    label_idx = header.index("label")
    X = np.array([[float(r[i]) for i in feat_idx] for r in rows], dtype=np.float64).T
    y = np.array([[float(r[label_idx])] for r in rows], dtype=np.float64).T
    return X, y


def load_loan(path: str) -> tuple[np.ndarray, np.ndarray]:
    header, rows = _read_csv(path)
    label_idx = header.index("label")
    cat_cols = [i for i, h in enumerate(header) if h.endswith("_cat")]
    num_cols = [i for i, h in enumerate(header) if h.endswith("_num")]

    vocab: dict[int, dict[str, int]] = {}
    for c in cat_cols:
        vals = sorted({r[c] for r in rows})
        vocab[c] = {v: j for j, v in enumerate(vals)}

    blocks: list[np.ndarray] = []
    for c in cat_cols:
        k = len(vocab[c])
        mat = np.zeros((k, len(rows)), dtype=np.float64)
        for j, r in enumerate(rows):
            mat[vocab[c][r[c]], j] = 1.0
        blocks.append(mat)
    nums = np.array([[float(r[c]) for c in num_cols] for r in rows], dtype=np.float64).T
    blocks.append(nums)
    X = np.vstack(blocks)
    y = np.array([[float(r[label_idx])] for r in rows], dtype=np.float64).T
    return X, y


def load_titanic(path: str) -> tuple[np.ndarray, np.ndarray]:
    # Columns: label, *_cat, *_num (one-hot categoricals + numeric block).
    header, rows = _read_csv(path)
    label_idx = header.index("label")
    cat_cols = [i for i, h in enumerate(header) if h.endswith("_cat")]
    num_cols = [i for i, h in enumerate(header) if h.endswith("_num")]

    vocab: dict[int, dict[str, int]] = {}
    for c in cat_cols:
        vals = sorted({r[c] for r in rows})
        vocab[c] = {v: j for j, v in enumerate(vals)}

    blocks: list[np.ndarray] = []
    for c in cat_cols:
        k = len(vocab[c])
        mat = np.zeros((k, len(rows)), dtype=np.float64)
        for j, r in enumerate(rows):
            mat[vocab[c][r[c]], j] = 1.0
        blocks.append(mat)

    nums = np.array([[float(r[c]) for c in num_cols] for r in rows], dtype=np.float64).T
    blocks.append(nums)
    X = np.vstack(blocks)
    y = np.array([[float(r[label_idx])] for r in rows], dtype=np.float64).T
    return X, y


def standardize(
    x_train: np.ndarray, x_other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=1, keepdims=True)
    sigma = x_train.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x_train - mu) / sigma, (x_other - mu) / sigma


def stratified_kfold_indices(y: np.ndarray, k: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    # y shape (1, m), binary. Returns (train_idx, val_idx) per fold.
    y_flat = y.astype(np.int32).ravel()
    idx_by_class: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(y_flat):
        idx_by_class[int(lab)].append(i)
    folds_train: list[list[int]] = [[] for _ in range(k)]
    for _, inds in idx_by_class.items():
        inds = np.array(inds, dtype=np.int32)
        rng.shuffle(inds)
        for j, ix in enumerate(inds):
            folds_train[j % k].append(int(ix))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(k):
        val_idx = np.array(folds_train[f], dtype=np.int32)
        train_idx = np.concatenate([folds_train[g] for g in range(k) if g != f]).astype(np.int32)
        splits.append((train_idx, val_idx))
    return splits
