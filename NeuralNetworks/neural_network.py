# Feedforward NN: sigmoid, L2. Vectorized forward/backprop. Binary/multiclass = sigmoid outputs + logistic loss.
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad_from_a(a: np.ndarray) -> np.ndarray:
    # dσ/dz given a = σ(z).
    return a * (1.0 - a)


class NeuralNetwork:
    # layer_sizes: [n0..nL] without bias. Theta[i] shape (n_{i+1}, n_i+1); col 0 = bias.

    def __init__(self, layer_sizes: list[int], rng: np.random.Generator | None = None):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layer sizes.")
        self.layer_sizes = list(layer_sizes)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.thetas: list[np.ndarray] = []
        self._init_random_weights()

    def _init_random_weights(self) -> None:
        self.thetas = []
        for i in range(len(self.layer_sizes) - 1):
            n_out = self.layer_sizes[i + 1]
            n_in = self.layer_sizes[i] + 1
            # Small random init in [-1, 1] as suggested in assignment
            w = self.rng.uniform(-1.0, 1.0, size=(n_out, n_in))
            self.thetas.append(w.astype(np.float64))

    def set_thetas(self, thetas: list[np.ndarray]) -> None:
        if len(thetas) != len(self.layer_sizes) - 1:
            raise ValueError("Wrong number of weight matrices.")
        self.thetas = [np.array(t, dtype=np.float64, copy=True) for t in thetas]

    def forward(
        self, x: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # x: (n0, m). Returns activations a_list and pre-activations z_list (see HW4 notation).
        m = x.shape[1]
        a = np.vstack([np.ones((1, m), dtype=np.float64), x])
        a_list = [a]
        z_list: list[np.ndarray] = []

        L = len(self.layer_sizes) - 1
        for l in range(L):
            z = self.thetas[l] @ a_list[-1]
            z_list.append(z)
            if l == L - 1:
                a_out = sigmoid(z)
                a_list.append(a_out)
            else:
                a_hid = np.vstack([np.ones((1, m), dtype=np.float64), sigmoid(z)])
                a_list.append(a_hid)
        return a_list, z_list

    def cost(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam: float,
        *,
        include_reg: bool = True,
    ) -> float:
        # Mean logistic loss over examples; add (lam/2m)*||theta_no_bias||^2 if include_reg.
        a_list, _ = self.forward(x)
        h = a_list[-1]
        m = x.shape[1]
        eps = 1e-15
        h = np.clip(h, eps, 1.0 - eps)
        # Sum over outputs, mean over examples
        j = -(y * np.log(h) + (1.0 - y) * np.log(1.0 - h))
        j_mean = float(np.sum(j) / m)
        if not include_reg or lam == 0.0:
            return j_mean
        reg = 0.0
        for theta in self.thetas:
            reg += float(np.sum(theta[:, 1:] ** 2))
        reg *= lam / (2.0 * m)
        return j_mean + reg

    def backward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam: float,
    ) -> list[np.ndarray]:
        # Gradients dJ/dTheta per layer, batch mean, with L2 on non-bias weights.
        m = x.shape[1]
        a_list, z_list = self.forward(x)
        L = len(self.thetas)

        deltas: list[np.ndarray] = [None] * L  # type: ignore
        h = a_list[-1]
        deltas[L - 1] = h - y

        for l in range(L - 2, -1, -1):
            theta_tp1 = self.thetas[l + 1]
            d_up = deltas[l + 1]
            # hidden pre-activation z is z_list[l]; activations (no bias) in a_list[l+1][1:,:]
            a_sig = a_list[l + 1][1:, :]
            gprime = sigmoid_grad_from_a(a_sig)
            delta = (theta_tp1[:, 1:].T @ d_up) * gprime
            deltas[l] = delta

        grads: list[np.ndarray] = []
        for l in range(L):
            grad = (deltas[l] @ a_list[l].T) / m
            theta = self.thetas[l]
            reg = (lam / m) * theta
            reg[:, 0] = 0.0
            grads.append(grad + reg)
        return grads

    def gradient_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam: float,
        alpha: float,
    ) -> None:
        grads = self.backward(x, y, lam)
        for l in range(len(self.thetas)):
            self.thetas[l] -= alpha * grads[l]

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lam: float,
        alpha: float,
        max_epochs: int,
        batch_size: int | None = None,
        *,
        tol: float | None = None,
        verbose: bool = False,
    ) -> list[float]:
        # Mini-batch or full-batch GD; returns per-epoch mean training J (with regularizer).
        m = x.shape[1]
        bs = batch_size if batch_size is not None else m
        hist: list[float] = []
        prev_j = None
        for ep in range(max_epochs):
            perm = self.rng.permutation(m)
            for start in range(0, m, bs):
                idx = perm[start : start + bs]
                self.gradient_step(x[:, idx], y[:, idx], lam, alpha)
            j = self.cost(x, y, lam, include_reg=True)
            hist.append(j)
            if tol is not None and prev_j is not None:
                if abs(prev_j - j) / (abs(prev_j) + 1e-12) < tol:
                    if verbose:
                        print(f"Early stop epoch {ep} J={j:.6f}")
                    break
            prev_j = j
        return hist

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        a_list, _ = self.forward(x)
        return a_list[-1]

    def predict_labels(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        # Binary single-output: threshold probabilities to 0/1.
        p = self.predict_proba(x)
        return (p >= threshold).astype(np.int32)


def compute_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    # y_true, y_pred: flat 0/1.
    y_true = y_true.astype(np.int32).ravel()
    y_pred = y_pred.astype(np.int32).ravel()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return acc, f1
