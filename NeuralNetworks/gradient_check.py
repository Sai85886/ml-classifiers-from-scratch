# QE.3: finite-difference vs analytic gradients (backprop_example*.txt).
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from neural_network import NeuralNetwork
from verify_backprop import parse_reference_file

BASE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class GradCheckResult:
    example: str
    eps: float
    max_abs_diff: float
    mean_abs_diff: float
    numeric_grads: list[np.ndarray]
    analytic_grads: list[np.ndarray]


def _numerical_grads(
    layer_sizes: list[int],
    thetas: list[np.ndarray],
    X: np.ndarray,
    Y: np.ndarray,
    lam: float,
    eps: float,
) -> list[np.ndarray]:
    net = NeuralNetwork(layer_sizes)
    net.set_thetas(thetas)

    grads: list[np.ndarray] = []
    for li, theta in enumerate(thetas):
        g = np.zeros_like(theta, dtype=np.float64)
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                t_plus = [np.array(t, copy=True) for t in thetas]
                t_minus = [np.array(t, copy=True) for t in thetas]
                t_plus[li][i, j] += eps
                t_minus[li][i, j] -= eps

                net.set_thetas(t_plus)
                j_plus = net.cost(X, Y, lam, include_reg=True)
                net.set_thetas(t_minus)
                j_minus = net.cost(X, Y, lam, include_reg=True)

                g[i, j] = (j_plus - j_minus) / (2.0 * eps)
        grads.append(g)
    return grads


def run_grad_check(example_file: str, eps: float) -> GradCheckResult:
    path = os.path.join(BASE, example_file)
    lam, layer_sizes, thetas, xs, ys = parse_reference_file(path)
    X = np.column_stack(xs)
    Y = np.column_stack(ys)

    net = NeuralNetwork(layer_sizes)
    net.set_thetas(thetas)
    analytic = net.backward(X, Y, lam)
    numeric = _numerical_grads(layer_sizes, thetas, X, Y, lam, eps)

    diffs = [np.abs(a - n) for a, n in zip(analytic, numeric)]
    max_abs = max(float(np.max(d)) for d in diffs)
    mean_abs = float(np.mean(np.concatenate([d.ravel() for d in diffs])))
    return GradCheckResult(example_file, eps, max_abs, mean_abs, numeric, analytic)


def _fmt_matrix(mat: np.ndarray) -> list[list[float]]:
    return [[float(f"{x:.8f}") for x in row] for row in mat]


def main() -> None:
    eps_values = [0.1, 0.000001]
    examples = ["backprop_example1.txt", "backprop_example2.txt"]
    out_lines: list[str] = []
    json_out: dict[str, dict[str, object]] = {}

    for ex in examples:
        json_out[ex] = {}
        for eps in eps_values:
            r = run_grad_check(ex, eps)
            key = f"eps={eps}"
            json_out[ex][key] = {
                "max_abs_diff": r.max_abs_diff,
                "mean_abs_diff": r.mean_abs_diff,
                "numeric_grads": [_fmt_matrix(g) for g in r.numeric_grads],
                "analytic_grads": [_fmt_matrix(g) for g in r.analytic_grads],
            }

            out_lines.append(f"===== {ex} | epsilon={eps} =====")
            out_lines.append(
                f"max_abs_diff={r.max_abs_diff:.10f} mean_abs_diff={r.mean_abs_diff:.10f}"
            )
            for i, g in enumerate(r.numeric_grads, start=1):
                out_lines.append(f"Numeric gradients Theta{i}:")
                for row in g:
                    out_lines.append(
                        "  " + "  ".join(f"{float(v): .8f}" for v in row.tolist())
                    )
            out_lines.append("")

    txt_path = os.path.join(BASE, "results", "gradient_check_output.txt")
    json_path = os.path.join(BASE, "results", "gradient_check_results.json")
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2)

    print(f"Saved {txt_path}")
    print(f"Saved {json_path}")
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
