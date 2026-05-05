"""
Demonstrate backprop correctness against backprop_example1.txt and backprop_example2.txt.
Run: python3 verify_backprop.py
"""

from __future__ import annotations

import os
import re

import numpy as np

from neural_network import NeuralNetwork, sigmoid, sigmoid_grad_from_a


def _fmt(v: float) -> str:
    return f"{v:8.5f}"


def _print_vec(name: str, arr: np.ndarray) -> None:
    s = "  ".join(_fmt(float(x)) for x in np.asarray(arr).ravel())
    print(f"\t{name}: [{s}]")


def run_reference_demo(
    lam: float,
    layer_sizes: list[int],
    thetas: list[np.ndarray],
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    title: str,
) -> None:
    """Print forward + backward trace matching homework reference style."""
    print(title)
    print(f"Regularization parameter lambda={lam:.3f}")
    print()
    print(
        "Initializing the network with the following structure "
        f"(number of neurons per layer): {layer_sizes}"
    )
    print()
    for i, th in enumerate(thetas, start=1):
        print(f"Initial Theta{i} (the weights of each neuron, including the bias weight, are stored in the rows):")
        for row in th:
            print("\t" + "  ".join(_fmt(float(x)) for x in row) + "  ")
        print()
    print("Training set")
    for t, (xv, yv) in enumerate(zip(xs, ys), start=1):
        print(f"\tTraining instance {t}")
        _print_vec("x", xv)
        _print_vec("y", yv)
    print()
    print("--------------------------------------------")
    print("Computing the error/cost, J, of the network")

    net = NeuralNetwork(layer_sizes)
    net.set_thetas(thetas)

    m = len(xs)
    X = np.column_stack(xs)
    Y = np.column_stack(ys)

    inst_costs: list[float] = []
    for t in range(m):
        print(f"\tProcessing training instance {t + 1}")
        x1 = X[:, t : t + 1]
        print(f"\tForward propagating the input [{', '.join(_fmt(float(v)) for v in x1.ravel())}]")
        a_list, z_list = net.forward(x1)
        print(f"\t\ta1: [{'   '.join(_fmt(float(v)) for v in a_list[0].ravel())}]")
        L = len(layer_sizes) - 1
        for l in range(1, L):
            print(f"\t\tz{l + 1}: [{'   '.join(_fmt(float(v)) for v in z_list[l - 1].ravel())}]")
            print(f"\t\t\ta{l + 1}: [{'   '.join(_fmt(float(v)) for v in a_list[l].ravel())}]")
        print(f"\t\tz{L + 1}: [{'   '.join(_fmt(float(v)) for v in z_list[L - 1].ravel())}]")
        h = a_list[-1].ravel()
        print(f"\t\ta{L + 1}: [{'   '.join(_fmt(float(v)) for v in h)}]")
        print(f"\t\tf(x): [{'   '.join(_fmt(float(v)) for v in h)}]")
        y1 = Y[:, t : t + 1].ravel()
        eps = 1e-15
        hh = np.clip(h, eps, 1.0 - eps)
        if len(h) == 1:
            jt = float(-(y1[0] * np.log(hh[0]) + (1.0 - y1[0]) * np.log(1.0 - hh[0])))
        else:
            jt = float(-np.sum(y1 * np.log(hh) + (1.0 - y1) * np.log(1.0 - hh)))
        inst_costs.append(jt)
        print(f"\tPredicted output for instance {t + 1}: [{'   '.join(_fmt(float(v)) for v in h)}]")
        print(f"\tExpected output for instance {t + 1}: [{'   '.join(_fmt(float(v)) for v in y1)}]")
        print(f"\tCost, J, associated with instance {t + 1}: {jt:.3f}")
        print()

    j_full = net.cost(X, Y, lam, include_reg=True)
    print(f"Final (regularized) cost, J, based on the complete training set: {j_full:.5f}")
    print()
    print()
    print("--------------------------------------------")
    print("Running backpropagation")

    for t in range(m):
        print(f"\tComputing gradients based on training instance {t + 1}")
        x1 = X[:, t : t + 1]
        y1 = Y[:, t : t + 1]
        a_list, z_list = net.forward(x1)
        L = len(net.thetas)
        # manual deltas matching NN formulas
        d = a_list[-1] - y1
        _print_vec(f"delta{L + 1}", d)
        deltas = [None] * L  # type: ignore
        deltas[L - 1] = d
        for l in range(L - 2, -1, -1):
            a_sig = a_list[l + 1][1:, :]
            gprime = sigmoid_grad_from_a(a_sig)
            deltas[l] = (net.thetas[l + 1][:, 1:].T @ deltas[l + 1]) * gprime
            _print_vec(f"delta{l + 2}", deltas[l])
        print()
        for l in range(L):
            grad_inst = deltas[l] @ a_list[l].T
            print(f"\t\tGradients of Theta{l + 1} based on training instance {t + 1}:")
            for row in grad_inst:
                print("\t\t\t" + "  ".join(_fmt(float(x)) for x in row.ravel()) + "  ")
            print()

    print("\tThe entire training set has been processed. Computing the average (regularized) gradients:")
    full_grads = net.backward(X, Y, lam)
    for l, g in enumerate(full_grads, start=1):
        print(f"\t\tFinal regularized gradients of Theta{l}:")
        for row in g:
            print("\t\t\t" + "  ".join(_fmt(float(x)) for x in row.ravel()) + "  ")
        print()


def parse_reference_file(path: str) -> tuple[float, list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    text = open(path, "r", encoding="utf-8").read()
    lam_m = re.search(r"lambda=([0-9.]+)", text)
    if not lam_m:
        raise ValueError("lambda not found")
    lam = float(lam_m.group(1))
    struct_m = re.search(r"\[([0-9\s]+)\]", text)
    if not struct_m:
        raise ValueError("structure not found")
    layer_sizes = [int(x) for x in struct_m.group(1).split()]
    # Parse matrices Theta1, Theta2, ...
    thetas: list[list[list[float]]] = []
    current: list[list[float]] | None = None
    for line in text.splitlines():
        if re.match(r"Initial Theta\d", line.strip()):
            if current is not None:
                thetas.append(current)
            current = []
            continue
        if line.strip().startswith("Training set"):
            if current is not None:
                thetas.append(current)
            current = None
            break
        if current is not None and line.strip() and line.strip()[0].isdigit() or (
            current is not None and line.strip().startswith("\t") and re.search(r"[0-9]", line)
        ):
            nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)]
            if len(nums) > 0:
                current.append(nums)
    if current is not None and len(current) > 0:
        thetas.append(current)
    theta_arrays = [np.array(t, dtype=np.float64) for t in thetas]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    in_train = False
    for line in text.splitlines():
        if "Training set" in line:
            in_train = True
            continue
        if in_train and "x:" in line:
            nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)]
            xs.append(np.array(nums, dtype=np.float64).reshape(-1, 1))
        if in_train and "y:" in line:
            nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)]
            ys.append(np.array(nums, dtype=np.float64).reshape(-1, 1))
        if in_train and line.strip().startswith("---"):
            break

    return lam, layer_sizes, theta_arrays, xs, ys


def verify_from_files(base_dir: str | None = None) -> None:
    base = base_dir or os.path.dirname(os.path.abspath(__file__))
    for fname in ("backprop_example1.txt", "backprop_example2.txt"):
        path = os.path.join(base, fname)
        lam, layer_sizes, thetas, xs, ys = parse_reference_file(path)
        run_reference_demo(lam, layer_sizes, thetas, xs, ys, f"===== {fname} =====\n")


def verify_correctness() -> None:
    """Entry point for graders: reproduces reference traces."""
    verify_from_files()


if __name__ == "__main__":
    verify_correctness()
