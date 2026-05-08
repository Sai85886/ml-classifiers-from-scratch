# Run digits, parkinsons, rice, credit in parallel; write summary CSVs.
# Stdout from threads may interleave; run a single *.py for clean logs.
import os
import threading

import pandas as pd

import credit
import digits
import parkinsons
import rice


def make_summary():
    root = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(root, "outputs")
    result_files = [
        ("digits", os.path.join(output_root, "digits", "digits_cv_results.csv")),
        ("parkinsons", os.path.join(output_root, "parkinsons", "parkinsons_cv_results.csv")),
        ("rice", os.path.join(output_root, "rice", "rice_cv_results.csv")),
        ("credit_approval", os.path.join(output_root, "credit_approval", "credit_approval_cv_results.csv")),
    ]

    rows = []
    for dataset_name, path in result_files:
        if not os.path.exists(path):
            print(f"[main] Summary: skip {dataset_name} (missing {path})", flush=True)
            continue
        df = pd.read_csv(path)
        best = df.sort_values(["mean_f1", "mean_acc"], ascending=False).iloc[0].to_dict()
        best["dataset"] = dataset_name
        rows.append(best)
        print(
            f"[main] Summary: {dataset_name} best row → model={best.get('model')} "
            f"acc={best.get('mean_acc'):.4f} F1={best.get('mean_f1'):.4f}",
            flush=True,
        )

    if not rows:
        print("[main] No results found for summary.", flush=True)
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_root, "summary_best_configs.csv"), index=False)
    summary[["dataset", "model", "mean_acc", "mean_f1"]].to_csv(
        os.path.join(output_root, "summary_report_table.csv"),
        index=False,
    )
    print("[main] Wrote outputs/summary_best_configs.csv and summary_report_table.csv", flush=True)


def _run_with_banner(module, name: str):
    print(f"[main] Starting thread {name!r}...", flush=True)
    try:
        module.run()
    finally:
        print(f"[main] Finished thread {name!r}.", flush=True)


def run_all():
    print("[main] Launching digits, parkinsons, rice, credit in parallel...", flush=True)
    threads = [
        threading.Thread(target=_run_with_banner, args=(digits, "digits"), name="digits"),
        threading.Thread(target=_run_with_banner, args=(parkinsons, "parkinsons"), name="parkinsons"),
        threading.Thread(target=_run_with_banner, args=(rice, "rice"), name="rice"),
        threading.Thread(target=_run_with_banner, args=(credit, "credit"), name="credit"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("\n[main] All dataset threads joined. Building summary...", flush=True)
    make_summary()
    print("[main] All done.", flush=True)


if __name__ == "__main__":
    run_all()
