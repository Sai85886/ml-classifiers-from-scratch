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
            continue
        df = pd.read_csv(path)
        best = df.sort_values(["mean_f1", "mean_acc"], ascending=False).iloc[0].to_dict()
        best["dataset"] = dataset_name
        rows.append(best)

    if not rows:
        print("No results found for summary.")
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_root, "summary_best_configs.csv"), index=False)
    summary[["dataset", "model", "mean_acc", "mean_f1"]].to_csv(
        os.path.join(output_root, "summary_report_table.csv"),
        index=False,
    )
    print("Saved summary files in outputs/")


def run_all():
    threads = [
        threading.Thread(target=digits.run, name="digits"),
        threading.Thread(target=parkinsons.run, name="parkinsons"),
        threading.Thread(target=rice.run, name="rice"),
        threading.Thread(target=credit.run, name="credit"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    make_summary()


if __name__ == "__main__":
    run_all()
