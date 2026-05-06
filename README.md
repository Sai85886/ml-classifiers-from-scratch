# CMPSCI 589 Final Project

This version is kept simple and student-style.

## Files

- `util.py`: common helpers for 10-fold CV, algorithm calls, metrics, plotting, and loading datasets.
- `digits.py`: runs experiments for digits dataset.
- `parkinsons.py`: runs experiments for parkinsons dataset.
- `rice.py`: runs experiments for rice dataset.
- `credit.py`: runs experiments for credit approval dataset.
- `main.py`: runs all datasets one by one and creates summary csv files.

## How to run

Run one dataset:

```bash
python3 digits.py
python3 parkinsons.py
python3 rice.py
python3 credit.py
```

Run everything:

```bash
python3 main.py
```

## Output

Results and plots are saved under `outputs/<dataset>/`.

Main summary files:
- `outputs/summary_best_configs.csv`
- `outputs/summary_report_table.csv`
