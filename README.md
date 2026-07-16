# Machine Learning From Scratch — COMPSCI 589 Final Project

End-to-end comparison of **custom-built** machine learning classifiers on four real-world datasets. Every learner is implemented in Python (no scikit-learn models for training or prediction). The pipeline runs **stratified 10-fold cross-validation**, sweeps hyperparameters, exports CSV results, and generates publication-style plots for a full written report.

**Course:** COMPSCI 589 — Machine Learning (UMass Amherst, Spring 2026)

---

## Highlights

- **Algorithms from scratch:** k-NN, distance-weighted k-NN, decision trees, random forests, feedforward neural networks (NumPy backprop), plus naive Bayes utilities from earlier coursework
- **Four benchmark datasets:** handwritten digits, Parkinson’s voice detection, rice morphology, credit approval (mixed numeric + categorical)
- **Rigorous evaluation:** accuracy and F1 (macro for multiclass), ≥6 hyperparameter settings per algorithm family, best-config summary tables
- **Reproducible experiments:** one command runs all main datasets; outputs land in `outputs/` as CSV + PNG
- **Extra credit:** OpenML Zoo benchmark, distance-weighted k-NN variant (EC4), and extended algorithm coverage on every required dataset

### Best cross-validated performance (selected configs)

| Dataset | Best model | Accuracy | F1 |
|---------|------------|----------|-----|
| Digits (10-class) | k-NN | 0.981 | 0.981 |
| Parkinson’s (binary) | k-NN | 0.949 | 0.965 |
| Rice (binary) | Neural network | 0.930 | 0.939 |
| Credit (binary) | Random forest | 0.874 | 0.861 |

Full tables, motivation, and figure analysis: see [`final_project_report.tex`](final_project_report.tex) (compile with `pdflatex`).

---

## Algorithms & implementations

| Algorithm | Location | Notes |
|-----------|----------|--------|
| k-NN | [`KNN/knn.py`](KNN/knn.py) | Euclidean distance, majority vote |
| Weighted k-NN | [`KNN/knn.py`](KNN/knn.py) + [`extra_credit4.py`](extra_credit4.py) | Inverse-distance² voting (EC4) |
| Decision tree | [`RandomForests/decision_tree.py`](RandomForests/decision_tree.py) | Gini / information gain, depth limits |
| Random forest | [`RandomForests/random_forest.py`](RandomForests/random_forest.py) | Bagging + feature subsampling |
| Neural network | [`NeuralNetworks/neural_network.py`](NeuralNetworks/neural_network.py) | Sigmoid MLP, L2, mini-batch SGD |
| Naive Bayes | [`NaiveBayes/`](NaiveBayes/) | Course homework utilities |

Shared CV, metrics, plotting, and data loaders: [`util.py`](util.py)

**Allowed external libraries:** NumPy, Pandas, Matplotlib; scikit-learn is used **only** to load bundled/OpenML datasets (`load_digits`, `fetch_openml`), not for learning.

---

## Datasets

| Script | Data source | Task |
|--------|-------------|------|
| [`digits.py`](digits.py) | sklearn `load_digits` | 10-class digit recognition (8×8) |
| [`parkinsons.py`](parkinsons.py) | [`parkinsons.csv`](parkinsons.csv) | Binary Parkinson’s diagnosis |
| [`rice.py`](rice.py) | [`rice.csv`](rice.csv) | Binary rice variety |
| [`credit.py`](credit.py) | [`credit_approval.csv`](credit_approval.csv) | Binary credit approval (one-hot categoricals) |
| [`extra_credit2.py`](extra_credit2.py) | OpenML **Zoo** | 7-class animal types (mixed attributes) |

---

## Quick start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/sai85886/ml-classifiers-from-scratch.git
cd ml-classifiers-from-scratch

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run experiments

```bash
# All four main datasets in parallel + summary CSVs
python main.py

# Or one dataset at a time (cleaner logs; NN steps can take several minutes)
python digits.py
python parkinsons.py
python rice.py
python credit.py

# Extra credit
python extra_credit2.py    # Zoo (downloads once via OpenML)
python extra_credit4.py    # Weighted k-NN on all four datasets
```

### 3. Outputs

Results are written under [`outputs/`](outputs/):

- `*_cv_results.csv` — every hyperparameter setting with mean ± std accuracy/F1  
- `*_accuracy.png`, `*_f1.png` — hyperparameter sweeps  
- `nn_learning_curve.png` — test cost **J** vs. training size (where NN is used)  
- `summary_best_configs.csv` — best row per dataset after `main.py`

---

## Project structure

```
├── digits.py / parkinsons.py / rice.py / credit.py   # Main experiment drivers
├── main.py                                           # Parallel runner + summary
├── extra_credit2.py / extra_credit4.py               # Extra credit experiments
├── util.py                                           # CV, metrics, plots, loaders
├── KNN/                    RandomForests/            NeuralNetworks/    NaiveBayes/
├── *.csv                   # Tabular datasets (except digits / zoo)
├── outputs/                # Generated metrics and figures
└── final_project_report.tex
```

---

## Team & contributions

| Member | Primary implementations |
|--------|-------------------------|
| **Sudheer Bulusu** | k-NN, Naive Bayes |
| **Sai Teja Boga** | Decision tree, random forest |
| **Rayirth Pakala** | Neural network |

---

## License & academic use

This repository was submitted as coursework for COMPSCI 589 at UMass Amherst. The original course handout and team org fork live at [umass-ml-forge/ml-project](https://github.com/umass-ml-forge/ml-project). Please cite or contact the authors before reusing code in other graded assignments.

---

## Contact

**Sai Teja Boga** — [GitHub @sai85886](https://github.com/sai85886)
