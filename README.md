# DSLR — Datascience × Logistic Regression

42 project (subject: `dslr.pdf`): recreate the Sorting Hat with a one-vs-all logistic
regression trained by gradient descent. No built-in statistics are used: `count`, `mean`,
`std`, `min`, `max` and quantiles are hand-written in `dslr_common.py`.

## Setup
```sh
python3 -m venv .venv && .venv/bin/pip install pandas numpy matplotlib
# scikit-learn is only needed to cross-check accuracy, not by the scripts
```

## Usage
```sh
./describe.py datasets/dataset_train.csv            # describe table (features as columns)
./histogram.py datasets/dataset_train.csv [-o h.png]
./scatter_plot.py datasets/dataset_train.csv [-o s.png]
./pair_plot.py datasets/dataset_train.csv [-o p.png]
./logreg_train.py datasets/dataset_train.csv [batch|mini_batch|stochastic] [-o weights.pkl] [--seed 42]
./logreg_predict.py datasets/dataset_test.csv weights.pkl   # writes houses.csv
./compare.py [houses.csv] [datasets/dataset_truth.csv]      # accuracy vs the truth file
```
Plots open a window; with `MPLBACKEND=Agg` pass `-o` to save them headless.

## Answers to the subject's questions
- **Histogram — most homogeneous course across the four houses:** *Arithmancy*
  (spread of house means / overall std = 0.03), runner-up *Care of Magical Creatures* (0.07).
- **Scatter plot — the two similar features:** *Astronomy* and *Defense Against the Dark Arts*
  (Pearson r = −1.000: DADA = −Astronomy / 100).
- **Pair plot — features used for training:** Astronomy, Herbology, Divination, Muggle Studies,
  Ancient Runes, History of Magic, Transfiguration, Charms, Flying. Dropped: Arithmancy and
  Care of Magical Creatures (homogeneous), Defense Against the Dark Arts (duplicate of
  Astronomy), Potions (weakest separation of the rest, no held-out gain).
  The list lives in `dslr_common.FEATURES` and is what `logreg_train.py` uses.

## Model
- Features are standardised with the training mean/std; missing values are filled with the
  **training** means. Means, stds, feature list and weights are saved in `weights.pkl`.
- Modes: `batch` (default, mandatory: lr 0.5, 1000 iterations), `mini_batch` (bonus: batch 32,
  lr 0.1, 50 epochs), `stochastic` (bonus: lr 0.01, 10 epochs). Shuffling is seeded (`--seed`).

## Accuracy
| | batch | mini_batch | stochastic |
|---|---|---|---|
| dataset_test vs dataset_truth | 99.00% | 99.00% | 99.00% |
| 80/20 held-out, seeds 0–4 | 98.1 / 99.1 / 98.1 / 97.8 / 97.8 % | same | same |

The held-out numbers equal those of a fully converged scikit-learn logistic regression on the
same splits, so the ~2% residual error is the limit of a linear model on this data, not of
the optimiser.
