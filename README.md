# DSLR — Datascience × Logistic Regression

42 project (subject: `dslr.pdf`): recreate the Sorting Hat with a one-vs-all logistic
regression trained by gradient descent. No built-in statistics are used: `count`, `mean`,
`std`, `min`, `max` and quantiles are hand-written in `dslr_common.py`.

## Setup
Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).
```sh
uv sync              # creates .venv with pandas, numpy, matplotlib (+ scikit-learn, dev group)
uv sync --no-dev     # without scikit-learn, which only cross-checks accuracy
```

## Usage
```sh
uv run describe.py datasets/dataset_train.csv [--basic]  # describe table (features as columns); --basic = the 8 rows of the subject only
uv run histogram.py datasets/dataset_train.csv [-o h.png]
uv run scatter_plot.py datasets/dataset_train.csv [-o s.png]
uv run pair_plot.py datasets/dataset_train.csv [-o p.png]
uv run logreg_train.py datasets/dataset_train.csv [batch|mini_batch|stochastic] [-o weights.pkl] [--seed 42]
uv run logreg_predict.py datasets/dataset_test.csv weights.pkl   # writes houses.csv
uv run compare.py [houses.csv] [datasets/dataset_truth.csv]      # accuracy vs the truth file
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

## Study guide: the complete pipeline

The aim is supervised multiclass classification: each training row represents a student,
course scores are the input features, and `Hogwarts House` is the known label. The test
set has the same kind of scores but no usable house labels. The program learns parameters
from training data, saves them, then uses those same parameters to predict the test set.
The evaluator compares the predicted houses with hidden labels using accuracy.

### Files and responsibilities

- `dslr_common.py` contains CSV error handling, the feature list, house colours and the
  statistics used throughout the project. `load_csv` uses pandas for reading, while
  `values`, `count`, `mean`, `std`, `min_value`, `max_value` and `quantile` implement
  the numerical summaries explicitly. Missing floating-point values are excluded.
- `describe.py` computes the eight summary rows of the subject for each numeric course
  and, as a bonus, five more (`Var`, `Range`, `IQR`, `Skew`, `Kurt`, each explained in a short legend under the table; `--basic` hides both).
  It prints columns in blocks sized to the terminal and excludes the `Index` identifier.
- `histogram.py` overlays one score histogram per house for each course. It ranks
  courses by the sample standard deviation of their house means divided by the overall
  standard deviation. This is a heuristic for similarity of means, not a proof that
  all aspects of the distributions are identical.
- `scatter_plot.py` calculates Pearson correlation for each pair of numeric courses,
  using only rows where both values exist, and plots the pair with the largest absolute
  correlation. A correlation near -1 means a strong inverse linear relationship.
- `pair_plot.py` plots each feature against every other feature, with histograms on the
  diagonal and a colour for each house. It prints the feature selection used by training.
- `logreg_train.py` prepares the features, trains one classifier per house and saves a
  model. `logreg_predict.py` loads the model, prepares test rows and writes predictions.
- `compare.py` compares predictions to a truth file with matching indices. It is a local
  checking helper; the evaluator supplies its own accuracy check.

### Descriptive statistics

For non-missing values x₁…xₙ, count is n and the mean is Σxᵢ/n. The sample standard
 deviation is sqrt(Σ(xᵢ − mean)²/(n − 1)); it is undefined for fewer than two values.
Minimum and maximum are the smallest and largest values. For a quantile q, the code
sorts the observations, finds position (n − 1)q and linearly interpolates between the
neighbouring values. The 25%, 50% and 75% quantiles are the first quartile, median and
third quartile. These calculations do not call pandas `describe` or a statistics library.

The bonus rows are hand-written too. `Var` is the sample variance (the square of the sample
standard deviation), `Range` is max − min and `IQR` is the 75% quantile minus the 25% one.
`Skew` is the adjusted Fisher-Pearson skewness G1 = g1·√(n(n−1))/(n−2) with g1 = m₃/m₂^1.5,
and `Kurt` is the adjusted excess kurtosis G2 = (n−1)/((n−2)(n−3))·((n+1)·g2 + 6) with
g2 = m₄/m₂² − 3, where mₖ is the average of (x − mean)^k. Skew needs n ≥ 3 and Kurt n ≥ 4;
otherwise (or for a constant column) they print `nan`. All five match `pandas` to about 1e-13.

### Feature choices and data preparation

The histogram suggests Arithmancy and Care of Magical Creatures are weak discriminators
because the house means are similar. Astronomy and Defense Against the Dark Arts are
nearly perfectly inversely correlated, so retaining both adds redundant information.
Potions was dropped after the visual analysis and the reported held-out comparison. The
nine selected features are declared in `FEATURES`; the choice is specific to this data.
Correlation is not causation, and a small difference in means alone does not guarantee
that a feature is useless.

For each selected feature, training computes its mean and sample standard deviation
using training rows only. `prepare` replaces missing values by that training mean and
standardises each score as (x − mean)/std. A standard deviation of zero is replaced by
1. It then adds a column of ones for the intercept. The stored training means and
standard deviations must also be used at prediction time: recomputing them on test data
would change the model's inputs and leak information from the test set.

### Logistic regression and gradient descent

For a particular house, one-vs-all converts labels to 1 for that house and 0 for all
others. Given a standardised row x (including the intercept), the model computes
z = x·θ and applies the sigmoid σ(z) = 1/(1 + exp(−z)). Its result is a score between
0 and 1. Four independent binary models are trained, and prediction selects the house
with the highest score. The four scores are not a softmax distribution and need not sum
 to one.

The objective for a binary classifier is average cross-entropy:
`J(θ) = −(1/m) Σ[y log(h) + (1−y) log(1−h)]`, where `h = σ(Xθ)`.
The gradient is `X.T @ (h − y) / m`; the update is `θ ← θ − learning_rate × gradient`.
`sigmoid`, `cost`, `gradient`, and the optimiser functions implement these operations
using NumPy array arithmetic, rather than a machine-learning fitting function. The cost
uses a small epsilon inside logarithms to avoid taking log(0).

In batch mode every update uses all training rows. Mini-batch mode shuffles rows and
updates on groups of 32; stochastic mode uses groups of one. An epoch visits all rows,
whereas an iteration in batch mode is one full-data update. The learning rates and
iteration counts are fixed in `HYPERPARAMS`, and the random seed controls shuffling.
Training prints the final cost of each binary model; cost is not accuracy.

### Saved model and prediction

The pickle contains the optimiser mode, house names, feature names, training means and
standard deviations, and a weight vector for each house. Prediction validates that the
required feature columns exist, uses the saved preprocessing statistics and weights,
then writes `Index,Hogwarts House` to `houses.csv`. Only load pickle files from trusted
sources: deserialising an untrusted pickle can execute code.

Accuracy is the number of correct predictions divided by the total number of predictions.
The 99% result above was measured on the supplied test/truth pair; it is not a guarantee
for other datasets. Held-out validation should split training data before computing
preprocessing statistics or choosing hyperparameters, to avoid leakage. During the
defence, be ready to explain the feature decisions, formulas, the difference between
training and prediction, and how you checked the output format and accuracy.

### Checks and limitations

The project pins Python 3.14 in `.python-version`; `uv sync` may need to download that
interpreter and the dependencies. The programs expect the named course columns and a
training house column. Empty or malformed input is not guaranteed to be handled for
all possible datasets. Visualization scripts call `plt.show()`; use a graphical session
or `MPLBACKEND=Agg` and `-o` for a headless run. The one-vs-all linear model and the
feature choices are tailored to the supplied Hogwarts data.
