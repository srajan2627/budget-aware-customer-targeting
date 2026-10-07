# Reliable Customer Targeting Under Limited Marketing Campaign Budgets

CIS 631 Machine Learning research project — Fall 2026.

**Authors:** Manav Mendonca and Srajan Jain

## Overview

This project investigates how to choose customers and email treatments when a marketing campaign can contact only a limited share of its audience. The proposed framework combines customer segmentation, causal uplift modeling, and generative AI to support decisions about whom to contact, which treatment to use, and what message to generate.

A purchase-probability model ranks customers by their likelihood of buying. Uplift modeling estimates how an email changes that likelihood relative to no email. We plan to compare these approaches at **5%, 10%, and 20% contact limits**, with particular attention to the stability of customer rankings and treatment recommendations.

## Current status

**EDA, data preparation, and the first baseline comparison are complete.** Regularized logistic regression and a constrained decision tree have been evaluated using the same five development folds. Response and T-learner policies are compared with no contact and fixed-email random targeting. Final test evaluation, uncertainty analysis, customer clustering, and generative campaigns remain pending.

- [Initial Hillstrom exploration notebook](code/EDA/631_InitialHillstromExploration.ipynb)
- [Project progress presentation](docs/presentations/CIS631_Project_Progress_Report.pptx)

## Research questions

1. What customer profiles emerge from historical behavior and purchasing channels, and how do observed campaign responses differ across those profiles?
2. How does uplift-based targeting compare with targeting based on purchase probability?
3. How do T-Learner, X-Learner, and ensemble approaches compare in incremental conversions under limited contact budgets?
4. How stable are rankings and treatment recommendations across training samples and model specifications?
5. Can generative AI turn selected profiles and treatment recommendations into relevant, factually grounded, treatment-aligned multi-step messages?

## Dataset

The notebook loads the **Hillstrom email marketing dataset** through `sklift.datasets.fetch_hillstrom(target_col="all")`.

- **64,000 customers** and **12 columns** in the initial combined dataframe.
- Three randomized treatment groups: `Mens E-Mail` (21,307), `Womens E-Mail` (21,387), and `No E-Mail` (21,306).
- Three post-campaign outcomes: website visit (`visit`), purchase (`conversion`), and spending (`spend`).
- Seven selected pre-campaign modeling features: `recency`, `history`, `mens`, `womens`, `newbie`, `zip_code`, and `channel`. The original data also includes `history_segment`, which is not included in this selected feature set.

The planned causal comparisons evaluate each email treatment separately against the shared no-email control group. The loader caches raw data locally, and the split script saves development/test CSVs in `datasets/processed`. These reproducible data exports are ignored by Git.

## Completed exploration

The notebook inspects data types, descriptive statistics, missing values, duplicate rows, treatment balance, outcome distributions, spending, customer characteristics, and correlations.

Its saved outputs report:

- **578 purchases**, an overall conversion rate of approximately **0.90%**.
- A website visit rate of **14.68%**.
- Purchase rates of approximately **1.25%** for Men's Email, **0.88%** for Women's Email, and **0.57%** for No Email.
- **Zero missing values** and **6,562 exact duplicate rows**. Duplicate rows are retained because there is no unique customer ID to establish that identical records represent the same customer.

These are descriptive results from the supplied notebook and presentation, not held-out model performance. The rare purchase outcome makes accuracy alone an inadequate basis for evaluating the planned models.

### Feature preparation

The notebook trims string fields, corrects `Surburban` to `Suburban`, and checks binary values. It includes median/mode imputation logic, although the inspected data has no missing values.

The notebook now splits customers before learned preprocessing and prepares:

- Raw development and test inputs with the seven pre-campaign features.
- Five shared development folds stratified by treatment × conversion.
- Development-fitted supervised and segmentation feature matrices, each with
  11 columns on this dataset, plus test features transformed with those same
  fitted preprocessors.

Treatment and post-campaign outcomes are excluded from feature matrices.
Supervised preprocessing imputes numeric medians, scales numeric features,
mode-imputes binary features, and mode-imputes/one-hot encodes categories.
Segmentation additionally uses log history and scales binary attributes.
Unknown categories are ignored during transformation.

The full-development matrices are for inspecting feature preparation. During
model selection, use raw inputs and fit a fresh preprocessing/model pipeline
within each training fold. Never cross-validate the already transformed
full-development matrices. Baseline model training is implemented below; final test evaluation is pending.

## Repository structure

```text
.
├── code/
│   ├── EDA/                 # Initial Hillstrom exploration notebook
│   ├── data/                # Reusable loading, splitting, preprocessing, and CV code
│   ├── models/              # Baseline response and T-learner models
│   ├── targeting/           # Budget-constrained targeting policies
│   └── evaluation/          # Initial development policy evaluation
├── configs/                 # Planned experiment settings and model parameters
├── datasets/
│   ├── raw/                 # Reserved for original dataset files
│   └── processed/           # Reserved for prepared data exports
├── docs/
│   ├── presentations/       # Project progress presentation
│   └── research/            # Reserved for research notes and references
├── notebooks/               # Reserved for additional exploratory notebooks
├── results/
│   ├── figures/             # Reserved for exported plots
│   └── tables/              # Reserved for experiment summaries
├── tests/                   # Planned automated checks
└── README.md
```

Some planned or reserved folders still contain `.gitkeep` placeholders. The existing EDA notebook lives in `code/EDA/`.

## Running the EDA notebook

The notebook uses Python, NumPy, pandas, Matplotlib, seaborn, scikit-learn, and scikit-uplift. A dependency lockfile and a verified reproducible environment are still pending.

From the repository root, a starting setup for macOS/Linux is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn scikit-uplift
python -m jupyterlab code/EDA/631_InitialHillstromExploration.ipynb
```

Select the environment's Python kernel and run the cells in order. The first cell also installs scikit-uplift and seaborn. Internet access is needed for package installation and the initial dataset download. These setup commands have not yet been validated in a clean environment.

## Development / final-test split

Set up and activate the project environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/data/split_data.py
```

In a new terminal, activate it again with `source .venv/bin/activate` from
the repository root. If already inside `code/data`, use
`source ../../.venv/bin/activate` and then `python split_data.py`.

`code/data/load_hillstrom.py` loads the data and applies the notebook's
deterministic cleaning. `code/data/split_data.py` reserves 20% for final testing
with `random_state=42`, stratified by treatment × conversion. It saves
`datasets/processed/development.csv` (51,200 customers) and `test.csv`
(12,800 customers). Rerunning replaces these exports with the same split for
the same source data. Use `--output-dir PATH` to choose another directory.

Both files retain all features, treatment, and outcomes. Read them with
`pd.read_csv(path, index_col="source_row")` to preserve the original row indices;
`source_row` is a tracking index, not a customer feature or a true customer ID.
Keep treatment and post-campaign outcomes out of customer feature inputs.

The EDA notebook contains the split immediately before Section 23. Sections
23–28 now use development-fitted preprocessing and only transform final test
features. Reserve test outcomes for final evaluation. Initial descriptive EDA
used the full dataset.

### Shared cross-validation and preprocessing

With the project environment active, run from the repository root:

```bash
python code/data/development_cv.py
python -m unittest discover -s tests -v
```

From `code/data`, the first command is `python development_cv.py`.
It reads only `development.csv` and saves `development_folds.csv`, mapping
original source rows to validation folds 1–5. Each fold has 40,960 training
customers and 10,240 validation customers. The same seed and input order
reproduce the assignments; all competing methods must reuse them. If loading
saved assignments after reordering data, align them by `source_row`.

`preprocessing.py` provides `build_preprocessor()` for supervised models and
`build_preprocessor(segmentation=True)` for clustering. Both return unfitted
transformers. For model selection, place a fresh transformer and the estimator
in a scikit-learn `Pipeline`, then pass the raw development features and
`cv=make_development_folds(development_df)` to cross-validation or grid search.
This lets each fold learn its own medians, modes, scaling statistics, and
category vocabulary. Refit the selected pipeline on all development data only
after model selection, then use it for the final test comparison.

The tests check missing and unseen values, feature exclusion, unchanged fitted
statistics after held-out transformation, reproducible fold membership, and
fold-specific fitting through a pipeline on synthetic data.

## Step 3 Baseline comparison

Activate the project environment and run from the repository root:

```bash
source .venv/bin/activate
python code/models/run_baselines.py
```

If your terminal is already in `code/data`, run
`python ../models/run_baselines.py` after activating `../../.venv/bin/activate`.
The runner reads only development data, reuses the saved fold assignments by
source row, and writes results to `results/baselines`. If fold assignments are
missing, it creates and saves five stratified folds. Existing output files are
replaced on rerun. `--output-dir PATH` selects another destination.

### Models and targeting rules

Each model family fits three separate purchase classifiers, one for each action.
Every classifier includes fresh imputation and encoding fitted on its training
arm only. Validation customers never contribute to their own model fit.

- **Logistic regression:** L2 regularization with `C=1`, the `lbfgs` solver, and
  at most 2,000 iterations. No class weighting or resampling is applied.
- **Decision tree:** maximum depth 3 and a minimum of 200 training customers
  per leaf. This is the single constrained tree comparison.

These are fixed starting configurations, not tuned models. No winner is selected
and no test data is opened. The implementation follows the scikit-learn
[logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
and [decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
interfaces.

The four strategy types are:

- **No contact:** assign No Email to everyone.
- **Random targeting:** choose a reproducible random subset and assign a fixed
  email type. Both Men's Email and Women's Email are reported, using the same
  selected customers; neither is chosen based on validation outcomes.
- **Purchase-probability targeting:** choose each customer's higher predicted
  email purchase probability, rank by that probability, and contact the top
  customers within the budget. This response baseline uses the email-arm models.
- **T-learner targeting:** subtract the predicted No Email purchase probability
  from each email prediction, choose the larger uplift, and rank customers by
  that uplift. Only positive predicted uplift is eligible for contact.

Response and uplift targeting share the same fitted arm models, so their
comparison isolates the difference in ranking rules. The two uplift scores are
`p_mens - p_no_email` and `p_womens - p_no_email`. They are estimated effects,
not observed customer-level counterfactual outcomes.

Budgets of 5%, 10%, and 20% apply to both email actions combined. Each validation
fold is treated as a separate campaign cohort; contact counts are rounded down.
Ties use reproducible random ordering, and each customer receives at most one
action. Seven policy variants across three budgets produce 21 comparisons.

### Outputs and how to interpret them

- `predictive_metrics.csv`: out-of-fold average precision, Brier score, and
  log loss for each observed treatment arm, alongside a training-arm prevalence
  baseline. These describe response prediction, not uplift accuracy.
- `policy_comparison.csv`: actual contact rates, email allocations, matched
  contacted purchases, and exploratory incremental-conversion estimates.
- `policy_by_fold.csv`: fold-level counts and incremental estimates.
- `oof_predictions.csv` and `policy_assignments.csv`: predictions and decisions
  indexed by source row, enabling later paired comparisons.
- `logistic_coefficients.csv` and `tree_action_0.txt` through `tree_action_2.txt`:
  interpretable coefficients and rules from models refitted on all development
  data. Numeric coefficients refer to standardized inputs; arm-specific
  coefficients are predictive associations, not individual causal effects.
- `logistic_development_model.joblib` and `tree_development_model.joblib`:
  refitted candidate models, not the fold models used to calculate validation
  results. Load with `code/models` on the Python import path so `baselines` can
  be imported; use the library versions recorded in `experiment.json`.
- `experiment.json`: settings, assumptions, software versions, and limitations.

Aggregate results and interpretation files can be committed. Model binaries and
per-customer exports are reproducible local artifacts ignored by Git.

The initial policy comparison uses inverse-probability weighting (IPW), assuming
balanced random assignment with probability 1/3 for each action. Assignment
probabilities are distinct from predicted purchase probabilities. For each
validation customer, the policy score is `1[observed action = recommended
 action] * conversion / assignment probability`. The no-contact score uses the
same calculation for No Email. Their paired difference estimates incremental
conversions; customers who are not contacted cancel exactly. Summing these
scores estimates incremental conversions across the development cohorts.
The approach follows the distinction between policy value and differences in
policy value described in the
[Stanford policy evaluation tutorial](https://bookdown.org/stanfordgsbsilab/ml-ci-tutorial/policy-evaluation-i---binary-treatment.html).

These are **development-only point estimates**, not final test results or proof
that a strategy is better. There are only 463 development purchases, random
selection uses one seeded ranking per fold, and confidence intervals, repeated
random baselines, and training stability have not yet been implemented. Step 4
should add paired uncertainty analysis and finalize the evaluation protocol
before opening test outcomes. Initial descriptive EDA used all records.
X-learner and ensemble models remain future comparisons; neither is assumed to
outperform the baselines.

## Remaining implementation

- [ ] Establish a reproducible environment, dependency versions, and experiment configurations.
- [x] Build reusable data preparation with held-out splits and training-only preprocessing.
- [ ] Select a cluster count, fit customer segments, and profile their treatment responses after clustering.
- [x] Train purchase-probability baselines with regularized logistic regression and a constrained tree.
- [x] Implement T-learner baselines for each email treatment versus control.
- [ ] Compare X-learner and ensemble uplift models after the baseline experiment.
- [x] Implement customer selection and treatment recommendations at 5%, 10%, and 20% contact limits.
- [ ] Evaluate estimated incremental conversions and compare targeting methods on held-out data.
- [ ] Repeat data splits and model configurations to measure ranking stability and treatment recommendation agreement.
- [ ] Generate multi-step campaign messages from selected customer profiles and model recommendations, then evaluate relevance, factual grounding, and treatment alignment.
- [ ] Add automated checks and export experiment tables, figures, and final research findings.

## Scope and limitations

Budgets in the current research plan are audience contact percentages. Monetary campaign costs and profit-based optimization have not yet been specified.

Hillstrom supports evaluation of audience selection and the recorded email treatments. It cannot establish that newly generated messages or drip campaigns improve conversion, because those messages were not part of the recorded experiment. Any generated-message evaluation must remain separate from claims about measured campaign lift.
