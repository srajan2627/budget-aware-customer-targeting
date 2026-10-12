# Reliable Customer Targeting Under Limited Marketing Campaign Budgets

CIS 631 Machine Learning research project — Fall 2026.

**Authors:** Manav Mendonca and Srajan Jain

## Overview

This project investigates how to choose customers and email treatments when a marketing campaign can contact only a limited share of its audience. The proposed framework combines customer segmentation, causal uplift modeling, and generative AI to support decisions about whom to contact, which treatment to use, and what message to generate.

A purchase-probability model ranks customers by their likelihood of buying. Uplift modeling estimates how an email changes that likelihood relative to no email. We plan to compare these approaches at **5%, 10%, and 20% contact limits**, with particular attention to the stability of customer rankings and treatment recommendations.

## Current status

**Steps 4 and 5 now have held-out evaluation and measured reliability. Step 6 includes customer segmentation and grounded GenAI drafts.** Both fixed baseline model families were refitted on development data and evaluated on the same 12,800 test customers. Paired confidence intervals, 100 random baseline rankings, and 20 training resamples per family are available. Clusters remain explanatory because the development ablation showed no targeting improvement. Message drafts have limited automated audits; independent message review and a new conversion experiment remain pending.

- [Initial Hillstrom exploration notebook](code/EDA/631_InitialHillstromExploration.ipynb)
- [Project progress presentation](docs/presentations/CIS631_Project_Progress_Report.pptx)
- [Progress report 5 with methods, results, and interpretation](docs/Progress_Reports/Progress_Report_5.docx)
- [Evaluation output guide](results/evaluation/README.md)

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
category vocabulary. Refit fixed candidates on all development data only
after freezing the comparison, then use them for the final test evaluation.

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

## Steps 4 through 6 Held-out decisions and reliability

With the existing development/test exports and project environment active:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/evaluation/run_evaluation.py
python code/genai/messages.py
python code/evaluation/plot_results.py
python -m unittest discover -s tests -v
```

The runner uses [the fixed evaluation protocol](configs/evaluation.json), writes
it before loading test outcomes, chooses clusters and runs the cluster ablation
on development data, and then evaluates both fixed Step 3 model families.
No winner or hyperparameter is selected using test outcomes. All results are
written to `results/evaluation`; rerunning replaces reproducible outputs.
The same fixed customers and source-row order are used for all strategies and
training refits. Do not tune future models against these now-observed test results.

At 5%, 10%, and 20%, logistic uplift estimates **6 [-12, 24]**, **24 [0, 48]**,
and **36 [9, 66]** additional conversions versus No Email. Tree uplift estimates
**9 [-9, 27]**, **12 [-6, 33]**, and **27 [3, 54]**. Brackets give pointwise
95% paired test-bootstrap intervals. Every contacting policy selected exactly
640, 1,280, or 2,560 customers: 5%, 10%, or 20% **across both emails combined**.
The positive-uplift eligibility rule can leave budget unused in other data.

Neither uplift policy has a clearly positive paired difference from the mean
random Men's Email baseline at any budget. Logistic response estimates 21,
33, and 51 additional conversions, with positive pointwise differences from
the mean random Men's policy, but sparse purchases and multiple comparisons
limit the conclusion. Predicting no purchase for everyone achieves 99.10%
test accuracy and is not a useful decision metric.

Selection is unstable across development bootstraps: logistic uplift shares
33.6%, 42.5%, and 45.3% of selected customers between refit pairs; tree uplift
shares 15.5%, 20.2%, and 32.8%. These are means across 190 dependent refit pairs
for each budget. Agreement on the email among shared selections is reported
separately from selection overlap. Test bootstrap intervals keep the policy
fixed; training-refit performance spreads are a distinct source of variation.

The two selected clusters primarily distinguish past Women's merchandise
purchases (sometimes both categories) from past Men's merchandise purchases
without Women's purchases. They describe purchase behavior, **not gender**.
Adding cluster membership to logistic uplift changes development estimates
by -3, 0, and -9 conversions at the three budgets, with no demonstrated benefit.
Cluster choice uses only development attributes; fresh clustering is fitted
inside each ablation fold. The global development choice of k means the
ablation is exploratory rather than fully nested selection.

The GenAI prototype contains nine Codex-authored draft messages for three
actual selected profiles, reusable prompts, a generator-callable interface,
fact ledgers, and content audits. All nine pass the limited checks. Independent
human review is pending, and these checks do not prove semantic grounding or
conversion improvement. Only the first stage corresponds to the evaluated
single-email policy; follow-up drafts are unsent sequence concepts requiring
a new experiment and total-contact budget.

- [Held-out policy value and confidence intervals](results/figures/heldout_policy_value.png)
- [Targeting reliability across training samples](results/figures/targeting_reliability.png)
- [Grounded drafts](results/evaluation/message_drafts.json)
- [Message review rubric and limitations](docs/research/message_review_rubric.md)

The observed Python 3.14 environment is pinned in
[requirements-evaluation.txt](requirements-evaluation.txt). A fresh-environment
installation has not been independently verified. Experiment metadata records
library versions and code/data/config hashes. The document builder in
`code/reporting/build_report5.py` uses Report 4 as its retained template and
requires `lxml` in a document-authoring environment; document layout is verified
using the bundled Documents renderer. No GitHub commit or push was performed.

## Remaining implementation

- [x] Record dependency versions, code/data hashes, and fixed experiment configurations.
- [ ] Verify installation in a fresh environment.
- [x] Build reusable data preparation with held-out splits and training-only preprocessing.
- [x] Compare cluster counts, fit training-only segments, and profile treatment responses with counts and uncertainty.
- [x] Train purchase-probability baselines with regularized logistic regression and a constrained tree.
- [x] Implement T-learner baselines for each email treatment versus control.
- [ ] Compare X-learner and ensemble uplift models after the baseline experiment.
- [x] Implement customer selection and treatment recommendations at 5%, 10%, and 20% contact limits.
- [x] Evaluate incremental conversions, actual contact fractions, and paired baseline differences on held-out randomized data.
- [x] Refit on different development samples and score fixed customers to measure selection and treatment stability.
- [x] Generate grounded campaign drafts and implement relevance, declared-fact, and treatment-alignment checks.
- [ ] Collect independent message-quality ratings and evaluate new copy/sequences in a new randomized study.
- [x] Add automated checks and export result tables, figures, and progress report 5.
- [ ] Add doubly robust evaluation as a separately validated extension.

## Scope and limitations

Budgets in the current research plan are audience contact percentages. Monetary campaign costs and profit-based optimization have not yet been specified.

Hillstrom supports evaluation of audience selection and the recorded email treatments. It cannot establish that newly generated messages or drip campaigns improve conversion, because those messages were not part of the recorded experiment. Any generated-message evaluation must remain separate from claims about measured campaign lift.
