# Reliable Customer Targeting Under Limited Marketing Campaign Budgets

CIS 631 Machine Learning research project — Fall 2026.

**Authors:** Manav Mendonca and Srajan Jain

## Overview

This project investigates how to choose customers and email treatments when a marketing campaign can contact only a limited share of its audience. The implemented workflow combines customer segmentation, causal uplift modeling, and grounded message prototypes to support decisions about whom to contact, which treatment to use, and what message to draft.

A purchase-probability model ranks customers by their likelihood of buying. Uplift modeling estimates how an email changes that likelihood relative to no email. We have compared these approaches at **5%, 10%, and 20% contact limits**, measuring both estimated incremental conversions and the stability of customer selections and treatment recommendations.

## Current status

**Steps 4 and 5 now have held-out evaluation and measured reliability. Step 6 includes customer segmentation and grounded GenAI drafts.** Both fixed baseline model families were refitted on development data and evaluated on the same 12,800 test customers. Paired confidence intervals, 100 random baseline rankings, and 20 training resamples per family are available. Clusters remain explanatory because the development ablation showed no targeting improvement. Message drafts have limited automated audits; independent message review and a new conversion experiment remain pending.

Progress through **October 12, 2026**:

- **Step 1 — Experimental design:** purchase conversion is the primary outcome;
  visits and spending are secondary recorded outcomes. Actions are Men's Email,
  Women's Email, and No Email. Both email types share one contact budget, with
  equal contact costs assumed. Monetary costs and profit have not been measured.
- **Step 2 — Data preparation:** reproducible 80/20 split, five shared development
  folds, deterministic cleaning, and training-only preprocessing are implemented.
- **Step 3 — Baselines:** fixed logistic and constrained-tree response models and
  T-learners were compared out of fold against No Email and random targeting.
- **Step 4 — Budget evaluation:** held-out policy values, paired confidence
  intervals and baseline differences, actual contact fractions, email allocations,
  separate treatment-control results, and supplementary Qini/AUUC are exported.
- **Step 5 — Reliability:** repeated development samples score the same fixed
  evaluation customers; selection overlap, treatment agreement, and training-value
  variation are measured separately from test-sample uncertainty.
- **Step 6 — Segmentation and GenAI:** training-fitted clusters are profiled with
  response counts and uncertainty; a separate cluster-feature ablation and nine
  grounded draft-message audits are complete. New-message conversion lift is untested.
- **Validation and reporting:** all 18 automated tests passed, numeric result
  exports reproduced byte-for-byte, and progress report 5 was rendered and visually
  checked in the format of report 4. Code, results, and reports are tracked on GitHub.

- [Initial Hillstrom exploration notebook](code/EDA/631_InitialHillstromExploration.ipynb)
- [Project progress presentation](docs/presentations/CIS631_Project_Progress_Report.pptx)
- [Progress report 2 on design and data splitting](docs/Progress_Reports/Progress_Report_2.docx)
- [Progress report 4 on preprocessing and development baselines](docs/Progress_Reports/Progress_Report_4.docx)
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

The completed causal comparisons evaluate each email treatment separately against
the shared no-email control group and evaluate the combined recommendation policy
using inverse-probability weighting. Outcomes describe the two weeks following
the historical experiment. `mens` and `womens` indicate merchandise purchased in
the past year, not customer gender. The loader caches raw data locally, and the
split script saves development/test CSVs in `datasets/processed`. These
reproducible data exports are ignored by Git.

## Completed exploration

The notebook inspects data types, descriptive statistics, missing values, duplicate rows, treatment balance, outcome distributions, spending, customer characteristics, and correlations.

Its saved outputs report:

- **578 purchases**, an overall conversion rate of approximately **0.90%**.
- A website visit rate of **14.68%**.
- Purchase rates of approximately **1.25%** for Men's Email, **0.88%** for Women's Email, and **0.57%** for No Email.
- **Zero missing values** and **6,562 exact duplicate rows**. Duplicate rows are retained because there is no unique customer ID to establish that identical records represent the same customer.

These are descriptive results from the supplied notebook and presentation,
not held-out model performance. The rare purchase outcome makes accuracy alone
an inadequate basis for evaluating targeting models.

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
full-development matrices. Baseline training and held-out evaluation are
implemented in the runners documented below.

## Repository structure

```text
.
├── code/
│   ├── EDA/                 # Initial Hillstrom exploration notebook
│   ├── data/                # Reusable loading, splitting, preprocessing, and CV code
│   ├── models/              # Baseline response and T-learner models
│   ├── targeting/           # One combined contact budget across both emails
│   ├── evaluation/          # IPW, paired uncertainty, reliability, and figures
│   ├── segmentation/        # Training-fitted clusters and response profiles
│   ├── genai/               # Grounded prompts, generator interface, and audits
│   └── reporting/           # Report 5 builder using the retained Report 4 template
├── configs/                 # Fixed evaluation protocol
├── datasets/
│   ├── raw/                 # Ignored downloaded data cache
│   └── processed/           # Ignored development/test/fold exports
├── docs/
│   ├── Progress_Reports/    # Reports 2, 4, and 5
│   ├── presentations/       # Project progress presentation
│   └── research/            # Message review rubric and limitations
├── notebooks/               # Reserved for additional exploratory notebooks
├── results/
│   ├── baselines/           # Step 3 development comparisons and interpretation
│   ├── evaluation/          # Held-out, reliability, cluster, and message outputs
│   ├── figures/             # Exported policy-value and reliability plots
│   └── tables/              # Reserved for additional experiment summaries
├── tests/                   # 18 preprocessing, model, policy, and message checks
├── requirements.txt         # Unpinned core data/model dependencies
├── requirements-evaluation.txt  # Pinned observed evaluation environment
└── README.md
```

Some planned or reserved folders still contain `.gitkeep` placeholders. The existing EDA notebook lives in `code/EDA/`.

## Running the EDA notebook

The notebook uses Python, NumPy, pandas, Matplotlib, seaborn, scikit-learn, and
scikit-uplift. The evaluation environment's installed versions are pinned in
`requirements-evaluation.txt`; JupyterLab and seaborn are additional notebook
dependencies. This is an observed-environment snapshot, not a verified clean
installation or a full notebook dependency lockfile.

From the repository root, a starting setup for macOS/Linux is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-evaluation.txt
python -m pip install jupyterlab seaborn
python -m jupyterlab code/EDA/631_InitialHillstromExploration.ipynb
```

Select the environment's Python kernel and run the cells in order. The first cell also installs scikit-uplift and seaborn. Internet access is needed for package installation and the initial dataset download. These setup commands have not yet been validated in a clean environment.

The recorded evaluation used Python **3.14.2**. Use Python 3.14 to recreate that
environment; the compatibility of these pinned versions with other Python
versions has not been independently checked. Notebook package-install cells
may change an environment; preserve the recorded versions when reproducing
the model results.

## Development / final-test split

Set up and activate the project environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-evaluation.txt
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
features. The completed evaluation contains 463 development purchases and
115 test purchases: 53 under Men's Email, 38 under Women's Email, and 24 under
No Email. Test outcomes have now been evaluated; future tuning needs new
untouched validation evidence. Initial descriptive EDA used the full dataset.

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

These are fixed starting configurations, not tuned models. The Step 3 runner
selects no winner and opens no test data; the Step 4 runner later evaluates both
fixed families on the held-out cohort. The implementation follows the scikit-learn
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

These Step 3 outputs are **development-only point estimates**, not final test
results or proof that a strategy is better. On 51,200 development customers,
the leading point estimates were logistic response at 5% (21 additional
conversions), tree response at 10% (48), and tree uplift at 20% (87). Random
Men's Email remained competitive at 18, 42, and 81, respectively. Those
single-ranking development results did not select a model winner.

The Step 3 experiment has only 463 purchases and one seeded random ranking per
fold. Its original output files do not contain uncertainty intervals. The
completed Step 4–6 experiment below adds paired uncertainty, repeated random
baselines, and training stability, and reports held-out results separately.
Initial descriptive EDA used all records.
X-learner and ensemble models remain future comparisons; neither is assumed to
outperform the baselines.

## Steps 4 through 6 Held-out decisions and reliability

### Reproducing the completed analysis

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

`code/genai/messages.py` prepares prompts and audits the checked-in drafts; it
does not call a generation API or create fresh model-generated copy. The
provider-neutral `generate_draft(context, generator)` interface is available
for a separately configured generator. The plotting runner exports the two
figures from the saved CSV results without refitting models.

### Step 4 Held-out policy results

All values below estimate additional conversions relative to No Email over
12,800 test customers. Budget order is **5%, 10%, then 20%**. Brackets give
pointwise 95% paired bootstrap intervals, not individual treatment effects.

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

For example, at 20% the paired logistic uplift-minus-random-Men difference is
18.57 [-4.44, 43.62] conversions; the tree uplift difference is
9.57 [-13.29, 32.82]. Both include zero. The corresponding logistic response
difference is 33.57 [4.52, 64.47], with the same pointwise-inference limitations.

The other primary comparisons in that same budget order are:

- **Logistic response:** 21 [3, 39], 33 [9, 60], and 51 [18, 87].
- **Tree response:** 9 [-9, 27], 27 [6, 51], and 33 [3, 63].
- **Mean random Men's Email:** 3.99 [1.23, 6.81], 7.77 [2.49, 13.17], and
  17.43 [6.87, 27.99].
- **Mean random Women's Email:** 1.95 [-0.45, 4.47], 3.27 [-1.59, 7.95], and
  8.25 [-0.93, 17.73].
- **No contact:** zero incremental conversions and zero contacts by definition;
  this does not mean zero purchases without email.

The random estimates average **100** shared seeded selections. Their intervals
describe the averaged random policy; variability of one realized random
campaign is reported separately. We use **2,000 paired customer bootstrap
resamples**, keeping fitted policies and rankings fixed and reusing identical
draws across strategies. The intervals exclude training and reranking variation,
are approximate after the outcome-stratified split, and are unadjusted for
multiple comparisons. A deployment winner has not been chosen.

Primary outputs are [policy_summary.csv](results/evaluation/policy_summary.csv)
and [paired_comparisons.csv](results/evaluation/paired_comparisons.csv), where
each difference is **policy minus reference**. Decisions use one shared budget:
at 20%, logistic uplift sends 2,132 Men's Emails plus 428 Women's Emails;
tree uplift sends 2,398 plus 162. Each totals 2,560, not 2,560 per email type.

### Separate treatment-level results

We also rank each email's uplift against control separately on the 8,522
Men's/control and 8,539 Women's/control customers. At 20%, the standalone
logistic comparisons estimate 26 [6, 44] and 8 [-8, 26] conversions;
tree comparisons estimate 16 [0, 32] and 18 [0, 36]. These cohorts share control
customers and differ from the full campaign cohort, so their values and
contact counts must not be added to form a combined-budget policy.

The combined policy's email contributions are a different decomposition.
At 20%, logistic uplift's Men's and Women's components are 33 [6, 63] and
3 [0, 9], summing to its total of 36. Tree components are 24 [0, 48] and
3 [-6, 15], summing to 27. Both views are exported at every budget:

- [Standalone treatment-control uplift at budget](results/evaluation/treatment_uplift_at_budget.csv)
- [Email contributions to the combined policy](results/evaluation/policy_treatment_components.csv)
- [Supplementary normalized Qini and AUUC](results/evaluation/treatment_curve_metrics.csv)

### Step 5 Reliability on fixed evaluation customers

Selection is unstable across development bootstraps: logistic uplift shares
33.6%, 42.5%, and 45.3% of selected customers between refit pairs; tree uplift
shares 15.5%, 20.2%, and 32.8%. These are means across 190 dependent refit pairs
for each budget. Agreement on the email among shared selections is reported
separately from selection overlap. Test bootstrap intervals keep the policy
fixed; training-refit performance spreads are a distinct source of variation.

Training resampling occurs within action arms, without fixing purchase counts.
Both model families receive the same development samples, and ties use the
same policy seed. Email agreement among customers selected by both refits is
86.5%, 85.9%, and 85.0% for logistic uplift, versus 69.1%, 74.4%, and 75.8%
for tree uplift. Full action agreement, preferred-email agreement, and Jaccard
overlap are also exported; full action agreement includes the many No Email
assignments and can overstate targeting stability.

Across training refits, logistic uplift's value ranges are 0–21, 0–36, and 9–42
additional conversions. Tree ranges are 0–15, 0–18, and 3–33. These ranges
are training-sample variation on the fixed test cohort, **not confidence
intervals**. Test-bootstrap uncertainty and random-allocation variation remain
separate. Quantiles across the 190 dependent refit pairs are descriptive spreads.

- [Reliability summary](results/evaluation/reliability_summary.csv)
- [Training-refit values](results/evaluation/training_refit_values.csv)
- [Paired strategy differences across training refits](results/evaluation/training_paired_comparisons.csv)
- [Random-allocation variability](results/evaluation/random_allocation_summary.csv)

### Step 6 Customer segmentation

K-means compares **2, 3, 4, and 5 clusters**, using development attributes only.
The fixed selection rule maximizes sampled silhouette among candidates whose
smallest cluster contains at least 5% of customers and whose mean stability
adjusted Rand index (ARI) is at least 0.75 across five clustering resamples.
The selected two-cluster solution has silhouette **0.258**, mean ARI **0.784**,
and minimum ARI **0.640**, so its stability remains limited. Three and five
clusters were more stable but had lower silhouette; four failed the ARI gate.

The two selected clusters primarily distinguish past Women's merchandise
purchases (sometimes both categories) from past Men's merchandise purchases
without Women's purchases. They describe purchase behavior, **not gender**.
Adding cluster membership to logistic uplift changes development estimates
by -3, 0, and -9 conversions at the three budgets, with no demonstrated benefit.
Cluster choice uses only development attributes; fresh clustering is fitted
inside each ablation fold. The global development choice of k means the
ablation is exploratory rather than fully nested selection.

Development cluster sizes are **28,043 (54.8%)** and **23,157 (45.2%)**;
fixed-test sizes are 7,139 and 5,661. The first group's development medians are
5 months since purchase and $184.23 in past-year purchases; the second group's
medians are 6 months and $136.09. Within-cluster randomized responses include
arm sizes, purchase counts, and Wilson rate intervals. Cluster IPW effects
describe assigning an email to everyone in a cluster without a contact cap;
they are explanatory results, not budget-policy performance.

- [Cluster count and stability comparison](results/evaluation/cluster_comparison.csv)
- [Development and test cluster profiles](results/evaluation/cluster_profiles.csv)
- [Cluster response counts and rate uncertainty](results/evaluation/cluster_test_responses.csv)
- [Cluster treatment-effect uncertainty](results/evaluation/cluster_test_effects.csv)
- [Separate development cluster-feature ablation](results/evaluation/development_cluster_ablation.csv)

### Step 6 Grounded GenAI prototype

The GenAI prototype contains nine Codex-authored draft messages for three
actual selected profiles, reusable prompts, a generator-callable interface,
fact ledgers, and content audits. All nine pass the limited checks. Independent
human review is pending, and these checks do not prove semantic grounding or
conversion improvement. Only the first stage corresponds to the evaluated
single-email policy; follow-up drafts are unsent sequence concepts requiring
a new experiment and total-contact budget.

The three profiles come from the fixed logistic uplift selections at 20%,
chosen using attributes and recommendations without consulting their conversion
outcomes. Audits check exact declared facts, explicit merchandise alignment,
a relevant attribute, and a limited prohibited-claim list. They can miss
undeclared or paraphrased inventions; passing is not an independent relevance
rating. The review rubric covers relevance, factual grounding, treatment
alignment, clarity, and sequence suitability. No emails have been sent.

- [Held-out policy value and confidence intervals](results/figures/heldout_policy_value.png)
- [Targeting reliability across training samples](results/figures/targeting_reliability.png)
- [Grounded drafts](results/evaluation/message_drafts.json)
- [Source contexts for the drafts](results/evaluation/message_contexts.json)
- [Reusable generation prompts](results/evaluation/message_prompts.json)
- [Limited message audit results](results/evaluation/message_audit.csv)
- [Message review rubric and limitations](docs/research/message_review_rubric.md)

### Validation and reproducibility

All **18 tests** passed in the recorded project environment. They cover
training-only preprocessing, excluded outcomes, missing and unseen values,
shared folds aligned by source row, arm-specific fitting, combined budgets,
positive-uplift eligibility, known IPW effects in synthetic randomized data,
paired bootstrap draws, selection/treatment agreement, varying purchase counts
across training samples, outcome-free clustering, and incorrect message facts
or treatment alignment. Numerical evaluation CSVs reproduced byte-for-byte;
combined contact limits, additive email contributions, and code hashes were
also checked. No new model results are inferred from these implementation checks.

The observed Python 3.14 environment is pinned in
[requirements-evaluation.txt](requirements-evaluation.txt). A fresh-environment
installation has not been independently verified. Experiment metadata records
library versions and code/data/config hashes. The document builder in
`code/reporting/build_report5.py` uses Report 4 as its retained template and
requires `lxml` in a document-authoring environment; document layout is verified
using the bundled Documents renderer. The six-page report explains the methods,
changes, results, and limitations; its retained Report 4 template is unchanged.
Project progress through Report 5 has been committed and pushed to the
[GitHub repository](https://github.com/srajan2627/budget-aware-customer-targeting).

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

Only 115 purchases occur in the held-out cohort. Prior full-data descriptive
EDA means that the test set was not historically untouched. Bootstrap
intervals are approximate after treatment-by-conversion stratification, are
pointwise rather than adjusted for multiple comparisons, and do not combine
training, test, and future-cohort uncertainty. The two-cluster solution has
limited resample stability, and the cluster-feature ablation is exploratory.

Hillstrom supports evaluation of audience selection and the recorded email
treatments. It cannot establish that newly generated messages or drip campaigns
improve conversion, because those messages were not part of the recorded
experiment. Any generated-message evaluation must remain separate from claims
about measured campaign lift. Independent message-quality review, new randomized
copy/sequence experiments, X-learner and ensemble comparisons, doubly robust
evaluation, and fresh-environment verification remain future work.
