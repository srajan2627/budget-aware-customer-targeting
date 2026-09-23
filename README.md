# Reliable Customer Targeting Under Limited Marketing Campaign Budgets

CIS 631 Machine Learning research project — Fall 2026.

**Authors:** Manav Mendonca and Srajan Jain

## Overview

This project investigates how to choose customers and email treatments when a marketing campaign can contact only a limited share of its audience. The proposed framework combines customer segmentation, causal uplift modeling, and generative AI to support decisions about whom to contact, which treatment to use, and what message to generate.

A purchase-probability model ranks customers by their likelihood of buying. Uplift modeling estimates how an email changes that likelihood relative to no email. We plan to compare these approaches at **5%, 10%, and 20% contact limits**, with particular attention to the stability of customer rankings and treatment recommendations.

## Current status

**Initial EDA and feature preparation are complete. Model training and campaign evaluation are pending.** The repository contains an exploratory notebook with saved outputs and a project progress presentation. It does not yet contain trained predictive or uplift models, fitted customer segments, budget-constrained targeting policies, or a generative campaign pipeline.

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

The planned causal comparisons evaluate each email treatment separately against the shared no-email control group. Dataset files are not currently stored in `datasets/`; the notebook retrieves data through the loader.

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

It prepares these in-memory objects:

- `X_model`: **64,000 × 7** selected pre-campaign features.
- `X_cluster` / `X_cluster_df`: **64,000 × 11** features for future clustering, using log-transformed historical spending, standardized numeric fields, and one-hot encoded geography and channel.
- `X_encoded` / `X_encoded_df`: **64,000 × 11** features for future supervised or uplift models, with numeric scaling, binary passthrough, and categorical encoding.
- Separate outcome vectors (`y_visit`, `y_conversion`, `y_spend`) and the treatment assignment.

Treatment and post-campaign outcomes are excluded from customer feature matrices. **The current notebook fits preprocessing on the full dataset for exploration.** Before evaluating models, create train/validation/test splits and fit learned preprocessing only on training data to prevent information leakage. The notebook does not currently export the prepared matrices to disk.

## Repository structure

```text
.
├── code/
│   ├── EDA/                 # Initial Hillstrom exploration notebook
│   ├── data/                # Planned reusable loading and preprocessing code
│   ├── models/              # Planned segmentation, response, and uplift models
│   ├── targeting/           # Planned customer selection and treatment policies
│   └── evaluation/          # Planned budget and reliability evaluation
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

Folders marked planned or reserved currently contain `.gitkeep` placeholders. The existing EDA notebook lives in `code/EDA/`.

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

## Remaining implementation

- [ ] Establish a reproducible environment, dependency versions, and experiment configurations.
- [ ] Build reusable data preparation with held-out splits and training-only preprocessing.
- [ ] Select a cluster count, fit customer segments, and profile their treatment responses after clustering.
- [ ] Train a purchase-probability baseline (called the propensity baseline in the presentation).
- [ ] Implement T-Learner, X-Learner, and ensemble uplift models for each email treatment versus control.
- [ ] Implement customer selection and treatment recommendations at 5%, 10%, and 20% contact limits.
- [ ] Evaluate estimated incremental conversions and compare targeting methods on held-out data.
- [ ] Repeat data splits and model configurations to measure ranking stability and treatment recommendation agreement.
- [ ] Generate multi-step campaign messages from selected customer profiles and model recommendations, then evaluate relevance, factual grounding, and treatment alignment.
- [ ] Add automated checks and export experiment tables, figures, and final research findings.

## Scope and limitations

Budgets in the current research plan are audience contact percentages. Monetary campaign costs and profit-based optimization have not yet been specified.

Hillstrom supports evaluation of audience selection and the recorded email treatments. It cannot establish that newly generated messages or drip campaigns improve conversion, because those messages were not part of the recorded experiment. Any generated-message evaluation must remain separate from claims about measured campaign lift.
