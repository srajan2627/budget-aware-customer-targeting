# Reliable Customer Targeting Under Limited Marketing Campaign Budgets

**CIS 631 · Machine Learning · Fall 2026**  
Minnesota State University, Mankato  
**Authors:** Manav Mendonca and Srajan Jain

## Overview

When a campaign can contact only a small share of its customers, can a model consistently identify people who will benefit from contact?

We compare **response targeting** (who is likely to buy after an email) with **uplift targeting** (whose purchase probability the email is predicted to increase). Both email types share one **5%, 10% or 20% contact budget**. We measure incremental outcomes alongside consistency in **whom to contact** and **which email to send**.

Our focus combines randomized data and multiple treatments under a shared budget, separating audience stability from treatment-choice stability and evaluation uncertainty from retraining instability. Segmentation supports interpretation.

**Status: November 17, 2026.** The committed experiments use purchase conversion as the primary outcome. The final revision will make **visits primary**, retain conversion and spend as secondary outcomes, and address the feedback below by **December 1, 2026**. These revisions are planned, not completed results.

## Completed work

- **Data preparation:** Hillstrom exploration, a reproducible 51,200/12,800 development/evaluation split, five shared development folds, and training-only preprocessing.
- **Models and policies:** logistic and constrained-tree response models, their T-learners, random targeting and no-contact baselines. The “tree uplift” model uses separate decision trees for each treatment arm; it is not a causal forest.
- **Evaluation:** inverse-probability-weighted policy values, paired comparisons, per-treatment results and supplementary Qini/AUUC. The protocol uses 2,000 paired evaluation bootstraps, 100 random rankings and 20 training resamples per model family.
- **Reliability and validation:** audience overlap, email agreement, training-size comparisons, conditional detection limits and **109 known-truth simulation scenarios**. Simulations exposed poor sparse-outcome interval coverage and nonempty zero-event slices with zero-width percentile intervals.
- **Reproducibility:** frozen configurations, seeds, input/code hashes and saved outputs. [Recorded verification](results/report8/verification.json) reports **26 passing tests** and repeated numerical exports matching byte-for-byte. Clean-environment installation remains unverified.

### Current findings

The logistic response policy has the strongest held-out point estimates. Neither uplift policy clearly beats random Men's-email targeting in the historical paired comparisons. Sparse purchases and interval limitations prevent a definitive deployment recommendation.

Mean contact-list overlap across training resamples:

| Contact budget | Logistic T-learner | Tree T-learner |
| --- | ---: | ---: |
| 5% | 33.6% | 15.5% |
| 10% | 42.5% | 20.2% |
| 20% | 45.3% | 32.8% |

For logistic uplift, email agreement among customers selected by both refits is approximately 85–87%. These overlaps are descriptive averages over dependent refit pairs. The existing independent-random-list reference is **not a treatment-permutation null**.

Evidence: [policy values](results/evaluation/policy_summary.csv), [paired differences](results/evaluation/paired_comparisons.csv), [stability results](results/evaluation/reliability_summary.csv) and [simulation results](results/report8/simulation_summary.csv).

## Data and design

**Hillstrom:** 64,000 customers in a 2008 randomized email campaign with Men's Email, Women's Email and No Email arms. Recorded outcomes are visits, conversion and spend. The data contain 578 purchases (0.90%) and a 14.68% visit rate.

The seven pre-treatment features are `recency`, `history`, `mens`, `womens`, `newbie`, `zip_code` and `channel`. One-hot encoding the two categorical features produces **11 model columns**. We retain `history` and exclude `history_segment`. Merchandise-history indicators do not identify customer gender. Treatment and post-campaign outcomes are excluded from model inputs.

Each customer receives at most one recommended action. Contact limits apply across both email types, with equal contact costs assumed. Uplift policies may leave budget unused when predicted benefits are nonpositive. Monetary profit has not been measured.

## Final revision plan

All items below are planned work responding to the professor's feedback.

| Requirement | Planned change |
| --- | --- |
| Primary outcome and power | Refit for visits; retain conversion and spend. Report minimum detectable uplift for each outcome and budget, with assumptions and units. |
| Stability reference | Refit after treatment-label permutations for every reported stability measure; add score rank correlations and retain separate audience/email agreement. |
| Second dataset | Add a two-action replication using the corrected [Criteo dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/), with a frozen reproducible sample. This tests selection reliability, not Hillstrom's two-email choice. |
| Baselines and ablations | Add S-learner, class transformation, a DR-learner and causal forest. Measure ensemble-versus-single and outcome-choice ablations on stability, using methods appropriate to each outcome. |
| Interpretation and novelty | Frame response-versus-uplift results using predictive-versus-causal targeting literature; explicitly distinguish the study from Pacella et al. and Yang et al. |
| Evaluation and inference | Repeat main comparisons over at least five development/evaluation splits. Report Qini, AUUC and uplift at 5%, 10% and 20% with CIs, per-treatment results, defined combined-policy metrics and paired effect sizes. Select interval methods using simulated coverage and address multiple comparisons. |
| GenAI scope | Remove GenAI from the main paper. Existing prototype code and drafts remain historical artifacts with no measured campaign-impact claims. |
| Failure analysis and reporting | Document pre-treatment features and limited heterogeneity; examine disagreement groups and negative observed held-out uplift with uncertainty. Add internal, external and construct-validity threats, plus manipulation and exclusion concerns. |
| Reproducibility package | Verify a clean installation and include split assignments, simulation scenarios, versions, seeds, output files and hashes behind reported results. |

**Schedule:** Nov 18–20: visits and power; Nov 21–24: stability and models; Nov 25–27: Criteo, repeated splits and failure analysis; Nov 28–30: verification and writing; **Dec 1: final submission**. Existing five development folds do not replace the planned repeated development/evaluation splits.

## Reproduce the current experiments

The recorded environment used **Python 3.14.2**. Exact library versions are in [requirements-evaluation.txt](requirements-evaluation.txt); this is an environment snapshot, not a verified clean-install lockfile. The commands below reproduce the existing conversion-focused workflow, not the planned revisions.

From a macOS/Linux terminal with Python 3.14 installed:

```bash
git clone https://github.com/srajan2627/budget-aware-customer-targeting.git
cd budget-aware-customer-targeting
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-evaluation.txt

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python code/data/split_data.py
python code/data/development_cv.py
python code/models/run_baselines.py
python code/evaluation/run_evaluation.py
python code/evaluation/plot_results.py
python code/evaluation/run_report8.py
python code/evaluation/plot_report8.py
python -m unittest discover -s tests -v
```

The loader downloads Hillstrom through `scikit-uplift`; initial setup requires internet access. Rerunning replaces generated outputs. Raw data, split/fold exports, fitted models and per-customer predictions are currently Git-ignored and regenerated locally. Preserve original source-row indices when joining assignments.

To explore the [EDA notebook](code/EDA/631_InitialHillstromExploration.ipynb), additionally install `jupyterlab` and `seaborn`. Its package-install cells can change the environment. Historical message audits run separately with `python code/genai/messages.py`; they do not generate new messages through an API or measure lift.

## Repository guide

| Location | Contents |
| --- | --- |
| [code/](code/) | EDA, data preparation, models, targeting, evaluation, supporting segmentation and historical prototypes |
| [configs/](configs/) | Fixed experiment settings and seeds |
| [datasets/](datasets/) | Local raw-data cache and generated splits |
| [results/](results/) | Baseline/evaluation outputs, simulation results, figures and experiment metadata |
| [docs/](docs/) | Progress reports and research protocols |
| [tests/](tests/) | Automated implementation and evaluation checks |

Detailed methods and outputs: [research protocol](docs/research/report8_protocol.md), [evaluation guide](results/evaluation/README.md), [Report 8 output guide](results/report8/README.md) and [latest committed progress report](docs/Progress_Reports/Progress_Report_8.docx).

## Interpretation and limitations

Only 115 purchases occur in the current evaluation cohort. Prior full-data EDA and subsequent evaluation mean it is not historically untouched. Historical intervals are pointwise, exclude retraining uncertainty and can under-cover sparse outcomes. Simulation detection limits depend on their specified scenarios and do not establish power for every model comparison.

Hillstrom's age and limited features restrict generalization and detectable heterogeneity. Stable selection alone does not establish treatment benefit, and offline uplift does not establish deployment profit. Segmentation remains exploratory. The revised analysis will retain these limitations and report inconclusive findings.

## Related work for the revision

- [Fernández-Loría and Provost (2022)](https://www.jmlr.org/papers/v23/19-480.html): treatment-effect estimation versus outcome prediction.
- [Pacella, Papadia and Giliberti (2026)](https://www.mdpi.com/2076-3417/16/10/4918): stability-aware uplift policy selection.
- [Yang, Liu and Huang (2026)](https://arxiv.org/abs/2603.20775): uplift evaluation under structural biases.
- [Zhu et al. (ICML 2025)](https://proceedings.mlr.press/v267/zhu25s.html) and [Verbeken et al. (JMLR 2025)](https://www.jmlr.org/papers/v26/22-1455.html): uplift evaluation and limitations of ranking metrics.
