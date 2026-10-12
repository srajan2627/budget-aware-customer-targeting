# Evaluation results and interpretation

This directory contains the fixed Step 4–6 experiment, evaluated on 12,800
Hillstrom test customers with 115 purchases. Report 5 describes the reasoning
and conclusions. The model configurations are unchanged from Step 3; no
winner or hyperparameter was selected using these test outcomes.

## Budget decisions

For each customer, the uplift policy chooses the higher of predicted Men's
Email minus No Email and predicted Women's Email minus No Email. It ranks by
that value and selects only strictly positive values up to floor(n * budget).
Each customer receives one action. Men's contacts plus Women's contacts must
never exceed 640, 1,280, or 2,560 at the three test budgets. The observed
selected fractions are exactly 5%, 10%, and 20% for every contacting policy;
No contact selects zero.

Response targeting shares the same three arm models but ranks by the higher
email purchase probability without subtracting No Email. Seeded tie breaking
is held fixed for both ranking rules and all training refits.

## Primary outputs

- `policy_summary.csv`: total incremental conversions, pointwise 95% intervals,
  conversions per 1,000 eligible customers, policy conversion-rate estimates,
  actual contact fractions, email allocation, and matched purchase counts.
- `paired_comparisons.csv`: all 63 paired contrasts. The sign is **policy minus
  reference**; inspect the interval of the difference, not the overlap of two
  separate policy intervals.
- `policy_treatment_components.csv`: additive email-specific contributions to
  each deterministic combined policy. Both components sum to its total.
- `treatment_uplift_at_budget.csv`: positive-uplift ranking for each email
  against control separately, on 8,522 Men's/control or 8,539 Women's/control
  customers. These cohorts overlap in control customers and have different
  sizes from the combined policy cohort. **Do not add their contact counts or
  estimated values to create a combined-budget result.**
- `treatment_curve_metrics.csv`: normalized Qini and normalized AUUC from
  scikit-uplift; these are supplementary whole-curve diagnostics, not accuracy
  or combined-policy value estimates.

The three-action IPW incremental score for a fixed recommendation pi is

`1[A=pi(X)] * Y / e(A) - 1[A=0] * Y / e(0)`.

Assignment probabilities e are 1/3 from the randomized dataset documentation,
not the model's purchase probabilities or estimated observed arm frequencies.
Unselected rows cancel exactly. Summing scores estimates additional conversions
over the eligible cohort relative to No Email. Binary comparisons use 1/2
assignment probabilities conditional on their two arms. Scores are evaluation
quantities, not observed individual counterfactual effects.

The 2,000 bootstrap draws resample aligned customer rows with replacement and
reuse the same draws for every strategy. Policies remain fixed. Conversion
counts are not fixed by outcome-stratified resampling. The original dataset
split was stratified on action and conversion, so intervals are approximate
population inference rather than an exact design-based analysis of that split.
They are pointwise and unadjusted for multiple comparisons.

## Reliability and random allocation

- `reliability_summary.csv`: selection overlap, Jaccard, shared-selection email
  agreement, all-customer action agreement, preferred-email agreement before
  budget exclusion, and training-value spreads.
- `reliability_pairs.csv`: 2,280 records (190 pairs times 4 strategies times 3
  budgets), scoring exactly the same customers. Pair quantiles are descriptive;
  pairs share refits and are not independent observations.
- `training_refit_values.csv`: 20 within-action development bootstrap refits
  per family, with identical training samples for both families. Purchase
  counts vary; outcomes are never used to choose the samples or test rankings.
- `training_paired_comparisons.csv`: strategy differences paired by training
  resample, separately from fixed-policy test confidence intervals.
- `random_allocation_values.csv` and `random_allocation_summary.csv`: value
  variation across 100 seeded random rankings at each budget. Both email
  baselines share selected customers. Primary random-policy intervals bootstrap
  the mean over these rankings; they are narrower than the variation of one
  realized random campaign. Allocation SD is reported separately.

Selection overlap is shared contacts / smaller selected count. Jaccard is
shared contacts / union contacts. Email agreement conditions on both refits
selecting the customer. Full action agreement includes the many customers
assigned No Email and should not be read as treatment stability.

Training-value SD and ranges describe variation of fitted policies on one
fixed evaluation dataset. They are not test confidence intervals, nor estimates
of the full uncertainty of a future campaign. Test and training uncertainties
have not been combined. Refits are sample perturbations, not just changes to
the algorithm's random seed.

## Segmentation and messages

- `cluster_comparison.csv`: development-only k selection, sampled silhouette,
  size gate, bootstrap ARI, and explicit fallback flag. k=2 wins by the frozen
  rule but its minimum ARI of 0.640 indicates limited stability.
- `cluster_profiles.csv`: development and fixed-test attributes and sizes.
- `cluster_test_responses.csv`: randomized-arm counts, purchase rates, Wilson
  intervals after the clusters are frozen.
- `cluster_test_effects.csv`: approximate IPW value intervals for assigning an
  email to all customers of a cluster. These are **uncapped** explanatory
  results, not budget-policy outputs.
- `development_cluster_ablation.csv`: fresh cluster fitting inside development
  folds; cluster-minus-original logistic uplift differences are -3, 0, and -9.
  No benefit demonstrated; no cluster-augmented model promoted on test outcomes.
- `message_contexts.json`, `message_prompts.json`, `message_drafts.json`, and
  `message_audit.csv`: three selected profiles, prompts, nine assistant-authored
  drafts, and limited content checks. These examples were chosen without
  inspecting their observed conversion outcomes. No emails were sent.

Purchase flags describe past merchandise categories, never gender. The copy
contains no invented prices, offers, personal names, or product specifications.
Passing limited content checks does not establish full semantic grounding,
independent relevance scores, or conversion lift. Stage 1 is a compatible
single-contact draft; stages 2 and 3 require a new sequence experiment and
contact budget. See `docs/research/message_review_rubric.md` for review criteria.

`frozen_protocol.json` is written before test outcomes are loaded.
`experiment.json` records data/code/config hashes, library versions, arm counts,
and limitations. Full per-customer predictions, assignments, and fitted models
are ignored local artifacts. The split and earlier full-data EDA mean that this
test cohort is not historically untouched. New models must not be tuned to
these published held-out results.

References: [Hillstrom documentation](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html),
[Dudik et al. policy evaluation](https://arxiv.org/abs/1103.4601),
[normalized Qini](https://www.uplift-modeling.com/en/latest/api/metrics/qini_auc_score.html),
[normalized uplift area](https://www.uplift-modeling.com/en/latest/api/metrics/uplift_auc_score.html).
