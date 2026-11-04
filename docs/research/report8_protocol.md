# Report 8 reliability and detection study

This follow-up investigates limitations identified after the original Hillstrom
evaluation. It is exploratory: the evaluation outcomes were already inspected.
It does not select a deployment model or provide new untouched test evidence.
The numeric design is frozen in `configs/report8.json` before each run. This is
a reproducible execution protocol, not a claim of prospective preregistration.

## Questions and comparisons

1. Does the original percentile bootstrap contain a known true policy benefit
   often enough, including when the number of contributing purchases is tiny?
2. How much selection overlap exceeds the expectation for independent random
   lists with the same contact counts?
3. How do selection consistency and estimated value change with nominal
   training draw count, holding algorithms and evaluation customers fixed?
4. Under specified outcome rates and benefits, how often is a positive benefit
   detected, and which tested effects meet an 80% detection target?

## Historical audit and chance reference

The original fitted policies and their per-email contributions are reconstructed
using the existing settings and tie seed. The original IPW point estimator and
earlier files are not changed. An audit counts positive and negative nonzero
scores. A nonempty group with no nonzero scores is flagged as uninformative;
its separate display-interval columns are missing rather than [0, 0]. An empty
contact group is a structural zero. A prespecified fewer-than-ten-event warning
is descriptive only; exceeding ten does not establish correct coverage.

For independent uniform subsets of sizes k1 and k2 from N customers, expected
intersection is k1*k2/N. Since overlap is intersection/min(k1,k2), the expected
overlap is max(k1,k2)/N. At equal sizes this is the contact fraction. Each refit
pair uses its actual counts, including budgets that might be underfilled.
Chance-adjusted overlap is (observed - expected)/(1 - expected). It is
undefined for an empty list or expected overlap one. The reference only
addresses customer selection, not email-choice correctness. A 10,000-pair
hypergeometric simulation independently checks the equal-size expectations.

## Training-size design

At each of 20 seeds, draw a bootstrap independently within each historical
action. Use nested prefixes for 25%, 50%, 75% and 100% nominal draw counts.
The sizes are the sum of floor(fraction * arm size), so smaller totals can be
one or two below fraction*51200. Samples are paired across fractions and model
families; purchase counts are allowed to vary. Draws repeat source records and
do not create independent customers. Export draws, unique source rows and
purchase counts by arm; any fitting failure must be recorded, not hidden.

Fit fresh training-only preprocessing and the fixed logistic and tree models.
The tree remains depth3/minimum leaf200: its effective flexibility changes
with sample size. Score the same 12,800 evaluation customers, using tie seed42
and one combined 5/10/20 percent budget. Include response and uplift policies.
Summarize selection overlap, Jaccard, shared-email agreement, chance adjustment,
and descriptive IPW value variation. Pair full-minus-smaller value differences
within training seed. The 190 refit pairs are dependent, not independent trials;
refit quantiles and standard deviations are not test-sample confidence intervals.
The full-size bootstrap exactly reproduces the earlier reliability design.

## Known-truth simulations

Following the aims/data-generating mechanism/estimand/methods/performance
structure recommended by [Morris, White and Crowther (2019)](https://doi.org/10.1002/sim.8086):

- **Aim:** test fixed-policy evaluation under rare outcomes and characterize
  conditional detection limits, separate from training uncertainty.
- **Data generation:** N=6400/12800/25600; k=floor(N*budget) selected before
  outcomes. Independent historical actions have probability1/3 each. Selected
  customers have homogeneous control probability p0=.001/.005, and purchase
  probability p0+delta under the recommended email. Effects delta are
  0/.0025/.005/.01/.02/.04. No model is trained and no real attributes are used.
  An additional sparse case has N12800,k122,p0=.005,delta=.01. These are 109
  parametric scenarios with 1000 independent trials each, not semi-synthetic
  experiments, external validation or observed Hillstrom causal effects.
- **Estimand:** expected incremental conversions for the fixed policy versus
  No Email, k*delta. The +delta is an absolute probability difference among
  selected customers. Increasing N at fixed contact fraction also increases k;
  it is not a fixed-contact comparison.
- **Score generation:** selected scores are +3 with probability(p0+delta)/3,
  -3 with probabilityp0/3, and zero otherwise. Unselected scores cancel exactly.
  The two nonzero counts have a multinomial distribution over k trials; their
  difference times3 is the IPW total. This reproduces the marginal distribution
  of independent randomized actions and binary outcomes without generating
  irrelevant opposite-email or unselected outcomes.
- **Methods:** compare the original full-N row-percentile bootstrap (2000
  draws) with a conservative simultaneous exact-event reference. Both see the
  same simulated event counts. The bootstrap is computed from full-cohort
  positive/negative/zero counts; it is distributionally identical to resampling
  the original N scores, not to conditioning on observed purchase counts.
- **Performance:** bias and its Monte Carlo SE, RMSE, interval width, coverage,
  zero-width frequency, positive detection, two-sided rejection and zero-event
  frequency. Positive detection means lower bound>0 for a two-sided95% interval.
  Under delta0, report two-sided false-positive frequency separately. Export
  Monte Carlo SEs and Wilson95% intervals for every simulated proportion.

Each scenario has an independent SeedSequence-derived data stream and an
independent bootstrap stream. Repetitions share counts across interval methods
but not across unrelated scenarios. With 1000 trials, the largest possible
binomial Monte Carlo SE is approximately1.58 percentage points. Wilson bounds
remain nondegenerate at observed proportions0 or1.

## Conservative reference and its limits

Under the homogeneous iid selected-population simulation, each event-count
marginal is binomial(k,q_plus) or binomial(k,q_minus). Construct a 97.5% two-sided
Clopper-Pearson interval for each (tail probability .0125 for alpha.05), using
SciPy's beta quantiles. By the union bound, both cover simultaneously with
probability at least95%, even though the margins are dependent. Form
3*k*[L_plus-U_minus, U_plus-L_minus] and intersect with the physical [-k,k]
effect range. An exact small-case enumeration and comparison to SciPy scalar
binomial intervals validate the implementation. See [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

This interval is deliberately conservative and can have very low detection
probability. Good coverage does not imply useful precision. Its guarantee
belongs to the stated binomial margins, not every heterogeneous finite-cohort
policy comparison. We do not install it as the general Hillstrom evaluator or
claim the sparse-interval problem has been solved for arbitrary learned policies.

## Detection grid

Report the smallest positive tested delta for which the Wilson95% lower bound
on the exact-reference detection proportion is at least80%. Do not interpolate,
extrapolate beyond the largest tested effect, or substitute an observed Hillstrom
estimate for the simulation truth. A missing threshold means it was not reached
on this tested grid with this conservative method. It is not evidence of an
absolute impossibility of detection. Monte Carlo bounds are pointwise; choosing
the first qualifying grid value does not create simultaneous certainty across
all scenarios. Original-bootstrap detection is exported for comparison but must
be interpreted alongside its coverage failures. The grid concerns policy versus
No Email; it does not establish power for differences between learned strategies.

## Out of scope

No fresh real-world validation, new causal learner, visit/spend optimization,
new clusters, GenAI quality ratings or message-conversion study is added. Keep
training variation, limited-evaluation uncertainty and simulation Monte Carlo
uncertainty separate. New independent data or a campaign trial is needed before
making deployment claims.
