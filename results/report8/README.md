# Report 8 output guide

These are new exploratory diagnostics, kept separately from `results/evaluation`.
Read `docs/research/report8_protocol.md` and `frozen_protocol.json` before
interpreting the numbers. Simulated probabilities are hypothetical and are not
estimated customer effects.

Run from the repository root with the existing environment activated:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/evaluation/run_report8.py
python code/evaluation/plot_report8.py
python -m unittest discover -s tests -v
```

The runner replaces this directory's experiment exports. It does not replace
the earlier campaign evaluation. `--output-dir PATH` selects another destination.

- `historical_interval_audit.csv`: 36 model/budget/component records. Counts
  contributing events, distinguishes structural zeros from nonempty zero-event
  groups, retains raw percentile endpoints, and withholds the separate display
  endpoints for the latter. Nonzero support is not a coverage guarantee.
- `chance_reference.csv`: 12 original-policy/budget comparisons with analytic
  chance expectations, excess overlap and chance-adjusted overlap. These concern
  selection consistency, not action correctness.
- `chance_reference_check.csv`: three analytic expectations checked against
  10,000 independent random-list pair intersections each. Random-pair quantiles
  are a reference distribution, not confidence bounds on model stability.
- `training_size_summary.csv`: 48 policy/size/budget summaries of the same fixed
  evaluation customers. Full-size results reproduce the original reliability
  study. Sizes denote bootstrap draw counts, not distinct independent customers.
- `training_size_values.csv`: 960 paired refit/policy/budget value estimates,
  actual contacts, email allocations and unique training counts.
- `training_sample_counts.csv`: 240 refit/size/action records showing draw and
  unique-record counts and purchases. `training_failures.json` records failures.
- `training_size_pairs.csv`: 9,120 dependent within-size refit pairs. They share
  fitted models and cannot be treated as 9,120 independent research experiments.
- `training_size_paired_differences.csv`: 36 full-minus-smaller descriptive
  comparisons matched on seed. Min/max and SD describe training variation,
  not evaluation-sample confidence intervals.
- `simulation_summary.csv`: 218 scenario/method summaries from 109,000 generated
  datasets, each evaluated by both interval methods. Coverage, detection, null
  rejection, width, bias and Monte Carlo uncertainty are exported. The sparse
  122-contact stress case is separate from the 108-scenario budget grid.
- `detection_limits.csv`: 18 sample-size/budget/control-rate conservative-reference
  grid thresholds. Missing values mean the 80% Monte Carlo lower-bound criterion
  was not reached on the tested grid. This does not establish a universal MDE.
- `frozen_protocol.json` and `experiment.json`: settings, assumptions, input/code
  hashes, versions and limitations. Authoring code is not an analysis input.
- `verification.json`: recorded reproducibility and cross-check results.

Figures are `results/figures/report8_chance_reference.png`,
`report8_training_size.png`, `report8_interval_coverage.png`, and
`report8_detection_limits.png`. Training bands are descriptive refit spreads;
simulation bars are Wilson95% intervals for Monte Carlo proportions.

Build the report using the optional document-authoring dependency `lxml`:

```bash
python code/reporting/build_report8.py
```

It uses the retained `docs/Progress_Reports/Progress_Report_7.docx` template and
writes `Progress_Report_8.docx`. Word layout was checked with the Documents
renderer; the renderer is an authoring tool, not an analysis dependency.
