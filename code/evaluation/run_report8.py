"""Reproduce Report 8 reliability diagnostics without replacing earlier results."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
for directory in ('data', 'models', 'targeting'):
    sys.path.insert(0, str(CODE / directory))
from baselines import ACTIONS, ArmResponseModels
from policies import assign_policy
from preprocessing import FEATURE_COLUMNS
from policy_value import interval, paired_bootstrap_totals, policy_scores
from reliability import pairwise_agreement
from report8_methods import (adjusted_overlap, chance_overlap, count_bootstrap_intervals,
                             exact_event_bounds, nested_training_samples,
                             probability_summary, support_diagnostic)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def chance_references(config, out):
    """Analytic reference conditional on each observed pair's contact counts."""
    pairs = pd.read_csv(ROOT / 'results/evaluation/reliability_pairs.csv')
    n = len(pd.read_csv(ROOT / 'datasets/processed/test.csv'))
    rows = []
    for (policy, budget), group in pairs.groupby(['policy', 'budget'], sort=True):
        expected = np.array([chance_overlap(n, int(r.contacts_left), int(r.contacts_right))
                             for r in group.itertuples()])
        observed = group.selection_overlap.to_numpy()
        rows.append(dict(policy=policy, budget=budget, customers=n, refit_pairs=len(group),
                         observed_overlap_mean=float(np.nanmean(observed)),
                         chance_overlap_mean=float(np.nanmean(expected)),
                         excess_overlap_mean=float(np.nanmean(observed - expected)),
                         adjusted_overlap_mean=float(np.nanmean([
                             adjusted_overlap(x, c) for x, c in zip(observed, expected)]))))
    save(rows, out / 'chance_reference.csv')
    rng = np.random.default_rng(config['seed'])
    rows = []
    for budget in config['budgets']:
        k = int(n * budget)
        # Hypergeometric draws are exactly the intersection-size distribution
        # of two independent uniform k-subsets of N, without materializing lists.
        overlaps = rng.hypergeometric(k, n - k, k, size=config['chance_pairs']) / k
        rows.append(dict(budget=budget, customers=n, contacts=k, independent_pairs=len(overlaps),
                         analytic_expected_overlap=k / n, simulated_mean=float(overlaps.mean()),
                         simulated_mean_mc_se=float(overlaps.std(ddof=1) / np.sqrt(len(overlaps))),
                         random_pair_p025=float(np.quantile(overlaps, .025)),
                         random_pair_p975=float(np.quantile(overlaps, .975))))
    save(rows, out / 'chance_reference_check.csv')


def audit_historical_intervals(development, test, config, out):
    a = test.treatment.map(dict(zip(ACTIONS, range(3)))).to_numpy()
    y = test.conversion.to_numpy()
    rows, columns = [], []
    for family in config['models']:
        p = ArmResponseModels(family).fit(development).predict_probabilities(test[FEATURE_COLUMNS])
        for strategy in ('response', 'uplift'):
            for budget in config['budgets']:
                pi = assign_policy(p, strategy, budget, seed=config['policy_tie_seed'])
                score = policy_scores(a, y, pi, [1 / 3] * 3)[1]
                for component, mask in [('combined', pi != 0), ('mens', pi == 1), ('womens', pi == 2)]:
                    s = score * mask
                    diagnostic = support_diagnostic(s, int(mask.sum()), config['sparse_event_warning_threshold'])
                    rows.append(dict(policy=f'{family}_{strategy}', budget=budget, component=component,
                                     selected=int(mask.sum()), estimated_incremental_conversions=float(s.sum()),
                                     **diagnostic))
                    columns.append(s)
    boot = paired_bootstrap_totals(np.column_stack(columns), repeats=config['bootstrap_repeats'], seed=2026)
    for i, row in enumerate(rows):
        lower, upper = interval(boot[:, i])
        row.update(original_bootstrap_lower=lower, original_bootstrap_upper=upper,
                   display_lower=np.nan if row['degenerate_nonempty'] else lower,
                   display_upper=np.nan if row['degenerate_nonempty'] else upper)
    save(rows, out / 'historical_interval_audit.csv')


def training_size_study(development, test, config, out):
    """No tuning: fixed algorithms, paired nested bootstrap sizes, fixed audience."""
    a = test.treatment.map(dict(zip(ACTIONS, range(3)))).to_numpy()
    y = test.conversion.to_numpy()
    feature_frame = test[FEATURE_COLUMNS]
    values, samples_log, failures = [], [], []
    assignments = {(fraction, family, strategy, budget): []
                   for fraction in config['training_fractions'] for family in config['models']
                   for strategy in ('response', 'uplift') for budget in config['budgets']}
    for repeat in range(config['training_refits']):
        seed = config['training_seed'] + repeat
        samples = nested_training_samples(development, config['training_fractions'], seed)
        for fraction, training in samples.items():
            for action, arm in training.groupby('treatment', sort=True):
                samples_log.append(dict(refit=repeat, sample_seed=seed, training_fraction=fraction,
                                        action=action, draws=len(arm), unique_source_rows=arm.index.nunique(),
                                        purchases=int(arm.conversion.sum())))
            for family in config['models']:
                try:
                    model = ArmResponseModels(family).fit(training)
                except (ValueError, ConvergenceWarning) as error:
                    failures.append(dict(refit=repeat, training_fraction=fraction, family=family, reason=str(error)))
                    continue
                p = model.predict_probabilities(feature_frame)
                for strategy in ('response', 'uplift'):
                    for budget in config['budgets']:
                        pi = assign_policy(p, strategy, budget, seed=config['policy_tie_seed'])
                        score = policy_scores(a, y, pi, [1 / 3] * 3)[1]
                        assignments[fraction, family, strategy, budget].append((repeat, pi))
                        values.append(dict(refit=repeat, sample_seed=seed, training_fraction=fraction,
                                           training_draws=len(training), unique_training_customers=training.index.nunique(),
                                           policy=f'{family}_{strategy}', budget=budget,
                                           selected=int((pi != 0).sum()), contact_fraction=float((pi != 0).mean()),
                                           mens_contacts=int((pi == 1).sum()), womens_contacts=int((pi == 2).sum()),
                                           estimated_incremental_conversions=float(score.sum())))
        print(f'Training-size study: paired refit {repeat + 1}/{config["training_refits"]}', flush=True)
    table = pd.DataFrame(values)
    summary, pairs = [], []
    for (fraction, family, strategy, budget), records in assignments.items():
        if len(records) < 2:
            raise RuntimeError(f'Insufficient successful refits for {(fraction, family, strategy, budget)}')
        comparisons = pairwise_agreement([pi for _, pi in records])
        for row in comparisons:
            row['refit_left'] = records[row['refit_left']][0]
            row['refit_right'] = records[row['refit_right']][0]
            row['chance_overlap'] = chance_overlap(len(test), row['contacts_left'], row['contacts_right'])
            row['adjusted_overlap'] = adjusted_overlap(row['selection_overlap'], row['chance_overlap'])
            row.update(training_fraction=fraction, policy=f'{family}_{strategy}', budget=budget)
        pairs.extend(comparisons)
        g = pd.DataFrame(comparisons)
        v = table.loc[table.training_fraction.eq(fraction) & table.policy.eq(f'{family}_{strategy}') & table.budget.eq(budget)]
        row = dict(training_fraction=fraction, policy=f'{family}_{strategy}', budget=budget,
                   successful_refits=len(records), dependent_refit_pairs=len(comparisons),
                   training_draws=int(v.training_draws.iloc[0]),
                   unique_training_customers_mean=float(v.unique_training_customers.mean()),
                   contact_fraction_mean=float(v.contact_fraction.mean()),
                   training_value_mean=float(v.estimated_incremental_conversions.mean()),
                   training_value_sd=float(v.estimated_incremental_conversions.std(ddof=1)),
                   training_value_p05=float(v.estimated_incremental_conversions.quantile(.05)),
                   training_value_p95=float(v.estimated_incremental_conversions.quantile(.95)))
        for metric in ('selection_overlap', 'selection_jaccard', 'treatment_agreement_shared', 'chance_overlap', 'adjusted_overlap'):
            row[f'{metric}_mean'] = float(g[metric].mean())
        summary.append(row)
    save(summary, out / 'training_size_summary.csv')
    save(values, out / 'training_size_values.csv')
    save(samples_log, out / 'training_sample_counts.csv')
    save(pairs, out / 'training_size_pairs.csv')
    (out / 'training_failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    # Paired within-seed differences from the full nominal draw count. Descriptive,
    # not 190 independent studies or fresh test-sample confidence intervals.
    comparisons = []
    for (policy, budget), group in table.groupby(['policy', 'budget']):
        pivot = group.pivot(index='refit', columns='training_fraction', values='estimated_incremental_conversions')
        for fraction in config['training_fractions']:
            if fraction == 1.:
                continue
            d = (pivot[1.] - pivot[fraction]).dropna()
            comparisons.append(dict(policy=policy, budget=budget, smaller_fraction=fraction,
                                    comparison='full_minus_smaller', paired_refits=len(d),
                                    mean_difference=float(d.mean()), sd_difference=float(d.std(ddof=1)),
                                    minimum_difference=float(d.min()), maximum_difference=float(d.max())))
    save(comparisons, out / 'training_size_paired_differences.csv')


def simulation_study(config, out):
    scenarios = []
    for n in config['simulation_evaluation_sizes']:
        for budget in config['budgets']:
            for control in config['simulation_control_rates']:
                for effect in config['simulation_effects']:
                    scenarios.append(dict(scenario='budget_grid', customers=n, budget=budget,
                                          selected=int(n * budget), control_rate=control, effect=effect))
    scenarios.append(dict(scenario='sparse_122_contacts', customers=12800, budget=122 / 12800,
                          selected=122, control_rate=.005, effect=.01))
    rows = []
    seeds = np.random.SeedSequence(config['seed']).spawn(len(scenarios))
    for index, (scenario, seed) in enumerate(zip(scenarios, seeds)):
        n, k, control, effect = (scenario[key] for key in ('customers', 'selected', 'control_rate', 'effect'))
        if k < 1 or not 0 <= control <= control + effect <= 1:
            raise ValueError('Invalid simulation scenario.')
        data_seed, bootstrap_seed = seed.spawn(2)
        rng = np.random.default_rng(data_seed)
        qplus, qminus = (control + effect) / 3, control / 3
        counts = rng.multinomial(k, [qplus, qminus, 1 - qplus - qminus], size=config['simulation_trials'])[:, :2]
        estimate = 3. * (counts[:, 0] - counts[:, 1])
        truth = k * effect
        methods = {
            'original_percentile_bootstrap': count_bootstrap_intervals(
                counts, n, config['bootstrap_repeats'], np.random.default_rng(bootstrap_seed), config['alpha']),
            'simultaneous_exact_event_bounds': exact_event_bounds(counts, k, config['alpha']),
        }
        for method, (lower, upper) in methods.items():
            row = dict(**scenario, method=method, trials=len(counts), true_incremental_conversions=truth,
                       estimate_mean=float(estimate.mean()), bias=float(estimate.mean() - truth),
                       bias_mc_se=float(estimate.std(ddof=1) / np.sqrt(len(counts))),
                       rmse=float(np.sqrt(np.mean((estimate - truth) ** 2))),
                       mean_interval_width=float(np.mean(upper - lower)),
                       expected_contributing_events=k * (qplus + qminus),
                       theoretical_zero_event_probability=(1 - qplus - qminus) ** k,
                       observed_zero_event_fraction=float((counts.sum(axis=1) == 0).mean()))
            for label, flags in [('coverage', (lower <= truth) & (upper >= truth)),
                                 ('positive_detection', lower > 0),
                                 ('two_sided_rejection', (lower > 0) | (upper < 0)),
                                 ('zero_width', lower == upper)]:
                for name, value in probability_summary(int(flags.sum()), len(flags)).items():
                    row[f'{label}_{name}'] = value
            rows.append(row)
        if (index + 1) % 12 == 0 or index + 1 == len(scenarios):
            print(f'Known-truth simulations: {index + 1}/{len(scenarios)} scenarios', flush=True)
    save(rows, out / 'simulation_summary.csv')
    table = pd.DataFrame(rows)
    mde = []
    for (n, budget, control), group in table.loc[
            table.scenario.eq('budget_grid') & table.method.eq('simultaneous_exact_event_bounds')
    ].groupby(['customers', 'budget', 'control_rate']):
        eligible = group.loc[group.effect.gt(0) & group.positive_detection_mc_lower.ge(config['target_detection_probability'])].sort_values('effect')
        row = dict(customers=n, budget=budget, selected=int(group.selected.iloc[0]), control_rate=control,
                   method='simultaneous_exact_event_bounds', target=config['target_detection_probability'],
                   tested_effect_max=float(group.effect.max()), status='not_reached_on_tested_grid',
                   grid_mde=np.nan, grid_mde_percentage_points=np.nan,
                   true_incremental_conversions_at_grid_mde=np.nan, detection_rate=np.nan,
                   detection_mc_lower=np.nan, detection_mc_upper=np.nan)
        if len(eligible):
            r = eligible.iloc[0]
            row.update(status='smallest_tested_effect_meeting_mc_lower_bound', grid_mde=float(r.effect),
                       grid_mde_percentage_points=float(r.effect * 100),
                       true_incremental_conversions_at_grid_mde=float(r.true_incremental_conversions),
                       detection_rate=float(r.positive_detection_rate),
                       detection_mc_lower=float(r.positive_detection_mc_lower),
                       detection_mc_upper=float(r.positive_detection_mc_upper))
        mde.append(row)
    save(mde, out / 'detection_limits.csv')


def run(config_path, out):
    warnings.filterwarnings('error', category=ConvergenceWarning)
    config = json.loads(config_path.read_text())
    if (config['training_refits'] < 2 or config['simulation_trials'] < 2
            or config['chance_pairs'] < 2 or config['bootstrap_repeats'] < 2):
        raise ValueError('Require repeated experiments.')
    out.mkdir(parents=True, exist_ok=True)
    # Freeze the design before fitting models or computing new result summaries.
    (out / 'frozen_protocol.json').write_text(json.dumps(config, indent=2) + '\n')
    development_path, test_path = [ROOT / 'datasets/processed' / f'{name}.csv' for name in ('development', 'test')]
    development = pd.read_csv(development_path, index_col='source_row').sort_index()
    test = pd.read_csv(test_path, index_col='source_row').sort_index()
    if not development.index.is_unique or not test.index.is_unique or len(development.index.intersection(test.index)):
        raise ValueError('Require unique disjoint development and evaluation source rows.')
    chance_references(config, out)
    audit_historical_intervals(development, test, config, out)
    training_size_study(development, test, config, out)
    simulation_study(config, out)
    sources = [config_path, development_path, test_path, ROOT / 'results/evaluation/reliability_pairs.csv',
               ROOT / 'results/evaluation/reliability_summary.csv',
               *[p for p in CODE.rglob('*.py') if 'reporting' not in p.parts]]
    metadata = dict(protocol_sha256=digest(config_path),
                    inputs_and_analysis_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(set(sources))},
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                  scipy=scipy.__version__, sklearn=sklearn.__version__),
                    development_customers=len(development), evaluation_customers=len(test),
                    evaluation_purchases=int(test.conversion.sum()),
                    limitations=[
                        'Exploratory repeated use of previously inspected Hillstrom evaluation outcomes; no new deployment winner.',
                        'm-out-of-n bootstrap draws with replacement do not create new independent customers.',
                        'Tree minimum leaf size remains 200 at every sample size, so smaller samples also change effective tree complexity.',
                        '190 refit pairs share models and are dependent; reported pair means and refit ranges are descriptive.',
                        'Simulation effects and rates are hypothetical, homogeneous within selections, and not fitted causal truths.',
                        'Simulations validate a fixed-policy estimator under stated assumptions, not learned rankings or external generalization.',
                        'Conservative exact bounds are a reference for iid homogeneous selected populations, not a replacement for heterogeneous policy-difference CIs.',
                        'MDE is the smallest tested effect meeting a Monte Carlo lower-bound criterion, not a continuous or universal detection limit.',
                        'Original percentile intervals can fail in sparse nonempty groups; flagged historical intervals are withheld in this audit.',
                        'Training variability, evaluation uncertainty, and Monte Carlo uncertainty remain separate.'])
    (out / 'experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved Report 8 experiments to {out}. Earlier evaluation exports were not replaced.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/report8.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/report8')
    args = parser.parse_args()
    run(args.config, args.output_dir)
