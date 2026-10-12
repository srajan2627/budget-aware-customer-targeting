"""Steps 4–6: frozen held-out evaluation, reliability, and segmentation."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import sklift
from sklearn.exceptions import ConvergenceWarning
from sklift.metrics import qini_auc_score, uplift_auc_score

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
for directory in ('data', 'models', 'targeting', 'segmentation'):
    sys.path.insert(0, str(CODE / directory))
from baselines import ACTIONS, ArmResponseModels
from preprocessing import FEATURE_COLUMNS
from run_baselines import load_shared_folds
from policies import assign_policy
from policy_value import interval, paired_bootstrap_totals, policy_scores
from reliability import pairwise_agreement, training_bootstrap
from clusters import choose_clusters, cluster_pipeline, cluster_responses, profiles


def save(records, path):
    pd.DataFrame(records).to_csv(path, index=False)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_policies(a, y, policies, budgets, e, config, out):
    """Policies map (label,budget) to one policy or repeated random policies."""
    keys, scores, metadata, components, random_rows = [], [], [], [], []
    for (label, budget), recommendations in policies.items():
        draws = np.atleast_2d(recommendations)
        incremental = np.stack([policy_scores(a, y, pi, e)[1] for pi in draws])
        score = incremental.mean(axis=0)
        keys.append((label, budget))
        scores.append(score)
        metadata.append(dict(policy=label, budget=budget, customers=len(a),
                             contacted=float((draws != 0).sum(axis=1).mean()),
                             contact_fraction=float((draws != 0).mean()),
                             mens_contacts=float((draws == 1).sum(axis=1).mean()),
                             womens_contacts=float((draws == 2).sum(axis=1).mean()),
                             matched_contact_purchases=float(((draws != 0) & (draws == a) & (y == 1)).sum(axis=1).mean()),
                             matched_control_purchases_in_selected=float(((draws != 0) & (a == 0) & (y == 1)).sum(axis=1).mean()),
                             random_rankings=len(draws), estimated_incremental_conversions=float(score.sum()),
                             incremental_conversions_per_1000=float(score.mean()*1000),
                             ipw_policy_conversion_rate=float(((a == 0)*y/e[0]).mean()+score.mean())))
        if len(draws) == 1:
            for action in (1, 2):
                components.append((label, budget, action, incremental[0] * (draws[0] == action)))
        else:
            random_rows.extend(dict(policy=label, budget=budget, ranking=i,
                                    estimated_incremental_conversions=float(row.sum()))
                               for i, row in enumerate(incremental))
    all_scores = np.column_stack(scores + [item[3] for item in components])
    bootstrap = paired_bootstrap_totals(all_scores, repeats=config['bootstrap_repeats'], seed=config['seed'])
    for i, row in enumerate(metadata):
        row['ci_lower'], row['ci_upper'] = interval(bootstrap[:, i])
        row['test_bootstrap_sd'] = float(bootstrap[:, i].std(ddof=1))
    save(metadata, out / 'policy_summary.csv')
    contrasts = []
    for budget in budgets:
        ids = [i for i, key in enumerate(keys) if key[1] == budget]
        for i, j in combinations(ids, 2):
            lower, upper = interval(bootstrap[:, j] - bootstrap[:, i])
            contrasts.append(dict(policy=keys[j][0], reference=keys[i][0], budget=budget,
                                  difference_incremental_conversions=float(scores[j].sum()-scores[i].sum()),
                                  ci_lower=lower, ci_upper=upper))
    save(contrasts, out / 'paired_comparisons.csv')
    rows = []
    for offset, (label, budget, action, score) in enumerate(components):
        lower, upper = interval(bootstrap[:, len(keys)+offset])
        rows.append(dict(policy=label, budget=budget, email=ACTIONS[action],
                         estimated_incremental_conversions=float(score.sum()), ci_lower=lower, ci_upper=upper))
    save(rows, out / 'policy_treatment_components.csv')
    save(random_rows, out / 'random_allocation_values.csv')
    random_summary = []
    for key, group in pd.DataFrame(random_rows).groupby(['policy', 'budget']):
        values = group.estimated_incremental_conversions
        random_summary.append(dict(policy=key[0], budget=key[1], allocation_mean=float(values.mean()),
                                   allocation_sd=float(values.std(ddof=1)),
                                   allocation_p05=float(values.quantile(.05)), allocation_p95=float(values.quantile(.95))))
    save(random_summary, out / 'random_allocation_summary.csv')


def treatment_comparisons(a, y, predictions, budgets, config, out):
    rows, areas = [], []
    for family, p in predictions.items():
        for action in (1, 2):
            mask = np.isin(a, [0, action])
            binary = (a[mask] == action).astype(int)
            uplift = p[mask, action] - p[mask, 0]
            areas.append(dict(model=family, email=ACTIONS[action], customers=int(mask.sum()),
                              qini_normalized=float(qini_auc_score(y[mask], uplift, binary)),
                              auuc_normalized=float(uplift_auc_score(y[mask], uplift, binary))))
            scores, selected_rows = [], []
            for budget in budgets:
                order = np.random.default_rng(config['policy_tie_seed']).permutation(mask.sum())
                order = order[uplift[order] > 0]
                selected = order[np.argsort(-uplift[order], kind='stable')][:int(mask.sum()*budget)]
                selection = np.zeros(mask.sum(), dtype=bool)
                selection[selected] = True
                score = selection * (2*binary*y[mask] - 2*(1-binary)*y[mask])
                scores.append(score)
                selected_rows.append(dict(model=family, email=ACTIONS[action], budget=budget,
                                          comparison_customers=int(mask.sum()), selected=int(selection.sum()),
                                          selected_fraction=float(selection.mean()),
                                          selected_treatment_customers=int((selection & (binary == 1)).sum()),
                                          selected_control_customers=int((selection & (binary == 0)).sum()),
                                          selected_treatment_purchases=int((selection*binary*y[mask]).sum()),
                                          selected_control_purchases=int((selection*(1-binary)*y[mask]).sum()),
                                          estimated_incremental_conversions=float(score.sum())))
            bootstrap = paired_bootstrap_totals(np.column_stack(scores), repeats=config['bootstrap_repeats'], seed=config['seed'])
            for i, row in enumerate(selected_rows):
                row['ci_lower'], row['ci_upper'] = interval(bootstrap[:, i])
            rows.extend(selected_rows)
    save(rows, out / 'treatment_uplift_at_budget.csv')
    save(areas, out / 'treatment_curve_metrics.csv')


def cluster_ablation(development, k, config, out):
    """Separate development experiment; test data never enters this function."""
    folds = load_shared_folds(development, ROOT / 'datasets/processed/development_folds.csv')
    probabilities = {label: np.zeros((len(development), 3)) for label in ('logistic_uplift', 'logistic_uplift_with_cluster')}
    for number, (train, val) in enumerate(folds, 1):
        training, validation = development.iloc[train].copy(), development.iloc[val].copy()
        clustering = cluster_pipeline(k).fit(training[FEATURE_COLUMNS])
        training['cluster'] = clustering.predict(training[FEATURE_COLUMNS]).astype(str)
        validation['cluster'] = clustering.predict(validation[FEATURE_COLUMNS]).astype(str)
        for label in probabilities:
            model = ArmResponseModels('logistic', include_cluster=label.endswith('with_cluster')).fit(training)
            probabilities[label][val] = model.predict_probabilities(validation)
        print(f'Development cluster ablation fold {number}/5', flush=True)
    a = development.treatment.map(dict(zip(ACTIONS, range(3)))).to_numpy()
    y = development.conversion.to_numpy()
    scores, rows = [], []
    for budget in config['budgets']:
        base_scores = []
        for label, p in probabilities.items():
            pi = np.zeros(len(a), dtype=int)
            for number, (_, val) in enumerate(folds, 1):
                pi[val] = assign_policy(p[val], 'uplift', budget, seed=42+number)
            score = policy_scores(a, y, pi, config['assignment_probabilities'])[1]
            base_scores.append(score)
            scores.append(score)
            rows.append(dict(policy=label, budget=budget, estimated_incremental_conversions=float(score.sum()),
                             contact_fraction=float((pi != 0).mean())))
        scores.append(base_scores[1]-base_scores[0])
        rows.append(dict(policy='cluster_minus_original', budget=budget,
                         estimated_incremental_conversions=float(scores[-1].sum()), contact_fraction=np.nan))
    bootstrap = paired_bootstrap_totals(np.column_stack(scores), repeats=config['bootstrap_repeats'], seed=config['seed'])
    for i, row in enumerate(rows):
        row['ci_lower'], row['ci_upper'] = interval(bootstrap[:, i])
    save(rows, out / 'development_cluster_ablation.csv')


def reliability(development, features, a, y, config, out):
    assignments = {(family, strategy, budget): [] for family in config['models']
                   for strategy in ('response', 'uplift') for budget in config['budgets']}
    preferred, values = {family: [] for family in config['models']}, []
    for repeat in range(config['training_refits']):
        training = training_bootstrap(development, config['seed'] + repeat)
        for family in config['models']:
            model = ArmResponseModels(family).fit(training)
            p = model.predict_probabilities(features)
            preferred[family].append(assign_policy(p, 'response', 1., seed=config['policy_tie_seed']))
            for strategy in ('response', 'uplift'):
                for budget in config['budgets']:
                    pi = assign_policy(p, strategy, budget, seed=config['policy_tie_seed'])
                    assignments[family, strategy, budget].append(pi)
                    score = policy_scores(a, y, pi, config['assignment_probabilities'])[1]
                    values.append(dict(refit=repeat, sample_seed=config['seed']+repeat, policy=f'{family}_{strategy}',
                                       budget=budget, contacted=int((pi != 0).sum()),
                                       estimated_incremental_conversions=float(score.sum())))
        print(f'Training resample {repeat+1}/{config["training_refits"]}', flush=True)
    rows, summary = [], []
    for (family, strategy, budget), policies in assignments.items():
        pair_rows = pairwise_agreement(policies)
        for row in pair_rows:
            i, j = row['refit_left'], row['refit_right']
            row['preferred_email_agreement_all'] = float((preferred[family][i] == preferred[family][j]).mean())
            row.update(policy=f'{family}_{strategy}', budget=budget)
        rows.extend(pair_rows)
        table = pd.DataFrame(pair_rows)
        record = dict(policy=f'{family}_{strategy}', budget=budget, refits=config['training_refits'], pairs=len(pair_rows))
        for name in ('selection_overlap', 'selection_jaccard', 'treatment_agreement_shared',
                     'action_agreement_all', 'preferred_email_agreement_all'):
            record[f'{name}_mean'] = float(table[name].mean())
            record[f'{name}_p05'] = float(table[name].quantile(.05))
            record[f'{name}_p95'] = float(table[name].quantile(.95))
        v = pd.DataFrame(values)
        v = v.loc[v.policy.eq(f'{family}_{strategy}') & v.budget.eq(budget), 'estimated_incremental_conversions']
        record.update(training_value_mean=float(v.mean()), training_value_sd=float(v.std(ddof=1)),
                      training_value_min=float(v.min()), training_value_max=float(v.max()))
        summary.append(record)
    save(rows, out / 'reliability_pairs.csv')
    save(summary, out / 'reliability_summary.csv')
    save(values, out / 'training_refit_values.csv')
    # Same training samples pair competing strategies. This spread is not a test CI.
    paired = []
    table = pd.DataFrame(values)
    for budget, group in table.groupby('budget'):
        pivot = group.pivot(index='refit', columns='policy', values='estimated_incremental_conversions')
        for left, right in combinations(pivot.columns, 2):
            difference = pivot[right]-pivot[left]
            paired.append(dict(policy=right, reference=left, budget=budget, training_difference_mean=float(difference.mean()),
                               training_difference_sd=float(difference.std(ddof=1)),
                               training_difference_min=float(difference.min()), training_difference_max=float(difference.max())))
    save(paired, out / 'training_paired_comparisons.csv')


def run(config_path, out):
    warnings.filterwarnings('error', category=ConvergenceWarning)
    # sklift 0.5.1 calls a deprecated sklearn helper; results remain supported here.
    warnings.filterwarnings('ignore', message='Function stable_cumsum is deprecated', category=FutureWarning)
    config = json.loads(config_path.read_text())
    if config['training_refits'] < 2 or config['random_rankings'] < 2:
        raise ValueError('Require repeated training and random targeting.')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'frozen_protocol.json').write_text(json.dumps(config, indent=2)+'\n')
    development_path, test_path = (ROOT / 'datasets/processed' / f'{cohort}.csv' for cohort in ('development', 'test'))
    development = pd.read_csv(development_path, index_col='source_row').sort_index()
    clustering, cluster_table = choose_clusters(development, stability_repeats=config['cluster_stability_refits'], seed=config['seed'])
    cluster_table.to_csv(out / 'cluster_comparison.csv', index=False)
    k = int(cluster_table.loc[cluster_table.selected, 'k'].iloc[0])
    cluster_ablation(development, k, config, out)
    models = {family: ArmResponseModels(family).fit(development) for family in config['models']}
    # Test outcomes first enter the workflow here, after all development fits.
    test = pd.read_csv(test_path, index_col='source_row').sort_index()
    if not development.index.is_unique or not test.index.is_unique or len(development.index.intersection(test.index)):
        raise ValueError('Development and test require unique, nonoverlapping source rows.')
    a = test.treatment.map(dict(zip(ACTIONS, range(3)))).to_numpy()
    y = test.conversion.to_numpy()
    features = test[FEATURE_COLUMNS]
    predictions = {family: model.predict_probabilities(features) for family, model in models.items()}
    budgets, e = config['budgets'], config['assignment_probabilities']
    policies = {}
    assignments = pd.DataFrame(index=test.index)
    predictions_export = pd.DataFrame(index=test.index)
    dummy = np.zeros((len(test), 3))
    for budget in budgets:
        policies['no_contact', budget] = np.zeros(len(test), dtype=int)
        for email in ('mens', 'womens'):
            policies[f'random_{email}_mean', budget] = np.stack([
                assign_policy(dummy, f'random_{email}', budget, seed=config['seed']+i)
                for i in range(config['random_rankings'])])
        for family, p in predictions.items():
            for strategy in ('response', 'uplift'):
                pi = assign_policy(p, strategy, budget, seed=config['policy_tie_seed'])
                policies[f'{family}_{strategy}', budget] = pi
                assignments[f'{family}_{strategy}_{int(budget*100)}pct'] = pi
    for family, p in predictions.items():
        for action in range(3):
            predictions_export[f'{family}_p_{action}'] = p[:, action]
        for action in (1, 2):
            predictions_export[f'{family}_uplift_{action}'] = p[:, action]-p[:, 0]
        joblib.dump(models[family], out / f'{family}_model.joblib')
    assignments.to_csv(out / 'test_assignments.csv', index_label='source_row')
    predictions_export.to_csv(out / 'test_predictions.csv', index_label='source_row')
    summarize_policies(a, y, policies, budgets, e, config, out)
    treatment_comparisons(a, y, predictions, budgets, config, out)
    reliability(development, features, a, y, config, out)
    dev_labels, test_labels = clustering.predict(development[FEATURE_COLUMNS]), clustering.predict(features)
    save(profiles(development, dev_labels, 'development')+profiles(test, test_labels, 'test'), out / 'cluster_profiles.csv')
    cluster_responses(test, test_labels).to_csv(out / 'cluster_test_responses.csv', index=False)
    cluster_effect_scores, effect_rows = [], []
    for cluster in range(k):
        for action in (1, 2):
            score = (test_labels == cluster)*((a == action)*y/e[action] - (a == 0)*y/e[0])
            cluster_effect_scores.append(score)
            effect_rows.append(dict(cluster=cluster, email=ACTIONS[action], cluster_customers=int((test_labels == cluster).sum()),
                                    estimated_incremental_conversions_all_in_cluster=float(score.sum())))
    boot = paired_bootstrap_totals(np.column_stack(cluster_effect_scores), repeats=config['bootstrap_repeats'], seed=config['seed'])
    for i, row in enumerate(effect_rows):
        row['ci_lower'], row['ci_upper'] = interval(boot[:, i])
    save(effect_rows, out / 'cluster_test_effects.csv')
    joblib.dump(clustering, out / 'cluster_model.joblib')
    # Select messaging examples from fixed recommendations and attributes only.
    eligible = assignments['logistic_uplift_20pct'].to_numpy() != 0
    examples = []
    for action in (1, 2):
        ids = np.flatnonzero(eligible & (assignments['logistic_uplift_20pct'].to_numpy() == action))
        chosen = []
        for cluster in range(k):
            members = ids[test_labels[ids] == cluster]
            if len(members):
                chosen.append(int(members[0]))
        for i in chosen[:3]:
            record = {name: test.iloc[i][name].item() if isinstance(test.iloc[i][name], np.generic) else test.iloc[i][name]
                      for name in FEATURE_COLUMNS}
            record.update(source_row=int(test.index[i]), cluster=int(test_labels[i]), recommended_action=ACTIONS[action],
                          estimated_uplift=float(predictions['logistic'][i, action]-predictions['logistic'][i, 0]))
            examples.append(record)
    (out / 'message_contexts.json').write_text(json.dumps(examples, indent=2)+'\n')
    manifest = dict(config_sha256=digest(config_path), development_sha256=digest(development_path), test_sha256=digest(test_path),
                    code_sha256={str(path.relative_to(ROOT)): digest(path) for path in sorted(CODE.rglob('*.py'))},
                    customers=len(test), purchases=int(y.sum()), selected_k=k,
                    test_arm_counts=test.groupby('treatment').size().to_dict(),
                    test_arm_purchases=test.groupby('treatment').conversion.sum().astype(int).to_dict(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                  sklearn=sklearn.__version__, sklift=sklift.__version__, joblib=joblib.__version__),
                    limitations=['Prior full-data descriptive EDA; test is not historically untouched.',
                                 'Outcome-stratified initial split; bootstrap intervals are approximate population inference, not exact design-based inference.',
                                 'Only 115 test purchases. Pointwise intervals, multiple comparisons, no deployment winner chosen.',
                                 'Fixed-policy evaluation bootstrap excludes training and reranking uncertainty; refit ranges are reported separately.',
                                 'Cluster ablation is exploratory development OOF; global k chosen on development attributes, so not fully nested model selection.',
                                 'Equal assignment probabilities 1/3 are supported by dataset documentation; no observational propensity fitting.',
                                 'Generated messages and multiple-touch sequences were never randomized in Hillstrom.'])
    (out / 'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Completed Steps 4–6 analysis in {out}; no commit or push performed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/evaluation.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/evaluation')
    args = parser.parse_args()
    run(args.config, args.output_dir)
