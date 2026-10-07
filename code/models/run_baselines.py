"""Run Step 3 on development data only; never open the final test dataset."""
import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.tree import export_text

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
for directory in ('data', 'targeting', 'evaluation'):
    sys.path.insert(0, str(CODE / directory))
from baselines import ACTIONS, FAMILIES, ArmResponseModels
from development_cv import make_development_folds
from preprocessing import FEATURE_COLUMNS
from policies import assign_policy
from policy_value import policy_scores

BUDGETS = (0.05, 0.10, 0.20)


def load_shared_folds(df, path):
    if not path.exists():
        folds = make_development_folds(df)
        assignment = pd.Series(0, index=df.index, name='validation_fold')
        for number, (_, validation) in enumerate(folds, 1):
            assignment.iloc[validation] = number
        path.parent.mkdir(parents=True, exist_ok=True)
        assignment.to_csv(path, index_label='source_row')
        return folds
    saved = pd.read_csv(path, index_col='source_row')
    if not saved.index.is_unique or set(saved.index) != set(df.index):
        raise ValueError('Saved folds must match development source rows exactly.')
    assignment = saved.loc[df.index, 'validation_fold'].to_numpy()
    if set(assignment) != {1, 2, 3, 4, 5}:
        raise ValueError('Expected exactly five saved validation folds numbered 1–5.')
    folds = []
    for number in range(1, 6):
        train, val = np.flatnonzero(assignment != number), np.flatnonzero(assignment == number)
        if df.iloc[val].groupby(['treatment', 'conversion']).ngroups != 6:
            raise ValueError('Every validation fold must contain all six strata.')
        folds.append((train, val))
    return folds


def probability_metrics(y, predictions):
    return dict(average_precision=average_precision_score(y, predictions),
                brier_score=brier_score_loss(y, predictions),
                log_loss=log_loss(y, predictions, labels=[0, 1]))


def run(development_path, fold_path, output_dir):
    # Fail on incomplete optimization instead of silently comparing bad fits.
    warnings.filterwarnings('error', category=ConvergenceWarning)
    df = pd.read_csv(development_path, index_col='source_row').sort_index()
    if not df.index.is_unique or not set(df['conversion'].unique()) == {0, 1}:
        raise ValueError('Require unique source rows and binary purchase outcomes.')
    if set(df['treatment'].unique()) != set(ACTIONS):
        raise ValueError('Require the three Hillstrom treatment groups.')
    folds = load_shared_folds(df, fold_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_codes = df['treatment'].map(dict(zip(ACTIONS, range(3)))).to_numpy()
    outcomes = df['conversion'].to_numpy()
    # Explicit balanced-randomization assumption, not purchase probabilities.
    propensities = np.repeat(1 / 3, 3)
    metadata = dict(
        evaluation='development out-of-fold only; not final test performance',
        initial_eda='Used all records before the test split was frozen.',
        seed=42, budgets=BUDGETS, actions=ACTIONS, assignment_probabilities=propensities.tolist(),
        assignment_assumption='Equal randomized allocation to three arms.',
        logistic=dict(C=1.0, solver='lbfgs', max_iter=2000, class_weight=None),
        tree=dict(max_depth=3, min_samples_leaf=200, random_state=42, class_weight=None),
        tuning='Fixed starting settings; no hyperparameter search or winner selection.',
        versions=dict(sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__),
        limitations='IPW point estimates only; no confidence intervals or stability analysis. '
                    'One fixed random ranking per fold. Rare purchases make estimates noisy.',
    )
    metrics, summary, fold_summary, coefficients = [], [], [], []
    saved_predictions = pd.DataFrame(index=df.index)
    saved_predictions['validation_fold'] = 0
    recommendations = pd.DataFrame(index=df.index)
    for family in FAMILIES:
        probabilities = np.full((len(df), 3), np.nan)
        constants = np.full((len(df), 3), np.nan)
        for number, (train, val) in enumerate(folds, 1):
            model = ArmResponseModels(family).fit(df.iloc[train])
            probabilities[val] = model.predict_probabilities(df.iloc[val])
            constants[val] = [model.prevalence_[action] for action in ACTIONS]
            saved_predictions.iloc[val, saved_predictions.columns.get_loc('validation_fold')] = number
            print(f'{family}: completed fold {number}/5', flush=True)
        if not np.isfinite(probabilities).all():
            raise RuntimeError('Out-of-fold predictions are incomplete.')
        for code, action in enumerate(ACTIONS):
            saved_predictions[f'{family}_p_{code}'] = probabilities[:, code]
            mask = action_codes == code
            for name, p in [(family, probabilities), ('training_arm_prevalence', constants)]:
                if name == 'training_arm_prevalence' and family != FAMILIES[0]:
                    continue
                metrics.append(dict(model=name, action=action, customers=int(mask.sum()),
                                    purchases=int(outcomes[mask].sum()),
                                    **probability_metrics(outcomes[mask], p[mask, code])))
        for code in (1, 2):
            saved_predictions[f'{family}_uplift_{code}'] = probabilities[:, code] - probabilities[:, 0]
        strategies = ['response', 'uplift']
        if family == FAMILIES[0]:
            strategies = ['no_contact', 'random_mens', 'random_womens'] + strategies
        for strategy in strategies:
            label = f'{family}_{strategy}' if strategy in ('response', 'uplift') else strategy
            for budget in BUDGETS:
                recommended = np.zeros(len(df), dtype=int)
                for number, (_, val) in enumerate(folds, 1):
                    recommended[val] = assign_policy(probabilities[val], strategy, budget, seed=42 + number)
                    value, incremental = policy_scores(action_codes[val], outcomes[val], recommended[val], propensities)
                    fold_summary.append(dict(policy=label, budget=budget, fold=number,
                                             customers=len(val), contacted=int((recommended[val] != 0).sum()),
                                             estimated_incremental_conversions=float(incremental.sum())))
                value, incremental = policy_scores(action_codes, outcomes, recommended, propensities)
                summary.append(dict(
                    policy=label, budget=budget, customers=len(df),
                    contacted=int((recommended != 0).sum()), contact_fraction=float((recommended != 0).mean()),
                    mens_contacts=int((recommended == 1).sum()), womens_contacts=int((recommended == 2).sum()),
                    matched_contact_purchases=int(((recommended != 0) & (recommended == action_codes) & (outcomes == 1)).sum()),
                    ipw_conversion_rate=float(value.mean()),
                    incremental_conversions_per_1000=float(incremental.mean() * 1000),
                    estimated_incremental_conversions=float(incremental.sum()),
                ))
                recommendations[f'{label}_{int(budget * 100)}pct'] = recommended
        # Refit each fixed candidate for later use; these fits are NOT scored here.
        fitted = ArmResponseModels(family).fit(df)
        joblib.dump(fitted, output_dir / f'{family}_development_model.joblib')
        for action, pipeline in fitted.models_.items():
            estimator = pipeline.named_steps['model']
            names = pipeline.named_steps['preprocess'].get_feature_names_out()
            if family == 'logistic':
                for name, weight in zip(['intercept', *names], [estimator.intercept_[0], *estimator.coef_[0]]):
                    coefficients.append(dict(action=action, feature=name, coefficient=float(weight)))
            else:
                (output_dir / f'tree_action_{ACTIONS.index(action)}.txt').write_text(
                    f'{action}\n' + export_text(estimator, feature_names=list(names), show_weights=True)
                )
    for name, records in [('predictive_metrics', metrics), ('policy_comparison', summary),
                          ('policy_by_fold', fold_summary), ('logistic_coefficients', coefficients)]:
        pd.DataFrame(records).to_csv(output_dir / f'{name}.csv', index=False)
    saved_predictions.to_csv(output_dir / 'oof_predictions.csv', index_label='source_row')
    recommendations.to_csv(output_dir / 'policy_assignments.csv', index_label='source_row')
    (output_dir / 'experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {len(summary)} policy/budget comparisons to {output_dir}')
    print('Final test data was not loaded. Estimates are exploratory development results.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-path', type=Path, default=ROOT / 'datasets/processed/development.csv')
    parser.add_argument('--fold-path', type=Path, default=ROOT / 'datasets/processed/development_folds.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/baselines')
    args = parser.parse_args()
    run(args.development_path, args.fold_path, args.output_dir)
