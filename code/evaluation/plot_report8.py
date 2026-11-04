"""Draw Report 8 scientific figures from measured CSV exports."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/targeting-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/report8'
FIGURES = ROOT / 'results/figures'
COLORS = {'logistic_uplift': '#245a81', 'tree_uplift': '#777777'}
LABELS = {'logistic_uplift': 'Logistic uplift', 'tree_uplift': 'Tree uplift'}


def run():
    FIGURES.mkdir(parents=True, exist_ok=True)
    chance = pd.read_csv(RESULTS / 'chance_reference.csv')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for policy, color in COLORS.items():
        rows = chance.loc[chance.policy.eq(policy)].sort_values('budget')
        ax.plot(rows.budget * 100, rows.observed_overlap_mean * 100, marker='o',
                label=LABELS[policy], color=color)
    ax.plot([5, 10, 20], [5, 10, 20], '--', color='black', label='Independent random lists (expected)')
    ax.set(xlabel='Shared contact budget (% of 12,800 customers)', ylabel='Mean selection overlap (%)',
           title='Retrained customer lists overlap more than chance', xticks=[5, 10, 20], ylim=(0, 65))
    ax.legend(frameon=False)
    fig.text(.02, .015, 'Source: Report 8 chance_reference.csv; 20 refits, 190 dependent pairs per model/budget.\nChance uses actual list sizes. Agreement is consistency, not correctness.', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, 1])
    fig.savefig(FIGURES / 'report8_chance_reference.png', dpi=180)
    plt.close(fig)

    training = pd.read_csv(RESULTS / 'training_size_summary.csv')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for col, budget in enumerate((.05, .10, .20)):
        for policy, color in COLORS.items():
            rows = training.loc[training.policy.eq(policy) & training.budget.eq(budget)].sort_values('training_fraction')
            x = rows.training_fraction * 100
            axes[0, col].plot(x, rows.selection_overlap_mean * 100, marker='o', color=color, label=LABELS[policy])
            axes[1, col].plot(x, rows.training_value_mean, marker='o', color=color, label=LABELS[policy])
            axes[1, col].fill_between(x, rows.training_value_p05, rows.training_value_p95, color=color, alpha=.12)
        axes[0, col].axhline(budget * 100, linestyle='--', color='black', linewidth=.8, label='Random-list expectation')
        axes[0, col].set(title=f'{int(budget * 100)}% shared contact budget', ylim=(0, 65))
        axes[1, col].axhline(0, color='black', linewidth=.8)
        axes[1, col].set(xlabel='Nominal training draw count (% of 51,200)', xticks=[25, 50, 75, 100], ylim=(-8, 55))
        axes[0, col].legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel('Mean selection overlap (%)')
    axes[1, 0].set_ylabel('Estimated extra purchases\n(mean and 5th–95th refit spread)')
    fig.suptitle('Training-size study on the same fixed evaluation customers')
    fig.text(.02, .015, 'Source: Report 8 training_size_summary.csv; 20 paired bootstraps per size; fixed model settings and tie seed.\nShading is a descriptive training-refit spread, not a confidence interval. Draws repeat customers; test reuse is exploratory.', fontsize=9)
    fig.tight_layout(rect=[0, .075, 1, .96])
    fig.savefig(FIGURES / 'report8_training_size.png', dpi=180)
    plt.close(fig)

    simulations = pd.read_csv(RESULTS / 'simulation_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    names = {'original_percentile_bootstrap': ('Original percentile bootstrap', '#245a81'),
             'simultaneous_exact_event_bounds': ('Conservative exact reference', '#777777')}
    for ax, base in zip(axes, (.001, .005)):
        for method, (label, color) in names.items():
            rows = simulations.loc[simulations.scenario.eq('budget_grid') & simulations.customers.eq(12800)
                                   & simulations.budget.eq(.05) & simulations.control_rate.eq(base)
                                   & simulations.method.eq(method)].sort_values('effect')
            ax.errorbar(rows.effect * 100, rows.coverage_rate * 100,
                        yerr=np.vstack([(rows.coverage_rate - rows.coverage_mc_lower) * 100,
                                        (rows.coverage_mc_upper - rows.coverage_rate) * 100]),
                        marker='o', color=color, label=label, capsize=3)
        ax.axhline(95, color='black', linestyle='--', linewidth=.8, label='Nominal 95% coverage')
        ax.set(title=f'Control purchase chance: {base * 100:g}%',
               xlabel='Known email benefit among selected customers\n(percentage points)', ylim=(0, 102))
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    axes[0].set_ylabel('Trials whose interval contains the true benefit (%)')
    fig.suptitle('Do the uncertainty intervals contain the known true benefit?')
    fig.text(.02, .015, 'Source: Report 8 parametric simulations; 12,800 evaluation records, 640 contacts, 1,000 trials per scenario.\nBars: 95% Wilson intervals for Monte Carlo proportions. Fixed-policy homogeneous scenarios, not real campaign results.', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .94])
    fig.savefig(FIGURES / 'report8_interval_coverage.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    colors = ['#245a81', '#687c42', '#777777']
    for ax, n in zip(axes, (6400, 12800, 25600)):
        for budget, color in zip((.05, .10, .20), colors):
            rows = simulations.loc[simulations.scenario.eq('budget_grid') & simulations.customers.eq(n)
                                   & simulations.budget.eq(budget) & simulations.control_rate.eq(.005)
                                   & simulations.method.eq('simultaneous_exact_event_bounds')].sort_values('effect')
            ax.errorbar(rows.effect * 100, rows.positive_detection_rate * 100,
                        yerr=np.vstack([(rows.positive_detection_rate - rows.positive_detection_mc_lower) * 100,
                                        (rows.positive_detection_mc_upper - rows.positive_detection_rate) * 100]),
                        label=f'{int(budget * 100)}% budget', color=color, marker='o', capsize=2)
        ax.axhline(80, color='black', linestyle='--', linewidth=.8, label='80% target')
        ax.set(title=f'{n:,} evaluation customers', ylim=(0, 103),
               xlabel='Known email benefit among selections\n(percentage points)')
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Trials detecting a positive benefit (%)')
    fig.suptitle('Detection limits under the conservative simulation reference')
    fig.text(.02, .015, 'Source: Report 8 parametric simulations; control purchase chance 0.5%, 1,000 trials/scenario, fixed policy.\nDetection: lower endpoint of a two-sided 95% interval >0. Bars: Monte Carlo 95% Wilson intervals; no model-training variation.', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .94])
    fig.savefig(FIGURES / 'report8_detection_limits.png', dpi=180)
    plt.close(fig)
    print('Saved four Report 8 figures.')


if __name__ == '__main__':
    run()
