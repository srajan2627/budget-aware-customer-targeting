"""Export static scientific figures from measured evaluation results."""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/targeting-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def run():
    directory = ROOT/'results/evaluation'
    figures = ROOT/'results/figures'
    figures.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(directory/'policy_summary.csv')
    fig, axes = plt.subplots(1,3,figsize=(13.5,4.8),sharey=True)
    labels = {'random_mens_mean':'Random men mean', 'random_womens_mean':'Random women mean',
              'logistic_response':'Logistic response','logistic_uplift':'Logistic uplift',
              'tree_response':'Tree response','tree_uplift':'Tree uplift'}
    for ax,budget in zip(axes,[.05,.1,.2]):
        rows=results.loc[results.budget.eq(budget) & results.policy.ne('no_contact')]
        for i,row in enumerate(rows.itertuples()):
            ax.errorbar(i,row.estimated_incremental_conversions,
                        yerr=[[row.estimated_incremental_conversions-row.ci_lower],
                              [row.ci_upper-row.estimated_incremental_conversions]],
                        fmt='o',capsize=4,color='#245a81' if 'uplift' in row.policy else '#555555')
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_xticks(np.arange(len(rows)),[labels[x] for x in rows.policy],rotation=60,ha='right')
        ax.set_title(f'{int(budget*100)}% shared budget')
        ax.set_xlabel('Policy (random baselines average 100 rankings)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Incremental conversions vs No Email (count)',fontsize=10)
    fig.suptitle('Held-out policy value with pointwise 95% paired bootstrap intervals')
    fig.text(.02,.015,'Source: fixed Hillstrom test cohort, n=12,800; two-week outcome. 2,000 fixed-policy row resamples; no multiplicity correction.',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.93])
    fig.savefig(figures/'heldout_policy_value.png',dpi=180)
    plt.close(fig)
    stability=pd.read_csv(directory/'reliability_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),sharey=True)
    for policy,marker,color in [('logistic_uplift','o','#245a81'),('tree_uplift','s','#777777')]:
        group=stability.loc[stability.policy.eq(policy)]
        for ax,metric,title in zip(axes,['selection_overlap','treatment_agreement_shared'],
                                   ['Customer selection overlap','Email agreement among shared selections']):
            ax.plot(group.budget*100,group[f'{metric}_mean']*100,marker=marker,color=color,label=policy.replace('_',' '))
            ax.set_title(title)
            ax.set_xlabel('Shared contact budget (% of cohort)')
            ax.set_xticks([5,10,20]); ax.set_ylim(0,100)
            ax.spines[['top','right']].set_visible(False)
            ax.legend(frameon=False)
    axes[0].set_ylabel('Mean across refit pairs (%)')
    fig.text(.02,.015,'Source: 20 paired development bootstraps, 190 dependent refit pairs; same 12,800 evaluation customers; fixed tie seed.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1])
    fig.savefig(figures/'targeting_reliability.png',dpi=180)
    plt.close(fig)
    print('Saved held-out value and reliability figures.')


if __name__=='__main__':
    run()
