"""Self-contained final progress checkpoint, constrained to three pages."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_report(paragraph, table):
    def read(name, group='report8'):
        with (ROOT / 'results' / group / name).open() as stream:
            return list(csv.DictReader(stream))

    def percent(value):
        return f'{float(value) * 100:.1f}%'

    def estimate(row):
        return (f'{float(row["estimated_incremental_conversions"]):g} '
                f'[{float(row["ci_lower"]):g}, {float(row["ci_upper"]):g}]')

    paragraph('Reliable Customer Targeting Under Limited Marketing Campaign Budgets', 'Title')
    paragraph('Project purpose and completed workflow', 'Heading1')
    paragraph('Our project asks whom a business should contact, and which email it should recommend, when only part of the audience can be reached. We have implemented the targeting and evaluation workflow, measured recommendation stability, developed customer profiles, and prepared a GenAI message prototype. This last progress report brings together earlier campaign findings and the new reliability experiments completed after Report 7. The core research workflow is operational; automatic message generation and independent message review remain incomplete.')
    paragraph('We use Hillstrom\'s 64,000-customer randomized experiment: Men\'s Email, Women\'s Email, or No Email, followed by two-week outcomes [1]. Development uses 51,200 records and five shared folds; evaluation uses 12,800 records. Models receive seven pre-email attributes, with preprocessing fitted on training records only. Later visits, purchases, and spending are excluded from model inputs. Purchase conversion is the primary outcome; visits and spending have not been evaluated as alternative targeting objectives.')
    paragraph('Fixed logistic regression and a small decision tree each estimate purchase chances under all three actions. Purchase-likelihood targeting ranks likely buyers; T-learner uplift targeting subtracts the No Email estimate to rank predicted email benefit. Uplift policies choose the better email, contact only positive-uplift customers, and apply one shared budget across both email types. Everyone else receives a No Email recommendation. Both model families reuse the same comparison design.')
    paragraph('Targeting effectiveness on historical evaluation customers', 'Heading1')
    summary = {(r['policy'], float(r['budget'])): r for r in read('policy_summary.csv', 'evaluation')}
    labels = [('random_mens_mean', 'Random Men email mean'),
              ('random_womens_mean', 'Random Women email mean'),
              ('logistic_response', 'Logistic likelihood'), ('logistic_uplift', 'Logistic uplift'),
              ('tree_response', 'Tree likelihood'), ('tree_uplift', 'Tree uplift')]
    rows = []
    for policy, label in labels:
        values = []
        for budget in (.05, .10, .20):
            r = summary[policy, budget]
            if policy.startswith('random_'):
                values.append(f'{float(r["estimated_incremental_conversions"]):.2f} '
                              f'[{float(r["ci_lower"]):.1f}, {float(r["ci_upper"]):.1f}]')
            else:
                values.append(estimate(r))
        rows.append([label, *values])
    table(['Targeting rule', '5% budget', '10% budget', '20% budget'], rows, [3336, 2200, 2200, 2200])
    paragraph('Values are estimated extra purchases versus No Email, with the original nominal 95% bootstrap intervals. All contacting policies selected 640, 1,280, or 2,560 customers: exactly 5%, 10%, or 20% combined. No Email has zero incremental benefit by definition, not zero purchases. Random results average 100 selections. These earlier estimates are retained, not new campaign sales; interval limitations are examined on page 2.')
    paragraph('Logistic likelihood has the highest point estimate at each budget. At 20%, logistic uplift estimates 18.57 more purchases than mean random Men\'s Email, but the original paired interval is [-4.44, 43.62]. Tree uplift also includes zero; neither establishes a clear advantage. Logistic likelihood\'s corresponding difference is 33.57 [4.52, 64.47], encouraging under the original analysis, but sparse outcomes and multiple comparisons prevent a proven-winner claim. The lesson is to retain simple baselines rather than assume uplift must outperform them.')

    paragraph('Reliability and strength of the evidence', 'Heading1', True)
    paragraph('Because each historical customer received one action, we cannot observe every alternative outcome. Inverse-probability weighting uses matching actions and known one-third assignment probabilities to estimate campaign benefit. Earlier uncertainty used 2,000 paired resamples. At 20%, logistic uplift recommends 2,132 Men\'s and 428 Women\'s Emails, with estimated contributions of 33 and 3 purchases. Separate email-control cohorts must not be added together.')
    paragraph('For uplift selection reliability, 20 paired development bootstraps score the same customers. The new reference compares overlap with independent random lists of matching sizes, checked using 10,000 random pairs per budget.')
    chance = {(r['policy'], float(r['budget'])): r for r in read('chance_reference.csv')}
    table(['Contact budget', 'Random expectation', 'Logistic uplift', 'Tree uplift'], [
        [percent(b), percent(chance['logistic_uplift', b]['chance_overlap_mean']),
         percent(chance['logistic_uplift', b]['observed_overlap_mean']),
         percent(chance['tree_uplift', b]['observed_overlap_mean'])] for b in (.05, .10, .20)
    ], [2136, 2600, 2600, 2600])
    paragraph('Both models overlap more than chance, but lists still change substantially. Among shared selections at 5%, email agreement is 86.5% for logistic uplift and 69.1% for the tree. Agreement is consistency, not correctness; acceptable business stability remains undefined.')
    paragraph('New training-size experiments use 25%, 50%, 75%, and 100% nominal draw counts, with 20 refits each. At 20% contact budget, logistic overlap rises 29.8%, 37.9%, 42.6%, and 45.3%; mean extra-purchase estimates are 24.75, 28.80, 26.70, and 27.45. These are refit averages, not the original policy\'s 36. Tree overlap changes nonmonotonically from 29.0% to 32.8%. More learning records help logistic consistency here, but value does not steadily improve.')
    paragraph('Bootstrap draws repeat records: the full size averages 32,359 unique customers. The 190 pairs are dependent, and refit spreads are not confidence intervals. The tree\'s fixed minimum leaf size of 200 also changes its effective flexibility across sizes.')
    paragraph('Checking whether uncertainty estimates can be trusted', 'Heading1')
    paragraph('We ran 109 known-truth parametric scenarios, with 1,000 trials each, varying evaluation size, purchase rarity, budget, and effect [2]. A fixed-policy stress case selected 122 customers, with purchase chances of 0.5% without email and 1.5% with the recommended email: a true expected benefit of 1.22 purchases.')
    sims = read('simulation_summary.csv')
    sparse = {r['method']: r for r in sims if r['scenario'] == 'sparse_122_contacts'}
    table(['Interval method', 'Trials covering true benefit', 'Zero-width intervals'], [
        [label, percent(sparse[method]['coverage_rate']), percent(sparse[method]['zero_width_rate'])]
        for method, label in [('original_percentile_bootstrap', 'Original percentile bootstrap'),
                              ('simultaneous_exact_event_bounds', 'Conservative exact reference')]
    ], [4536, 2700, 2700])
    paragraph('Two historical nonempty Women\'s Email groups also had misleading [0, 0] intervals: 68 logistic-likelihood recommendations at 5%, and 122 logistic-uplift recommendations at 10%. The audit flags them and withholds display bounds while retaining original exports. Sparse intervals need caution; this does not prove all point estimates wrong. The wider exact reference [3] has 100% observed coverage here, not a general guarantee for heterogeneous policies.')

    paragraph('Detection limits and research interpretation', 'Heading1', True)
    paragraph('Detection requires a two-sided 95% interval above zero. With 12,800 records and a hypothetical 0.5% control chance, the conservative reference does not meet the 80% target on the tested grid at 5% or 10% budget. At 20%, four percentage points is the smallest qualifying tested effect (99.5% detection). We require the Monte Carlo 95% lower bound to reach 80%. These are conditional thresholds, not campaign predictions; one point means 0.5% to 1.5%.')
    paragraph('The findings explain reliability limits: larger learning samples help logistic consistency, while rare purchases weaken inference. The simulations assess fixed policies, not learned rankings or external generalization. Only 115 evaluation purchases were observed; prior full-data exploration and repeated test use make the follow-ups exploratory. Predicting no purchase already gives 99.1% accuracy, so accuracy cannot judge campaign value.')
    paragraph('Segmentation findings', 'Heading1')
    paragraph('We compared two to five training-fitted clusters for size, separation, and stability. The selected groups contain 54.8% and 45.2% of development customers, mainly separating past Women\'s merchandise purchases from Men\'s-only purchases, not gender. Mean stability is 0.784. Adding membership changes development uplift estimates by -3, 0, and -9 purchases across budgets, with no demonstrated improvement. Clusters remain explanatory; this comparison is exploratory.')
    paragraph('GenAI prototype and next deliverables', 'Heading1')
    paragraph('The prototype includes contexts, prompts, a generator interface, and nine Codex-authored drafts for three selected profiles. All pass limited declared-fact and treatment-alignment checks. The script does not call an AI service for fresh copy; undeclared inventions can escape the checks. Independent ratings are pending. Only the first stage matches the evaluated single-email policy.')
    paragraph('Next deliverables are a connected generator with recorded model/prompt settings, adversarial checks using invented facts and wrong categories, and two independent reviews of relevance, grounding, alignment, and clarity. We will revise drafts and report disagreements, exclude No Email customers, and never infer gender from merchandise history. New copy or follow-up conversion benefits require a separate randomized experiment.')
    paragraph('Completed and remaining work', 'Heading1')
    paragraph('Completed: the targeting workflow, historical evaluation, reliability follow-ups, segmentation, message prototypes, and fixed-policy diagnostics. All 26 tests pass; numerical exports reproduce and hashes are recorded. The research contribution is a working budget decision process plus evidence about when its recommendations and measurements become unreliable. Deployment readiness has not been established.')
    paragraph('Remaining priorities: GenAI integration/review, sparse-outcome inference, and final report/presentation consolidation. Clean-environment reproduction is unverified. X-learner/ensembles, visits/spending, doubly robust evaluation, and a second dataset remain unimplemented extensions. No new campaign, profit, or message-conversion gain is claimed. Execution and evidence: README.md, results/evaluation, results/report8, and docs/research/report8_protocol.md.')
    paragraph('[1] Hillstrom: https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html')
    paragraph('[2] Morris et al. (2019): https://doi.org/10.1002/sim.8086')
    paragraph('[3] SciPy exact intervals: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html')
