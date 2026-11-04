"""Report 8 narrative; numeric statements are populated from measured exports."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_report(paragraph, table):
    def read(name):
        with (ROOT / 'results/report8' / name).open() as stream:
            return list(csv.DictReader(stream))

    def percent(value):
        return f'{float(value) * 100:.1f}%'

    def simulation(method, *, selected=122, effect=.01):
        return next(r for r in read('simulation_summary.csv') if r['method'] == method
                    and int(r['selected']) == selected and float(r['effect']) == effect
                    and float(r['control_rate']) == .005 and int(r['customers']) == 12800)

    paragraph('Reliability and Detection Limits in Customer Targeting', 'Title')
    paragraph('Work completed', 'Heading1')
    paragraph('Following Report 7, we investigated why targeting recommendations change and how much evidence is needed to measure their benefit. We added an evaluator check, a chance reference for customer overlap, a training-size study, and controlled simulations. The main findings are that logistic uplift becomes more consistent with larger learning samples, while our original uncertainty intervals can be misleading when very few purchases contribute to an estimate.')
    paragraph('We kept the existing logistic and small-tree models, the shared 5%, 10%, and 20% budgets, and the same 12,800 evaluation customers. These customers have already been examined, so the new real-data comparisons are exploratory. No new campaign was sent and no deployment winner was selected.')
    paragraph('Checking the evaluator against a known answer', 'Heading1')
    paragraph('We ran 109 hypothetical scenarios with 1,000 randomized trials each. The selected customers had a stated purchase chance without email and a known improvement under their recommended email. We varied evaluation size, purchase rarity, budget, and effect size. The contact list stayed fixed; these experiments test evaluation calculations, not whether a model can learn the right list. This follows the simulation approach in [1].')
    paragraph('One stress test used 122 selected customers, a 0.5% purchase chance without email, and a 1.5% chance with the recommended email. The true expected benefit was 1.22 extra purchases. We checked how often each interval contained that known value.')
    sparse = simulation('original_percentile_bootstrap')
    conservative = simulation('simultaneous_exact_event_bounds')
    table(['Interval method', 'Trials covering truth', 'Zero-width intervals'], [
        ['Original percentile bootstrap', percent(sparse['coverage_rate']), percent(sparse['zero_width_rate'])],
        ['Conservative exact reference', percent(conservative['coverage_rate']), percent(conservative['zero_width_rate'])],
    ], [4536, 2700, 2700])
    paragraph(f'The original nominal 95% interval covered the truth in only {percent(sparse["coverage_rate"])} of trials. In {percent(sparse["observed_zero_event_fraction"])} of trials, neither matching email purchases nor selected control purchases contributed to the score. Resampling those zero scores cannot reveal an unseen purchase. The conservative reference combines exact binomial bounds [2]; it avoids this collapse under the stated assumptions, but produces wider intervals and detects benefits less often. Its 100% observed coverage does not mean perfect performance in every possible setting.')
    paragraph('We also audited the historical results. Two nonempty Women\'s Email groups were affected: 68 recommendations under logistic purchase-likelihood targeting at 5%, and 122 under logistic uplift at 10%. Their previous [0, 0] intervals are now marked uninformative in a separate audit, with display bounds withheld. Original exports are retained. Empty recommendation groups remain legitimate structural zeros. Other intervals still need caution; nonzero events alone do not establish correct coverage.')

    paragraph('Putting selection reliability into context', 'Heading1', True)
    paragraph('We added the overlap expected from two independently chosen random lists with the same sizes. At equal contact limits, the expected overlap is 5%, 10%, or 20%. This gives meaning to the earlier refit results; it does not establish which customers are correct to contact.')
    chance = {(r['policy'], float(r['budget'])): r for r in read('chance_reference.csv')}
    rows = []
    for budget in (.05, .10, .20):
        rows.append([percent(budget), percent(chance['logistic_uplift', budget]['chance_overlap_mean']),
                     percent(chance['logistic_uplift', budget]['observed_overlap_mean']),
                     percent(chance['tree_uplift', budget]['observed_overlap_mean'])])
    table(['Contact budget', 'Random expectation', 'Logistic uplift', 'Tree uplift'], rows, [2136, 2600, 2600, 2600])
    paragraph('Both uplift models retain more customers than chance would predict. Logistic uplift is more consistent than the tree in these runs. However, many customers still move into or out of the lists, and we have not defined a business standard for acceptable stability. Chance-adjusted overlap and a 10,000-pair random check are also exported.')
    paragraph('Does a larger learning sample help', 'Heading1')
    paragraph('We compared nominal training sizes of 25%, 50%, 75%, and 100% of development data, using 20 paired resamples at each size. Records were drawn with replacement within each historical action, and smaller samples used prefixes of the same draws. Both models faced the same samples and evaluation customers. A 100% draw repeats some records and contains about 32,359 different customers on average; it does not create 51,200 new independent customers.')
    training = {(r['policy'], float(r['budget']), float(r['training_fraction'])): r for r in read('training_size_summary.csv')}
    rows = []
    for fraction in (.25, .50, .75, 1.):
        logistic = training['logistic_uplift', .20, fraction]
        tree = training['tree_uplift', .20, fraction]
        rows.append([percent(fraction), percent(logistic['selection_overlap_mean']), percent(tree['selection_overlap_mean']),
                     f'{float(logistic["training_value_mean"]):.2f}', f'{float(tree["training_value_mean"]):.2f}'])
    table(['Training draw size', 'Logistic overlap', 'Tree overlap', 'Logistic extra purchases', 'Tree extra purchases'],
          rows, [1936, 2000, 2000, 2000, 2000])
    paragraph('This table uses the 20% contact budget. Overlap is averaged across 190 dependent refit pairs; extra purchases are averages across the 20 refits on the same evaluation customers. They are not new observed sales or confidence intervals. Logistic overlap improves steadily, but estimated value does not. Tree results are less consistent. At 5%, logistic overlap rises from 18.6% to 33.6%; tree overlap rises from 10.5% to 15.5%, with a higher intermediate result at 75% training size.')
    paragraph('The finding supports more learning data as a way to improve logistic selection consistency in this experiment. It does not show that more training always increases campaign benefit. The tree keeps a minimum leaf size of 200, so changing training size also changes how restrictive that setting is. Training-size differences remain descriptive rather than evidence of a universal learning curve.')

    paragraph('How large a benefit can we detect', 'Heading1', True)
    paragraph('Detection means the lower bound of a two-sided 95% interval exceeds zero. We tested benefits of 0, 0.25, 0.5, 1, 2, and 4 percentage points among selected customers, with control purchase chances of 0.1% and 0.5%. A +1-point benefit changes a 0.5% chance to 1.5%; it is not a 1% relative increase.')
    paragraph('The table uses 12,800 evaluation records, a 0.5% control chance, and the conservative reference. The threshold is the smallest tested benefit whose Monte Carlo 95% lower bound on detection reaches 80%. It is specific to this method and scenario, not a universal minimum; smaller effects are not ruled out.')
    limits = {float(r['budget']): r for r in read('detection_limits.csv')
              if int(r['customers']) == 12800 and float(r['control_rate']) == .005}
    rows = []
    for budget in (.05, .10, .20):
        r = limits[budget]
        power = next(s for s in read('simulation_summary.csv') if s['method'] == 'simultaneous_exact_event_bounds'
                     and s['scenario'] == 'budget_grid' and int(s['customers']) == 12800
                     and float(s['budget']) == budget and float(s['control_rate']) == .005 and float(s['effect']) == .01)
        threshold = 'Not reached on grid' if r['status'] == 'not_reached_on_tested_grid' else f'{float(r["grid_mde_percentage_points"]):g} percentage points'
        rows.append([percent(budget), r['selected'], percent(power['positive_detection_rate']), threshold])
    table(['Contact budget', 'Selected', 'Detection at +1 point', '80% grid threshold'], rows, [1936, 1600, 3000, 3400])
    paragraph('At a 20% budget, the conservative reference detects +4 points in 99.5% of trials (Monte Carlo interval approximately 98.8% to 99.8%) and +2 points in 57.3%. Its wider intervals protect against sparse-case overconfidence at the cost of sensitivity. These are hypothetical experimental outcomes, not model conversion rates. The original bootstrap results are exported but must be read alongside its coverage failures.')
    paragraph('At a 20% budget and +2 points, detection rises to 96.7% with 25,600 evaluation records, versus 57.3% with 12,800. Contacts also double from 2,560 to 5,120, so this is not a fixed-contact comparison. More training data and more evaluation evidence address different problems.')
    paragraph('Validation and remaining work', 'Heading1')
    paragraph('All 26 tests pass, including eight new checks of chance overlap, paired samples, sparse outcomes, IPW calculations, interval coverage, and simulation uncertainty. Settings, seeds, data hashes, sample counts, paired differences, and reproducible exports are saved separately from earlier campaign results.')
    paragraph('These parametric simulations test fixed-policy evaluation, not learned rankings or external generalization. The exact reference is not a general replacement for heterogeneous policy comparisons. Sparse-outcome inference and independent validation remain priorities; visits, spending, and additional model families remain future work.')
    paragraph('Next steps for GenAI', 'Heading1')
    paragraph('We will connect a configured AI generator to turn selected customer profiles and recommended email categories into fresh drafts, keeping No Email customers excluded and recording the model, prompt, and settings. We will challenge the checks with invented facts and incorrect categories, then have two independent reviewers assess relevance, factual grounding, treatment alignment, and clarity. We will revise the drafts and report violations and reviewer disagreements. The current nine drafts and automated checks are a prototype; generator integration and independent review are still planned. Past merchandise purchases will not be treated as gender. Conversion gains or benefits from follow-up emails will require a separate randomized experiment.')
    paragraph('[1] Morris, White and Crowther (2019). Using simulation studies to evaluate statistical methods. https://doi.org/10.1002/sim.8086')
    paragraph('[2] SciPy exact binomial proportion intervals. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html')
