"""Build Progress Report 5 from Report 4 without changing retained package parts.

Requires lxml in the document-authoring runtime. Run with the bundled document
Python and render with the Documents skill; the scientific project environment
does not need this optional authoring dependency.
"""
from copy import deepcopy
import csv
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT/'docs/Progress_Reports/Progress_Report_4.docx'
OUTPUT = ROOT/'docs/Progress_Reports/Progress_Report_5.docx'
RESULTS = ROOT/'results/evaluation'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w':W}


def tag(name):
    return f'{{{W}}}{name}'


def read(name):
    with (RESULTS/name).open() as file:
        return list(csv.DictReader(file))


def number(value, digits=1):
    return f'{float(value):.{digits}f}'


def run():
    with ZipFile(REFERENCE) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    document = E.fromstring(original['word/document.xml'])
    body = document.find('w:body',NS)
    section = deepcopy(body.find('w:sectPr',NS))
    patterns = {}
    for p in body.findall('w:p',NS):
        style = p.find('w:pPr/w:pStyle',NS)
        name = style.get(tag('val')) if style is not None else 'Normal'
        if ''.join(p.itertext()).strip() and name not in patterns:
            patterns[name] = deepcopy(p)
    if 'Heading2' not in patterns:
        patterns['Heading2'] = deepcopy(patterns['Heading1'])
        patterns['Heading2'].find('w:pPr/w:pStyle',NS).set(tag('val'),'Heading2')
    table_pattern = deepcopy(body.find('w:tbl',NS))
    for child in list(body):
        body.remove(child)

    def paragraph(text,style='Normal',new_page=False):
        p = deepcopy(patterns[style])
        for child in list(p):
            if child.tag != tag('pPr'):
                p.remove(child)
        ppr = p.find('w:pPr',NS)
        if ppr is None:
            ppr = E.SubElement(p,tag('pPr'))
        if new_page:
            E.SubElement(ppr,tag('pageBreakBefore'))
        r = E.SubElement(p,tag('r'))
        t = E.SubElement(r,tag('t')); t.text=text
        body.append(p)

    def table(headers,rows,widths):
        tbl=deepcopy(table_pattern)
        source_rows=tbl.findall('w:tr',NS)
        prototypes=[deepcopy(source_rows[0].find('w:tc',NS)),deepcopy(source_rows[1].find('w:tc',NS))]
        for row in source_rows:
            tbl.remove(row)
        grid=tbl.find('w:tblGrid',NS)
        for child in list(grid): grid.remove(child)
        for width in widths:
            E.SubElement(grid,tag('gridCol'),{tag('w'):str(width)})
        for i,values in enumerate([headers,*rows]):
            tr=E.SubElement(tbl,tag('tr'))
            props=E.SubElement(tr,tag('trPr'))
            E.SubElement(props,tag('cantSplit'))
            if i == 0: E.SubElement(props,tag('tblHeader'))
            for j,value in enumerate(values):
                tc=deepcopy(prototypes[0 if i == 0 else 1])
                pr=tc.find('w:tcPr',NS)
                pr.find('w:tcW',NS).set(tag('w'),str(widths[j]))
                p=tc.find('w:p',NS)
                for child in list(p):
                    if child.tag != tag('pPr'):p.remove(child)
                pp=p.find('w:pPr',NS)
                align=pp.find('w:jc',NS)
                if align is None:align=E.SubElement(pp,tag('jc'))
                align.set(tag('val'),'left' if j == 0 else 'center')
                r=E.SubElement(p,tag('r'));rp=E.SubElement(r,tag('rPr'))
                E.SubElement(rp,tag('sz'),{tag('val'):'20'})
                if i == 0:E.SubElement(rp,tag('b'))
                E.SubElement(r,tag('t')).text=str(value)
                tr.append(tc)
            tbl.append(tr)
        body.append(tbl)
        paragraph('')

    labels={'no_contact':'No contact','random_mens_mean':'Random Men mean','random_womens_mean':'Random Women mean',
            'logistic_response':'Logistic response','logistic_uplift':'Logistic T learner',
            'tree_response':'Tree response','tree_uplift':'Tree T learner'}
    summary=read('policy_summary.csv')
    lookup={(r['policy'],float(r['budget'])):r for r in summary}

    paragraph('Reliable Customer Targeting Under Limited Marketing Campaign Budgets','Title')
    paragraph('Srajan Jain and Manav Mendonca')
    paragraph('We completed held-out budget evaluation, repeated-training reliability analysis, customer segmentation, and grounded campaign drafts. The main finding is that uplift targeting has positive point estimates at all three budgets, but neither T learner clearly improves on random Men\'s Email after accounting for test uncertainty. Customer selection also changes substantially across training samples. Logistic response targeting has the strongest held-out estimates; this comparison does not establish a deployment winner.')
    paragraph('Step 4 Evaluate the actual budget decision','Heading1')
    paragraph('For each customer, we estimate the purchase probability under No Email and each email. We subtract the No Email prediction from each email prediction, choose the larger estimated uplift, and rank customers by it. We contact only customers with strictly positive uplift, up to one shared cap; everyone else receives No Email. At 5%, 10%, and 20%, the 12,800-customer test cohort permits 640, 1,280, and 2,560 total contacts, respectively.')
    paragraph('We kept the two Step 3 model configurations fixed and refitted them on all 51,200 development customers. No test outcome was used to choose hyperparameters, a model winner, or cluster count. The protocol records seeds, assignment probabilities, resampling settings, and source hashes. The existing treatment-by-conversion split was retained; earlier descriptive EDA had already used all records.')
    paragraph('Held out incremental conversions','Heading2')
    rows=[]
    for label in labels:
        values=[]
        for budget in [.05,.1,.2]:
            r=lookup[label,budget]
            values.append(f'{number(r["estimated_incremental_conversions"])} [{number(r["ci_lower"])}, {number(r["ci_upper"])}]')
        rows.append([labels[label],*values])
    table(['Policy','5% budget','10% budget','20% budget'],rows,[2916,2340,2340,2340])
    paragraph('Values are estimated additional conversions relative to No Email, followed by pointwise 95% confidence intervals. Every contacting policy used exactly its full shared allowance: 5%, 10%, or 20%. No contact selected 0%. Random baselines average 100 seeded rankings, using the same selected customers for both email types; their narrower intervals describe an averaged random policy rather than a single realized random campaign.')

    paragraph('How we estimate incremental value','Heading1',True)
    paragraph('Each customer received only one historical action, so we cannot observe the outcome under every recommended action. We use inverse-probability weighting with the documented randomized assignment probability of 1/3 per action. The policy score is 1[observed action equals recommended action] times conversion divided by 1/3. Subtracting the same customer\'s No Email score gives a paired incremental score; customers assigned No Email by the policy cancel exactly. Summing scores estimates incremental conversions across the cohort, not directly observed additional purchases [1, 2].')
    paragraph('We resample test customer rows 2,000 times and reuse the identical draws for all strategies. Predictions, rankings, and recommendations stay fixed. Intervals therefore measure limited-test uncertainty conditional on the fitted policies. They exclude model-training variation and new-cohort reranking. We do not stratify the resamples by observed conversion, because fixing the purchase count would suppress the uncertainty of interest. Intervals are approximate and pointwise, with no adjustment for multiple comparisons.')
    paragraph('Paired differences from random Men Email','Heading2')
    contrasts=read('paired_comparisons.csv')
    table(['Policy','Budget','Difference','95% interval'],[
        [labels[r['policy']],f'{float(r["budget"])*100:.0f}%',number(r['difference_incremental_conversions']),f'[{number(r["ci_lower"])}, {number(r["ci_upper"])}]']
        for r in contrasts if r['reference']=='random_mens_mean' and r['policy'] in ('logistic_response','logistic_uplift','tree_uplift')
    ],[2916,1140,2220,3660])
    paragraph('All six T learner differences include zero: the observed estimates do not distinguish either uplift policy from the averaged random Men\'s Email baseline. Logistic response has positive pointwise differences at all budgets, but the 5% lower bound is only 0.7 conversions. Multiple comparisons, sparse outcomes, and historically exposed EDA limit how strongly we can generalize this result. At 20%, logistic uplift minus logistic response is -15 conversions with interval [-42, 12], so a causal ranking rule does not automatically improve this fitted model.')
    paragraph('The test set contains only 115 purchases. At 5%, logistic uplift has five matched contacted purchases and three selected control purchases, giving 3 times (5 minus 3) = 6. A random 640-customer subset contains about 5.75 observed purchases on average across historical actions. That illustrates sparse evaluation data; it is not a prediction of targeting performance. Predicting no purchase for everyone gives 99.10% test accuracy and is therefore an unhelpful primary metric.')

    paragraph('Separate email results and reliability','Heading1',True)
    paragraph('We export two complementary treatment views. First, each email is ranked separately against the shared control on its binary comparison cohort: 8,522 customers for Men\'s Email and 8,539 for Women\'s Email. Binary assignment probabilities are 1/2 conditional on these two arms. At 20%, logistic treatment-control uplift estimates are 26 [6, 44] for Men and 8 [-8, 26] for Women. Tree estimates are 16 [0, 32] and 18 [0, 36]. These are separate hypothetical targeting comparisons and must not be added into a combined budget policy.')
    paragraph('Second, the combined policy is decomposed by its assigned email. At 20%, logistic uplift allocates 2,132 Men\'s Emails and 428 Women\'s Emails, exactly 2,560 combined contacts. Their estimated contributions are 33 [6, 63] and 3 [0, 9], which sum to 36. Tree uplift allocates 2,398 and 162 contacts, contributing 24 [0, 48] and 3 [-6, 15], which sum to 27. A zero component can result from no observed contributing purchases and does not prove that a treatment has no effect.')
    paragraph('Qini and normalized AUUC are supplementary ranking diagnostics [3, 4]. Logistic normalized Qini is 0.0279 for Men and 0.1206 for Women; tree values are 0.0276 and 0.1105. These whole-curve summaries do not establish the value of the combined recommendation policy at a fixed budget.')
    paragraph('Step 5 Measure reliability explicitly','Heading1')
    paragraph('We draw 20 bootstrap development samples within action arms, allowing purchase counts to vary. Both model families use the same training samples. Every refit scores the same 12,800 evaluation customers, with the policy tie seed fixed. We compare all 190 refit pairs. Selection overlap is intersection size divided by the smaller selected count; all refits used the full cap here. Jaccard overlap uses intersection divided by union. Email agreement is measured only among customers selected by both refits, avoiding inflation from the many unchanged No Email assignments.')
    stability=read('reliability_summary.csv')
    table(['Uplift policy','Budget','Selection overlap','Shared email agreement','Jaccard'],[
        [labels[r['policy']].replace(' T learner',' uplift'),f'{float(r["budget"])*100:.0f}%',f'{float(r["selection_overlap_mean"])*100:.1f}%',f'{float(r["treatment_agreement_shared_mean"])*100:.1f}%',f'{float(r["selection_jaccard_mean"])*100:.1f}%']
        for r in stability if r['policy'].endswith('uplift')
    ],[2580,1080,2220,2616,1440])
    paragraph('Logistic uplift is more stable than tree uplift in these runs, but even its 20% selections overlap only 45.3% on average. At 5%, approximately 215 of 640 selected customers remain shared for logistic uplift and 99 for the tree. These results make stability a substantive project finding. Pair quantiles are descriptive spreads across dependent pairs, not independent-observation confidence intervals.')

    paragraph('Training variation and customer segmentation','Heading1',True)
    paragraph('Performance uncertainty has two separately reported sources. Fixed-policy test bootstrap standard deviations for logistic uplift are 8.5, 12.3, and 14.3 conversions; tree values are 9.0, 9.4, and 13.2. Across training resamples, logistic uplift values range from 0 to 21, 0 to 36, and 9 to 42; their standard deviations are 5.7, 9.0, and 8.9. Tree ranges are 0 to 15, 0 to 18, and 3 to 33, with standard deviations 4.5, 5.2, and 7.9. Training ranges are not confidence intervals. They describe how the fitted strategy changes on the same finite test cohort.')
    paragraph('Response targeting also varies across refits. Its mean selection overlaps are 50.4%, 56.7%, and 62.5% for logistic regression and 18.5%, 28.0%, and 42.9% for the tree. The preferred email across all customers agrees 79.3% for logistic regression and 63.6% for the tree; selected-customer agreement is reported separately. Random-allocation variation is another distinct quantity: a single random Men\'s policy at 5% has an empirical allocation standard deviation of 5.4 conversions across 100 rankings. None of these spreads is combined into a single purported uncertainty measure.')
    paragraph('Step 6 Develop segmentation around targeting','Heading1')
    paragraph('We fit K means only on development attributes, using log purchase history, scaled numeric and binary inputs, and one-hot categories. We compare 2 through 5 clusters. The prespecified rule chooses the largest silhouette among candidates with at least 5% of customers per cluster and mean bootstrap adjusted Rand index of at least 0.75. Five clustering resamples per candidate are scored on the same fixed development customers; labels are compared with a permutation-invariant measure.')
    cluster_rows=read('cluster_comparison.csv')
    table(['Clusters','Silhouette','Smallest share','Mean stability ARI','Eligible'],[
        [r['k'],number(r['silhouette'],3),f'{float(r["min_cluster_fraction"])*100:.1f}%',number(r['bootstrap_ari_mean'],3),'Yes' if r['eligible']=='True' else 'No'] for r in cluster_rows
    ],[1140,1800,2040,3156,1800])
    paragraph('Two clusters are selected: development sizes are 28,043 (54.8%) and 23,157 (45.2%). The first includes customers with past Women\'s merchandise purchases, of whom 18.3% also bought Men\'s merchandise; median recency is 5 months and past-year spending is $184.23. The second consists of past Men\'s merchandise purchasers without recorded Women\'s merchandise purchases; medians are 6 months and $136.09. These are purchase-history segments, never gender categories. Test proportions are 55.8% and 44.2%.')
    paragraph('The selected two-cluster solution has only moderate stability: its minimum resample ARI is 0.640. Three and five clusters are more stable but have lower silhouette under our rule. Four clusters fail the stability gate. Five clustering resamples provide an initial diagnostic, not a definitive choice of customer taxonomy.')

    paragraph('Cluster response and grounded campaign drafts','Heading1',True)
    paragraph('After freezing clusters, we describe randomized treatment responses within the test clusters. The table shows observed counts and conversion rates with Wilson 95% intervals. Separate IPW cluster-effect intervals are also exported; they estimate hypothetical email assignment to all customers in a cluster, without a campaign cap, and are not our budget-policy results.')
    response_rows=read('cluster_test_responses.csv')
    action_names={'Mens E-Mail':'Men Email','Womens E-Mail':'Women Email','No E-Mail':'No Email'}
    table(['Cluster and action','Customers','Purchases','Conversion rate and 95% interval'],[
        [f'{r["cluster"]} {action_names[r["action"]]}',r['customers'],r['purchases'],
         f'{float(r["conversion_rate"])*100:.2f}% [{float(r["rate_ci_lower"])*100:.2f}, {float(r["rate_ci_upper"])*100:.2f}]'] for r in response_rows
    ],[2580,1440,1440,4476])
    paragraph('Cluster 0 has higher observed purchase rates under both emails than under No Email. Cluster 1 has similar Women\'s Email and control rates. Its Women\'s Email IPW estimate is only 3 additional conversions, with interval [-24.1, 30]. These small counts do not establish customer-level effects, nor that cluster-specific targeting improves performance.')
    paragraph('We tested that latter question separately on the five shared development folds. A fresh clustering pipeline is fitted inside each fold, and one-hot cluster membership is added to the same logistic arm models. Incremental values change from 12 to 9 at 5%, 21 to 21 at 10%, and 42 to 33 at 20%. Paired cluster-minus-original intervals are [-9, 0], [0, 0], and [-24.1, 6]. There is no demonstrated improvement, so we retain clusters for explanation. Membership largely repeats an existing purchase flag; changing regularized encoding can still change predictions. The globally chosen development k makes this an exploratory ablation rather than fully nested selection.')
    paragraph('Develop GenAI around the validated workflow','Heading2')
    paragraph('We produced three stages of draft copy for each of three actual customers selected by logistic uplift at the 20% budget. Example profiles were chosen by recorded attributes and recommendation, without consulting their conversion outcomes. Prompts document all seven attributes, their meanings, and the recommended email; mens and womens mean past merchandise purchases. An explicit fact ledger links asserted customer facts to source values. The generator interface accepts a provider callable; these nine drafts were generated by the Codex assistant in this session, with no external API call or email delivery.')
    paragraph('For a customer with past purchases in both merchandise categories, the model recommends Men\'s Email. The first draft says: "Your past purchases include men\'s merchandise. Take a look at men\'s merchandise when you next shop." For a customer with past Women\'s merchandise purchases, the first draft says: "Your past purchases include women\'s merchandise. Explore women\'s merchandise whenever you next shop." A ten-month recency motivates a re-engagement tone in the second profile without exposing that exact history in the message.')

    paragraph('What the additions establish and what remains','Heading1',True)
    paragraph('All nine drafts pass limited checks for declared facts, category alignment, a relevant attribute, and prohibited claims. Undeclared or paraphrased inventions can evade these checks. Author review finds modest personalization and repetition in later stages. A 1 to 5 relevance, grounding, alignment, clarity, and sequence rubric is available; independent reviewer scores and agreement are pending. Product-specific copy requires an approved catalog and offers.')
    paragraph('Only the first draft stage corresponds to the evaluated one-email action. Stages 2 and 3 are unsent sequence concepts that require a new randomized study, separately specified timing and total-contact budget. Hillstrom records a two-week outcome after its historical email experiment [2]. It cannot establish that our generated copy or a new sequence increases conversion. Message-quality checks remain separate from the measured targeting results.')
    paragraph('Implementation and validation','Heading2')
    paragraph('New code covers paired IPW uncertainty, treatment comparisons, repeated-training agreement, clustering and its ablation, grounded drafts, and figures. Optional cluster encoding preserves the default baseline configuration. All 18 tests pass, checking known synthetic policy effects, one cap across opposite emails, No Email cancellation, paired resampling, training purchase-count variation, outcome-free clustering, and rejection of wrong message facts or treatment alignment.')
    paragraph('Outputs include 21 policy-budget summaries, 63 paired contrasts, 12 binary treatment-budget comparisons, 24 model-policy treatment components, 20 training refits per family, 2,280 stability records, and four cluster candidates. Aggregate results and drafts are reviewable; models and full per-customer exports remain ignored local artifacts. requirements-evaluation.txt records the observed Python 3.14 environment; fresh installation is unverified.')
    paragraph('Limitations include sparse purchases, approximate inference after an outcome-stratified split, prior full-data EDA, and multiple comparisons. No model was adjusted after seeing test results or selected for deployment. Tuned models, X learners, and ensembles need new untouched validation evidence. Doubly robust evaluation can follow the validated IPW checks; it is not a result in this report [1]. Monetary ROI requires campaign costs.')
    paragraph('We retain response and random targeting as comparators, report stability alongside value, and use clusters for explanation. Steps 4 and 5 have measured results; Step 6 provides segmentation and grounded message prototypes. No GitHub commit or push has been made.')
    paragraph('References','Heading2')
    paragraph('[1] Dudik, Langford, and Li. Doubly Robust Policy Evaluation and Learning. ICML 2011. https://arxiv.org/abs/1103.4601')
    paragraph('[2] scikit uplift 0.5.1. Hillstrom dataset documentation. https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html')
    paragraph('[3] scikit uplift 0.5.1. Normalized Qini area documentation. https://www.uplift-modeling.com/en/latest/api/metrics/qini_auc_score.html')
    paragraph('[4] scikit uplift 0.5.1. Normalized uplift area documentation. https://www.uplift-modeling.com/en/latest/api/metrics/uplift_auc_score.html')
    body.append(section)
    # Cloned paragraph tracking identifiers must not repeat across new content.
    for element in body.iter():
        for attribute in list(element.attrib):
            if E.QName(attribute).localname in ('paraId','textId','rsidR','rsidRDefault','rsidP','rsidRPr'):
                del element.attrib[attribute]
    header=E.fromstring(original['word/header1.xml'])
    texts=header.findall('.//w:t',NS)
    for text in texts:
        if text.text == '4':text.text='5'
        elif text.text == '07':text.text='12'
    settings=E.fromstring(original['word/settings.xml'])
    update=settings.find('w:updateFields',NS)
    if update is None:update=E.SubElement(settings,tag('updateFields'))
    update.set(tag('val'),'true')
    changed={'word/document.xml':E.tostring(document,xml_declaration=True,encoding='UTF-8',standalone=True),
             'word/header1.xml':E.tostring(header,xml_declaration=True,encoding='UTF-8',standalone=True),
             'word/settings.xml':E.tostring(settings,xml_declaration=True,encoding='UTF-8',standalone=True)}
    with ZipFile(OUTPUT,'w') as archive:
        for name,content in original.items():archive.writestr(name,changed.get(name,content))
    with ZipFile(OUTPUT) as archive:
        assert all(archive.read(name)==content for name,content in original.items() if name not in changed)
    print(f'Saved {OUTPUT}; only document, header, and updateFields setting changed.')


if __name__=='__main__':
    run()
