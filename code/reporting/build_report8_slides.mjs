/** Editable Report 8 PowerPoint with source-backed data and speaker notes.
 * Uses the bundled Artifact Tool authoring runtime, not the scientific .venv.
 * Set REPORT8_SLIDES_WORKSPACE to a private build folder before running.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const ROOT = process.cwd();
const runtime = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime');
const modules = process.env.RUNTIME_NODE_MODULES || path.join(runtime, 'dependencies/node/node_modules');
process.env.RUNTIME_NODE_MODULES = modules;
const runtimeRequire = createRequire(path.join(modules, '__runtime__.cjs'));
const { Presentation, PresentationFile } = await import(pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href);
const SKILL = process.env.PRESENTATIONS_SKILL_DIR || path.join(os.homedir(),
  '.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations');
const { resolvePresentationFont, applyPresentationChartFont, makeNativeBulletParagraphs, finalizePresentation } =
  await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const WORK = process.env.REPORT8_SLIDES_WORKSPACE;
if (!WORK || !path.isAbsolute(WORK)) throw new Error('Set REPORT8_SLIDES_WORKSPACE to an absolute private build folder.');
const REVISION = process.env.REPORT8_SLIDES_REVISION || 'v1';
const OUTPUT = path.join(WORK, 'output', `Progress_Report_8_Presentation_${REVISION}.pptx`);
const PREVIEWS = path.join(WORK, `previews-${REVISION}`);
await fs.mkdir(path.join(WORK, 'build'), { recursive: true });
await fs.mkdir(path.join(WORK, 'output'), { recursive: true });
await fs.mkdir(PREVIEWS, { recursive: true });

function parseCSV(text) {
  const rows = []; let row = [], value = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') { value += '"'; i++; } else quoted = !quoted;
    } else if (c === ',' && !quoted) { row.push(value); value = ''; }
    else if (c === '\n' && !quoted) { row.push(value.replace(/\r$/, '')); rows.push(row); row = []; value = ''; }
    else value += c;
  }
  if (value || row.length) { row.push(value); rows.push(row); }
  const [header, ...body] = rows;
  return body.filter(r => r.some(Boolean)).map(r => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ''])));
}
const csv = async name => parseCSV(await fs.readFile(path.join(ROOT, name), 'utf8'));
const policies = await csv('results/evaluation/policy_summary.csv');
const paired = await csv('results/evaluation/paired_comparisons.csv');
const chance = await csv('results/report8/chance_reference.csv');
const training = await csv('results/report8/training_size_summary.csv');
const sims = await csv('results/report8/simulation_summary.csv');
const ablation = await csv('results/evaluation/development_cluster_ablation.csv');
const drafts = JSON.parse(await fs.readFile(path.join(ROOT, 'results/evaluation/message_drafts.json'), 'utf8'));
const policy = (name, budget) => policies.find(r => r.policy === name && +r.budget === budget);
const comparison = name => paired.find(r => r.policy === name && r.reference === 'random_mens_mean' && +r.budget === .2);
const pct = x => `${(+x * 100).toFixed(1)}%`;
const fmt = x => Number.isInteger(+x) ? String(+x) : (+x).toFixed(2);
const C = { ink: '#142735', blue: '#245A81', gray: '#7A838B', muted: '#52616D', pale: '#E7EEF4',
  line: '#D9E1E7', white: '#FFFFFF', green: '#687C42', amber: '#8A5B21' };
const FONT = resolvePresentationFont();
const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const manifest = [];
const base = 'https://github.com/srajan2627/budget-aware-customer-targeting/blob/fa34769dc7b555779fa15230907886237046c881/';
const external = {
  hillstrom: 'https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html',
  simulation: 'https://doi.org/10.1002/sim.8086',
  exact: 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html',
  ipw: 'https://arxiv.org/abs/1103.4601',
};

function text(slide, value, x, y, w, h, size = 28, color = C.ink, bold = false, name = 'body') {
  const shape = slide.shapes.add({ geometry: 'textbox', name,
    position: { left: x, top: y, width: w, height: h }, fill: 'none',
    line: { fill: 'none', width: 0 } });
  shape.text = value;
  shape.text.style = { typeface: FONT, fontSize: size, bold, color,
    insets: 0, wrap: 'square', autoFit: 'none', verticalAlignment: 'top' };
  return shape;
}
function bullets(slide, items, x = 72, y = 190, w = 1136, h = 400, size = 30) {
  const shape = text(slide, '', x, y, w, h, size);
  shape.text = makeNativeBulletParagraphs(items, { marginLeftPoints: 20, hangingPoints: 10, spaceAfterPoints: 17 });
  return shape;
}
function add(title, subtitle, notes, sources = [], foot = '') {
  const slide = p.slides.add(); slide.background.fill = C.white;
  const number = p.slides.items.length;
  text(slide, title, 64, 42, 1152, 64, 44, C.ink, true, 'slide-title');
  if (subtitle) text(slide, subtitle, 66, 120, 1146, 58, 25, C.muted, false, 'subtitle');
  if (foot) text(slide, foot, 66, 650, 1100, 48, 18, C.muted, false, 'source-caption');
  text(slide, String(number).padStart(2, '0'), 1180, 666, 40, 24, 18, C.muted, false, 'slide-number');
  slide.speakerNotes.text = notes.trim() + '\n\nSources\n' + sources.map(s => s.startsWith('http') ? s : base + s).join('\n');
  manifest.push({ slide: number, title, sources, notes: notes.trim() });
  return slide;
}
function table(slide, values, y = 188, height = 370, widths = [400, 250, 486], size = 24, left = 66) {
  const t = slide.tables.add({ rows: values.length, columns: values[0].length,
    left, top: y, width: widths.reduce((a, b) => a + b, 0), height, columnWidths: widths, values });
  t.borders.assign({ style: 'solid', fill: C.line, width: 1 });
  t.cells.block({ row: 0, column: 0, rowCount: values.length, columnCount: values[0].length }).assign({
    textStyle: { typeface: FONT, fontSize: size, color: C.ink },
    fill: C.white, margins: { top: 10, bottom: 8, left: 14, right: 12 }, anchor: 'center' });
  for (let r = 0; r < values.length; r++) {
    t.rows[r].height = height / values.length;
    for (let c = 0; c < values[0].length; c++) {
      t.getCell(r, c).text.style = { typeface: FONT, fontSize: size, color: C.ink,
        bold: r === 0, alignment: c === 0 ? 'left' : 'center', verticalAlignment: 'middle', autoFit: 'none' };
      if (r === 0) t.getCell(r, c).fill = C.pale;
    }
  }
  return t;
}
function chart(slide, kind, config) {
  const c = slide.charts.add(kind, { chartFill: C.white, plotAreaFill: C.white,
    lineOptions: { smooth: false },
    titleTextStyle: { fontSize: 24, fill: C.ink, bold: false },
    hasLegend: true, legend: { position: 'bottom', overlay: false, textStyle: { fontSize: 21, fill: C.ink } },
    ...config });
  applyPresentationChartFont(c, { fontFamily: FONT });
  return c;
}
const axis = (title, extra = {}) => ({ title: { text: title, textStyle: { fontSize: 21, fill: C.ink } },
  textStyle: { fontSize: 20, fill: C.muted }, line: { fill: C.line, width: 1 }, ...extra });
// Chart snapshots keep six decimal places, well beyond displayed precision,
// so the values remain portable within Excel's fifteen-significant-digit limit.
const lineSeries = (name, values, color) => ({ name, values: values.map(v => Math.round(+v * 1e6) / 1e6), fill: color, smooth: false,
  line: { fill: color, width: 3 }, marker: { symbol: 'circle', size: 8, fill: color } });

// 1. Minimal cover.
{
  const s = p.slides.add(); s.background.fill = C.ink;
  text(s, 'Progress Report 8', 72, 62, 1100, 50, 30, '#BBD3E5', false, 'report-label');
  text(s, 'Reliable customer targeting\nunder limited budgets', 72, 205, 1120, 195, 62, C.white, true, 'deck-title');
  text(s, 'Srajan Jain and Manav Mendonca\nCIS 631 Machine Learning', 76, 480, 1090, 85, 28, C.white);
  text(s, '9 November 2026\nResearch results through 4 November 2026', 76, 609, 1090, 62, 21, '#BBD3E5');
  const notes = 'Our project helps a business decide whom to email when it cannot contact everyone. We compare likely buyers with customers whose buying chance might improve because of the email. Report 8 brings together the earlier targeting results and our new checks of reliability and uncertainty. The main modeling workflow works, but fresh AI message generation and independent message review remain unfinished. We will explain the results, how much of the project is implemented, and the exact next deliverables. This presentation uses the results reported on November 4. We have not sent a new campaign.';
  s.speakerNotes.text = notes + '\n\nSource\n' + base + 'docs/Progress_Reports/Progress_Report_8.docx';
  manifest.push({ slide: 1, title: 'Progress Report 8', notes, sources: ['docs/Progress_Reports/Progress_Report_8.docx'] });
}

// 2. Context and implemented scope.
{
  const s = add('Project purpose and current scope', 'A limited contact budget makes customer selection a decision problem',
    `We use records from an old randomized email experiment with 64,000 customers. About one third received Men's Email, one third received Women's Email, and one third received no email. We know whether customers bought within two weeks. We use 51,200 development records and 12,800 evaluation records. The models use only information recorded before the email. We excluded later visits, purchases and spending from the inputs. The current project evaluates purchase conversion. Visits and spending remain possible extensions. The labels mens and womens describe past merchandise purchases, not customer gender.`,
    ['README.md', 'code/data/preprocessing.py', external.hillstrom],
    'Historical randomized email data. Evaluation audience: 12,800 customers with 115 purchases.');
  text(s, '64,000', 72, 206, 570, 90, 66, C.blue, true);
  text(s, 'historical customer records', 76, 302, 540, 45, 30);
  text(s, '51,200 development records\n12,800 evaluation records\nTwo-week purchase outcome', 76, 380, 560, 175, 29);
  bullets(s, ['Purchase estimates for three actions', 'One budget across both email types', 'Reliability and evaluation diagnostics', 'GenAI message prototype'], 695, 215, 490, 355, 29);
}

// 3. Operational policy and combined cap.
{
  const s = add('The budget decision', 'Each selected customer receives one recommended email',
    `The model estimates purchase chances with Men's Email, Women's Email and No Email. For uplift targeting we subtract the No Email estimate from each email estimate. We choose the email with the larger improvement, rank customers by that improvement and contact only positive-uplift customers up to the cap. Everyone else receives No Email. The cap covers both email types together. At 20%, logistic uplift recommends 2,132 Men's Emails and 428 Women's Emails. Their total is 2,560. We do not give each email type its own 20% budget. Purchase-likelihood targeting uses the same fitted email-arm models but ranks the higher purchase probability rather than the improvement.`,
    ['code/targeting/policies.py', 'code/models/baselines.py', 'results/evaluation/policy_summary.csv'],
    'Contact limits are audience fractions. Monetary costs and profit remain unmeasured.');
  bullets(s, ['Estimate email benefit against No Email', 'Choose the higher benefit and rank customers', 'Select positive benefit within the shared cap', 'Recommend No Email to everyone else'], 72, 190, 660, 365, 29);
  table(s, [['Budget', 'Total contacts'], ['5%', '640'], ['10%', '1,280'], ['20%', '2,560']], 218, 272, [200, 240], 26, 770);
  text(s, "20% example: 2,132 Men's Emails + 428 Women's Emails = 2,560 contacts", 76, 572, 1120, 50, 26, C.blue, true);
}

// 4. All principal contacting policies with original intervals.
{
  const s = add('Estimated extra purchases by targeting rule', 'Original historical evaluation results compared with No Email',
    `Each entry estimates additional purchases across the 12,800 evaluation customers. Brackets contain the original nominal 95% paired bootstrap intervals. These are historical policy estimates, not sales from a new campaign. No Email has zero incremental benefit because it is the comparison point. Customers can still buy without an email. Logistic likelihood has the highest point estimate at all three budgets. The random baselines average 100 selections, so their uncertainty describes an average random policy rather than one campaign. We later found that sparse outcomes can make some bootstrap intervals misleading. Treat the original intervals as qualified evidence. The models have not established a deployment winner.`,
    ['results/evaluation/policy_summary.csv', 'configs/evaluation.json', 'results/report8/historical_interval_audit.csv'],
    'Brackets: original nominal 95% intervals. Sparse outcomes and multiple comparisons limit inference.');
  const labels = [['random_mens_mean', 'Random Men email mean'], ['random_womens_mean', 'Random Women email mean'],
    ['logistic_response', 'Logistic likelihood'], ['logistic_uplift', 'Logistic uplift'],
    ['tree_response', 'Tree likelihood'], ['tree_uplift', 'Tree uplift']];
  const values = [['Targeting rule', '5% budget', '10% budget', '20% budget'], ...labels.map(([key, label]) =>
    [label, ...[.05, .1, .2].map(b => { const r = policy(key, b); return `${fmt(r.estimated_incremental_conversions)} [${(+r.ci_lower).toFixed(key.startsWith('random') ? 1 : 0)}, ${(+r.ci_upper).toFixed(key.startsWith('random') ? 1 : 0)}]`; })])];
  table(s, values, 190, 378, [395, 247, 247, 247], 24);
  text(s, 'Logistic likelihood leads the point estimates. Evidence remains uncertain.', 72, 592, 1132, 50, 25, C.blue, true);
}

// 5. Paired baseline differences and treatment allocation.
{
  const s = add('Baseline comparisons remain uncertain', "20% shared budget versus the mean random Men's Email policy",
    `A higher point estimate is not enough to establish that one strategy is better. This table shows paired differences, using the same resampled evaluation customers for each strategy. Logistic uplift estimates about 19 extra purchases beyond the average random Men's Email policy, but its interval runs from about minus 4 to plus 44. Tree uplift also includes zero. Logistic likelihood has a positive original interval, which is encouraging, but the rare purchases, multiple comparisons and later coverage concerns limit the conclusion. At 20%, logistic uplift's email contributions are 33 for Men's Email and 3 for Women's Email, summing to 36. Separate treatment-control cohorts are different comparisons and must not be added into a combined-budget result.`,
    ['results/evaluation/paired_comparisons.csv', 'results/evaluation/policy_treatment_components.csv'],
    'Original pointwise paired intervals. Neither uplift model establishes a clear advantage over random Men email.');
  table(s, [['Policy', 'Estimated difference', 'Original 95% interval'],
    ...[['logistic_response', 'Logistic likelihood'], ['logistic_uplift', 'Logistic uplift'], ['tree_uplift', 'Tree uplift']].map(([key, label]) => {
      const r = comparison(key); return [label, `+${(+r.difference_incremental_conversions).toFixed(2)}`, `[${(+r.ci_lower).toFixed(2)}, ${(+r.ci_upper).toFixed(2)}]`];
    })], 202, 260, [420, 330, 386], 27);
  text(s, 'The simpler likelihood baseline remains a serious comparison', 76, 510, 1100, 55, 32, C.blue, true);
  text(s, 'Model agreement and estimated value answer different questions', 76, 575, 1100, 45, 26, C.muted);
}

// 6. Native chance-reference chart.
{
  const s = add('Customer lists overlap more than chance', 'Selection consistency across 20 refits on the same evaluation audience',
    `We retrain on different development samples and score the same customers. Overlap asks how many customers appear in both selected lists, divided by the smaller list size. At 5%, logistic uplift overlaps by 33.6% on average, compared with 15.5% for the tree and 5% for independent random lists. With 640 contacts, that is about 215 shared customers for logistic uplift and 99 for the tree. We average 190 refit pairs, but these pairs share fitted models and are dependent. Above-chance overlap demonstrates consistency, not correct targeting. At 5%, email agreement among shared selections is 86.5% for logistic and 69.1% for the tree. An acceptable business stability threshold remains undefined.`,
    ['results/report8/chance_reference.csv', 'results/evaluation/reliability_summary.csv'],
    'Source: chance_reference.csv. Means across 190 dependent pairs. Agreement does not establish correctness.');
  const values = key => [.05, .1, .2].map(b => +chance.find(r => r.policy === key && +r.budget === b).observed_overlap_mean);
  chart(s, 'scatter', { position: { left: 64, top: 190, width: 805, height: 420 }, title: 'Mean selection overlap',
    scatterOptions: { style: 'lineWithMarkers' }, series: [lineSeries('Logistic uplift', values('logistic_uplift'), C.blue),
      lineSeries('Tree uplift', values('tree_uplift'), C.gray), lineSeries('Random expectation', [.05, .1, .2], C.green)]
        .map(series => ({ ...series, xValues: [.05, .1, .2] })),
    xAxis: axis('Shared contact budget', { min: .05, max: .2, majorUnit: .05, numberFormatCode: '0%' }), yAxis: axis('Selection overlap (%)', { min: 0, max: .6, majorUnit: .2,
      numberFormatCode: '0%', majorGridlines: { fill: C.line, width: 1 } }) });
  text(s, '5% budget', 920, 230, 280, 46, 30, C.blue, true);
  text(s, '640 contacts\n\nAbout 215 shared\nwith logistic uplift\n\nAbout 99 shared\nwith tree uplift', 920, 292, 280, 290, 26);
}

// 7. Two native charts separate consistency and refit value.
{
  const s = add('More training improves logistic consistency', '20% contact budget with matched training samples and fixed model settings',
    `We repeat the models at 25%, 50%, 75% and 100% nominal training draw counts. Both models receive matched samples, and smaller samples use prefixes of the same draws. Logistic overlap rises from 29.8% to 45.3%, while its mean estimated extra purchases move from 24.75 to 28.80, 26.70 and 27.45. So greater consistency does not guarantee a steady increase in value. The tree is less consistent. The 100% bootstrap repeats records and averages about 32,359 unique customers. These are not newly collected customers. The tree also keeps minimum leaf size 200, which changes its effective flexibility with sample size. These averages describe 20 retrained policies and differ from the original fixed logistic uplift estimate of 36.`,
    ['results/report8/training_size_summary.csv', 'results/report8/training_sample_counts.csv'],
    'Source: training_size_summary.csv. Refit means describe training variation, not evaluation confidence intervals.');
  const series = metric => ['logistic_uplift', 'tree_uplift'].map((key, i) => lineSeries(i ? 'Tree uplift' : 'Logistic uplift',
    [.25, .5, .75, 1].map(f => +training.find(r => r.policy === key && +r.budget === .2 && +r.training_fraction === f)[metric]), i ? C.gray : C.blue));
  chart(s, 'line', { position: { left: 52, top: 190, width: 574, height: 417 }, title: 'Selection consistency',
    categories: ['25%', '50%', '75%', '100%'], series: series('selection_overlap_mean'),
    xAxis: axis('Nominal training draw count'), yAxis: axis('Overlap (%)', { min: 0, max: .6, numberFormatCode: '0%', majorUnit: .2,
      majorGridlines: { fill: C.line, width: 1 } }) });
  chart(s, 'line', { position: { left: 650, top: 190, width: 574, height: 417 }, title: 'Mean estimated campaign benefit',
    categories: ['25%', '50%', '75%', '100%'], series: series('training_value_mean'),
    xAxis: axis('Nominal training draw count'), yAxis: axis('Extra purchases', { min: 0, max: 40, majorUnit: 10,
      majorGridlines: { fill: C.line, width: 1 } }) });
}

// 8. Known-truth evaluator stress test.
{
  const s = add('Sparse purchases can undermine confidence intervals', 'A controlled stress test exposes a limitation of the original evaluator',
    `In this hypothetical test, 122 customers receive recommendations. Their purchase chance is 0.5% without email and 1.5% with the recommended email. The true expected incremental benefit is therefore 1.22 purchases. We generate 1,000 repeated trials with historical actions randomized equally. The original nominal 95% bootstrap interval contains that known value in only 44.4% of trials. In 45.8% of trials there are no contributing positive or negative purchase events, and resampling zeros cannot reveal unseen events. The conservative exact reference covers the true value in all 1,000 trials, but its wider intervals sacrifice sensitivity. That observed 100% coverage applies to this scenario. The reference has homogeneous iid assumptions and is not a general solution for every real-data policy comparison. The historical audit also flags two nonempty Women's Email components: 68 logistic-likelihood recommendations at 5% and 122 logistic-uplift recommendations at 10%. Their display intervals are withheld while original exports remain traceable.`,
    ['results/report8/simulation_summary.csv', 'results/report8/historical_interval_audit.csv', 'docs/research/report8_protocol.md', external.simulation, external.exact],
    'Source: 122-contact stress test, 1,000 trials. Conditional simulation coverage.\nHistorical audit flags two nonempty zero-event email components.');
  const sparse = sims.filter(r => r.scenario === 'sparse_122_contacts');
  const vals = ['original_percentile_bootstrap', 'simultaneous_exact_event_bounds'].map(key => +sparse.find(r => r.method === key).coverage_rate);
  chart(s, 'bar', { position: { left: 64, top: 190, width: 780, height: 423 }, title: 'Intervals containing the known benefit',
    categories: ['Original bootstrap', 'Exact reference'], series: [{ name: 'Observed coverage', values: vals, fill: C.blue,
      points: [{ idx: 0, fill: C.blue }, { idx: 1, fill: C.gray }] }], hasLegend: false,
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 110 },
    xAxis: axis('Interval method'), yAxis: axis('Trials covering truth (%)', { min: 0, max: 1.1, numberFormatCode: '0%', majorUnit: .25,
      majorGridlines: { fill: C.line, width: 1 } }),
    dataLabels: { showValue: true, position: 'outEnd', numberFormatCode: '0.0%', textStyle: { fontSize: 27, bold: true, fill: C.ink } } });
  text(s, '1.22', 914, 212, 280, 90, 66, C.blue, true);
  text(s, 'known extra purchases\n\n122 selected customers\n\n45.8% of trials have\nno contributing events', 914, 314, 285, 250, 27);
}

// 9. Detection curves with numeric x-axis.
{
  const s = add('Detection depends on effect size and contact budget', 'Conservative reference with 12,800 hypothetical evaluation records and 0.5% control chance',
    `Detection means that the lower bound of a two-sided 95% interval is above zero. We tested absolute email benefits from zero to four percentage points. The graph uses the conservative reference. We require the Monte Carlo 95% lower bound on detection to reach 80% before a tested effect qualifies. At 5% and 10% budgets, the tested grid does not reach that criterion. At 20%, the smallest qualifying tested effect is four points, with 99.5% detection. One percentage point would mean a change from 0.5% to 1.5%, not a 1% relative increase. The curves assess a fixed policy under hypothetical assumptions, not model performance. At two points and 20% budget, increasing evaluation size to 25,600 raises detection from 57.3% to 96.7%, but it doubles contacts as well.`,
    ['results/report8/detection_limits.csv', 'results/report8/simulation_summary.csv', 'configs/report8.json'],
    'Source: 1,000 trials per scenario. Conditional tested-grid thresholds, not universal minimum effects.');
  const effects = [0, .0025, .005, .01, .02, .04];
  const series = [.05, .1, .2].map((budget, i) => ({ ...lineSeries(`${budget * 100}% budget`, effects.map(effect => +sims.find(r =>
    r.scenario === 'budget_grid' && +r.customers === 12800 && +r.control_rate === .005 && +r.effect === effect && +r.budget === budget &&
    r.method === 'simultaneous_exact_event_bounds').positive_detection_rate), [C.blue, C.green, C.gray][i]), xValues: effects.map(x => x * 100) }));
  series.push({ name: '80% target', values: [.8, .8], xValues: [0, 4], line: { fill: C.muted, width: 1 }, marker: { symbol: 'none' } });
  chart(s, 'scatter', { position: { left: 64, top: 190, width: 820, height: 423 }, title: 'Trials detecting a positive benefit',
    series, scatterOptions: { style: 'lineWithMarkers' }, xAxis: axis('Known benefit (percentage points)', { min: 0, max: 4, majorUnit: 1 }),
    yAxis: axis('Detection (%)', { min: 0, max: 1.05, majorUnit: .2, numberFormatCode: '0%', majorGridlines: { fill: C.line, width: 1 } }) });
  text(s, '20% budget', 922, 240, 280, 45, 29, C.blue, true);
  text(s, '+4 points', 922, 313, 280, 70, 46, C.ink, true);
  text(s, '99.5% detection\n\n5% and 10% budgets\nremain below the\nqualification criterion', 922, 408, 280, 180, 26);
}

// 10. Segmentation evidence, with editable negative values.
{
  const s = add('Clusters explain shopping patterns', 'Adding cluster membership did not demonstrate better targeting',
    `We compared two to five clusters, fitting them only on development attributes. The selection rule considers size, separation and stability. The two selected groups contain 54.8% and 45.2% of development customers. They mainly distinguish past Women's merchandise purchases, sometimes both categories, from Men's-only merchandise purchases. They describe behavior, not gender. Mean resample stability is 0.784 and the minimum is 0.640, so stability is limited. We separately tested adding cluster membership to logistic uplift within development folds. The changes are minus 3, zero and minus 9 estimated purchases at the three budgets. This gives no demonstrated improvement. The comparison is exploratory because the global choice of cluster count is not fully nested. Clusters remain useful for explanation.`,
    ['results/evaluation/cluster_comparison.csv', 'results/evaluation/cluster_profiles.csv', 'results/evaluation/development_cluster_ablation.csv'],
    'Source: development cluster ablation. Point estimates only. Cluster membership describes purchases, never gender.');
  text(s, '54.8% / 45.2%', 76, 224, 470, 70, 48, C.blue, true);
  text(s, 'two development groups\n\nMean stability: 0.784\n\nProfiles support explanation\nMembership adds no\ndemonstrated targeting benefit', 78, 315, 470, 250, 28);
  chart(s, 'bar', { position: { left: 572, top: 190, width: 640, height: 423 }, title: 'Change after adding cluster membership',
    categories: ['5%', '10%', '20%'], series: [{ name: 'Cluster minus original', values: [.05, .1, .2].map(b => +ablation.find(r =>
      r.policy === 'cluster_minus_original' && +r.budget === b).estimated_incremental_conversions), fill: C.gray }],
    hasLegend: false, barOptions: { direction: 'column', gapWidth: 100 }, xAxis: axis('Shared contact budget'),
    yAxis: axis('Change in extra purchases', { min: -12, max: 3, majorUnit: 3, majorGridlines: { fill: C.line, width: 1 } }),
    dataLabels: { showValue: true, position: 'outEnd', numberFormatCode: '0', textStyle: { fontSize: 27, fill: C.ink } } });
}

// 11. GenAI provenance and actual sample.
{
  const s = add('GenAI message prototype', 'Nine existing drafts for three selected customer profiles',
    `The AI message component is a prototype. We have customer contexts, prompts, a provider-neutral generator interface, declared-fact ledgers and nine existing drafts written by the Codex assistant. Running messages.py prepares prompts and audits those checked-in drafts. It does not call an external AI service to generate fresh copy. All nine pass the limited checks, but an undeclared invented claim can escape them. Independent quality ratings remain pending. The example on this slide is an existing first-stage draft. It uses documented merchandise history without assuming gender, discounts or product availability. Only the first stage corresponds to the evaluated one-email policy. Stages two and three remain unsent concepts. We have not measured sales effects from any generated message.`,
    ['code/genai/messages.py', 'results/evaluation/message_drafts.json', 'results/evaluation/message_audit.csv', 'docs/research/message_review_rubric.md'],
    'Existing assistant-authored draft. Independent review and fresh generation remain pending. No emails sent.');
  const example = drafts.drafts[0].messages[0];
  text(s, 'Existing first-stage draft', 76, 205, 600, 48, 29, C.blue, true);
  text(s, example.subject, 76, 277, 570, 74, 33, C.ink, true);
  text(s, example.body, 76, 374, 570, 155, 29);
  bullets(s, ['Prompts and fact ledgers ready', 'Limited audits pass for all nine', 'Fresh generation pending', 'Independent review pending'], 730, 230, 450, 340, 28);
}

// 12. Honest completion by deliverable, without an invented overall percent.
{
  const s = add('How much of the project is complete', 'The core research workflow is implemented. Important delivery and validation work remains.',
    `We should explain completion by deliverable rather than claim an arbitrary overall percentage. The data preparation, fixed models, budget decisions and reliability experiments are implemented. Historical evaluation also works, but the inference has documented limitations. Segmentation supports explanation. The GenAI work remains partial because we have drafts and checks but no connected fresh-generation service and no independent ratings. Suitable sparse-outcome inference and independent external evidence remain open. The project has 26 passing tests and reproducible numerical exports, but a clean-environment installation is still unverified. This is a working research prototype. It has not established deployment readiness or a proven marketing winner. The status rows are not equally weighted tasks or a grading rubric.`,
    ['README.md', 'results/report8/verification.json', 'docs/Progress_Reports/Progress_Report_8.docx'],
    'Completion refers to implementation and evidence. No unweighted checklist is presented as an overall percentage.');
  table(s, [['Workstream', 'Current status', 'Evidence or remaining gap'],
    ['Data and baseline models', 'Implemented', 'Fixed splits and training pipelines'],
    ['Budget decisions and value', 'Implemented', 'Shared caps and historical estimates'],
    ['Reliability research', 'Implemented', 'Refits, chance and simulation studies'],
    ['Customer segmentation', 'Implemented', 'Explanatory profiles and ablation'],
    ['GenAI message delivery', 'Partial', 'Generation and independent review open'],
    ['Inference and validation', 'Open', 'Sparse intervals and fresh evidence'],
    ['Reproduction and reporting', 'Partial', '26 tests pass, clean install pending']], 190, 420, [390, 235, 511], 24);
}

// 13. GenAI follow-up with review deliverables.
{
  const s = add('Next steps for GenAI', 'The next deliverable is reproducible generation with reviewed message quality',
    `The first step is to connect a configured generator and record its model, prompt and settings. Customers assigned No Email must remain excluded. Second, challenge the audit using deliberately invented facts, incorrect merchandise categories and gender assumptions. This checks failure detection rather than just counting passes on easy examples. Third, obtain two independent reviewers, ideally blinded to the generating setup, and use the existing one-to-five rubric for relevance, factual grounding, treatment alignment and clarity. Finally revise the drafts, resolve factual violations and report reviewer disagreements. These tasks evaluate message quality. They do not establish that messages increase conversion. Any new message or multi-email sequence needs a separate randomized experiment to support a sales claim.`,
    ['docs/research/message_review_rubric.md', 'code/genai/messages.py', 'docs/Progress_Reports/Progress_Report_8.docx'],
    'Success evidence: recorded generator run, adversarial cases, two independent reviews, and revised drafts.');
  bullets(s, ['Connect generation and record model, prompt and settings',
    'Challenge the checks with invented facts and wrong categories',
    'Collect two independent reviews using the existing quality rubric',
    'Revise drafts and report violations and disagreements'], 76, 203, 1110, 365, 31);
  text(s, 'New copy or follow-up conversion claims require a separate randomized experiment', 78, 582, 1110, 55, 26, C.blue, true);
}

// 14. Remaining research work and explicit deferred scope.
{
  const s = add('Next steps for validation and submission', 'Remaining work follows the gaps that the experiments exposed',
    `Our statistical priority is a suitable sparse-outcome inference method for heterogeneous real-data policy comparisons. The conservative exact simulation reference is not already a general replacement. We also need independent evidence before confirming model superiority. A comparable second randomized dataset or a new experiment can provide a different type of evidence than controlled simulations. We should reproduce the workflow in a fresh environment and document any installation issues. Finally consolidate the final report, presentation and demonstration, making the code and results traceable and keeping completion claims consistent. The broader proposal also mentions X-learners, ensembles, visit and spending objectives, doubly robust evaluation and another dataset. We have not completed those extensions. Their inclusion should follow the course requirements and agreed scope.`,
    ['docs/research/report8_protocol.md', 'README.md', 'docs/Progress_Reports/Progress_Report_8.docx'],
    'Deferred scope remains explicit. New real-world validation and monetary profit analysis have not been completed.');
  bullets(s, ['Validate sparse-outcome inference for real policy comparisons',
    'Obtain independent evidence before confirming model superiority',
    'Reproduce the workflow in a clean environment',
    'Consolidate the final report and demonstration with traceable evidence'], 76, 195, 1110, 325, 29);
  text(s, 'Unimplemented extensions', 78, 548, 1100, 40, 27, C.blue, true);
  text(s, 'X-learner and ensembles, visits and spending, doubly robust evaluation, second dataset', 78, 594, 1100, 50, 25);
}

// 15. Supported contribution and deployment conditions.
{
  const s = add('Research contribution and completion boundary', 'A working decision process with measured limits on reliability',
    `Our contribution is a functioning budget-aware targeting workflow together with evidence about when its recommendations and measurements become unreliable. The simpler purchase-likelihood baseline has the strongest original estimates. Logistic uplift becomes more consistent with larger learning samples, but estimated value does not steadily increase. The sparse-event simulations reveal that passing software tests does not guarantee trustworthy statistical intervals. Completing GenAI integration and review would strengthen the prototype, while the statistical and independent-validation gaps still need attention. Before real deployment we would need a new randomized campaign, including a suitable outcome window and total contact budget for any sequence. Actual contact costs and profit have not been measured. A research project can succeed by reporting these limitations clearly, even if uplift does not beat the baseline.`,
    ['docs/Progress_Reports/Progress_Report_8.docx', 'results/evaluation/policy_summary.csv', 'docs/research/report8_protocol.md'],
    'No deployment winner, new-message conversion improvement, or monetary profit claim.');
  bullets(s, ['Simple baselines remain competitive', 'Larger samples improve logistic consistency in these runs',
    'Sparse purchases limit confidence in campaign estimates'], 76, 208, 1110, 260, 33);
  text(s, 'Core research workflow implemented\nGenAI integration, review and independent validation remain open', 78, 498, 1110, 110, 31, C.blue, true);
}

// 16. Execution and evidence, all source details in notes.
{
  const s = add('Evidence and execution', 'Saved results and instructions support a reproducible project demonstration',
    `Start from the project root and activate the existing .venv. The Report 8 runner reproduces the chance comparison, historical interval audit, training-size study and simulations in results/report8. It preserves the earlier evaluation outputs. The plotting script recreates four figures, and the test command runs the 26 automated checks. The original targeting results live in results/evaluation, while the report and message review rubric explain their meaning and limits. Report 8 itself is three pages. All figures have saved source data in the repository. A clean-environment installation remains a future check. References below document the historical dataset, policy evaluation background, simulation design and exact binomial interval construction.`,
    ['README.md', 'results/report8/README.md', 'docs/research/report8_protocol.md', external.hillstrom, external.ipw, external.simulation, external.exact],
    'Detailed source links appear in the speaker notes for each slide.');
  text(s, 'python code/evaluation/run_report8.py\npython code/evaluation/plot_report8.py\npython -m unittest discover -s tests -v', 76, 210, 1110, 180, 31, C.blue);
  text(s, 'Original estimates: results/evaluation\nNew diagnostics: results/report8\nMethods and limitations: docs/research/report8_protocol.md\nMessage review: docs/research/message_review_rubric.md', 78, 428, 1110, 185, 27);
}

if (p.slides.items.length !== 16) throw new Error('Expected sixteen total slides.');
await fs.writeFile(path.join(WORK, `content-${REVISION}.json`), JSON.stringify({
  sourceReport: 'docs/Progress_Reports/Progress_Report_8.docx',
  presentationDate: '2026-11-09', evidenceDate: '2026-11-04', font: FONT, slides: manifest,
}, null, 2));
const candidate = path.join(WORK, 'build', `candidate-${REVISION}.pptx`);
await (await PresentationFile.exportPptx(p)).save(candidate);
console.log(`Exported ${p.slides.items.length} slides. Finalizing ${REVISION}...`);
const finalized = await finalizePresentation({
  workspaceDir: WORK, candidatePath: candidate, finalPath: OUTPUT,
  pythonExecutable: path.join(runtime, 'dependencies/python/bin/python3.12'),
  integrityValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit',
    ...[3, 4, 5, 12].flatMap(n => ['--require-native-table-slide', String(n)])],
  explicitTotalSlideCount: 16,
  requiredNativeTableOwnerSlides: [3, 4, 5, 12], requiredNativeChartOwnerSlides: [6, 7, 8, 9, 10],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: 'design', families: [FONT] }, verifyArtifactToolImport: true,
  receiptPath: path.join(WORK, 'build', `validation-${REVISION}.json`),
});
console.log(JSON.stringify(finalized));
// Inspect actual final-file renders after any portable chart-workbook packaging.
const { FileBlob } = await import(pathToFileURL(runtimeRequire.resolve('@oai/artifact-tool')).href);
const finalDeck = await PresentationFile.importPptx(await FileBlob.load(OUTPUT));
for (const [index, slide] of finalDeck.slides.items.entries()) {
  const blob = await finalDeck.export({ slide, format: 'png', scale: 1 });
  await fs.writeFile(path.join(PREVIEWS, `slide-${String(index + 1).padStart(2, '0')}.png`), new Uint8Array(await blob.arrayBuffer()));
}
console.log(`Saved final deck and ${finalDeck.slides.items.length} slide previews to ${WORK}`);
