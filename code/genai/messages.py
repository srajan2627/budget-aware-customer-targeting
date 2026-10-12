"""Grounded message prompts, pluggable generation, and explicit draft audits.

The checked-in drafts were written by the Codex assistant, not produced by an
unconfigured external API. Audits check declared facts and simple content rules;
they are not independent human relevance ratings or proof of conversion lift.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {'recency', 'history', 'mens', 'womens', 'newbie', 'zip_code', 'channel'}


def make_prompt(context):
    if context['recommended_action'] == 'No E-Mail':
        raise ValueError('No Email customers must not receive a message.')
    facts = {key: context[key] for key in sorted(ALLOWED)}
    return (
        'Draft three email stages as JSON with recommended_action and messages. '
        'Each message needs stage, subject, body, declared_customer_facts, and rationale. '
        'Only stage 1 corresponds to the recorded single-contact recommendation; '
        'stages 2 and 3 are unsent sequence concepts requiring a separate experiment and budget. '
        'Use only the following customer facts. mens and womens indicate past merchandise purchases, '
        'never gender. history is dollars spent in the past year, recency is months since last purchase, '
        'newbie indicates a new customer in the past twelve months. Do not infer demographics, '
        'discounts, product specifications, availability, personal names, or uplift guarantees. '
        'Do not mention model scores in customer copy. Every asserted customer fact must be listed '
        'in declared_customer_facts with the exact source value. Keep the recommended merchandise '
        'category explicit and use documented purchase history for relevance. '
        f'Recommended action: {context["recommended_action"]}. Facts: {json.dumps(facts, sort_keys=True)}'
    )


def generate_draft(context, generator):
    """Provider-neutral callable: generator(prompt) -> a parsed JSON draft."""
    draft = generator(make_prompt(context))
    return dict(source_row=context['source_row'], **draft)


def audit_draft(draft, context):
    rows = []
    messages = draft.get('messages', [])
    if [m.get('stage') for m in messages] != [1, 2, 3]:
        raise ValueError('Require three ordered draft stages, with only stage 1 policy-compatible.')
    expected = context['recommended_action']
    category = "men's" if expected == 'Mens E-Mail' else "women's"
    if expected == 'No E-Mail':
        raise ValueError('No Email customers must not receive a draft.')
    for message in messages:
        text = (message['subject']+' '+message['body']).lower().replace('’', "'")
        claims = message.get('declared_customer_facts', {})
        factual = bool(claims) and all(key in ALLOWED and value == context[key] for key, value in claims.items())
        alignment = draft.get('recommended_action') == expected and category+' merchandise' in text
        # Limited checks: all free-text factual claims still need independent review.
        prohibited = bool(re.search(r'\b(discount|sale|free shipping|guarantee|gender|male|female|husband|wife|age)\b|\d+%', text))
        relevance = bool(set(claims) & {'mens', 'womens', 'recency', 'history', 'channel'})
        rows.append(dict(source_row=context['source_row'], stage=message['stage'],
                         declared_facts_match=factual, category_alignment=alignment,
                         relevant_attribute_declared=relevance, prohibited_claim_flag=prohibited,
                         content_checks_pass=factual and alignment and relevance and not prohibited,
                         policy_compatible_single_contact=message['stage'] == 1,
                         independent_human_review='pending'))
    return rows


def run(directory):
    import csv
    contexts = json.loads((directory/'message_contexts.json').read_text())
    drafts = json.loads((directory/'message_drafts.json').read_text())
    lookup = {row['source_row']: row for row in contexts}
    if set(lookup) != {row['source_row'] for row in drafts['drafts']} or len(drafts['drafts']) != len(lookup):
        raise ValueError('Require exactly one draft sequence for each source context.')
    prompts = [dict(source_row=row['source_row'], prompt=make_prompt(row)) for row in contexts]
    (directory/'message_prompts.json').write_text(json.dumps(prompts, indent=2)+'\n')
    rows = [item for draft in drafts['drafts'] for item in audit_draft(draft, lookup[draft['source_row']])]
    with (directory/'message_audit.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not all(row['content_checks_pass'] for row in rows):
        raise ValueError('One or more draft content checks failed.')
    print(f'{len(rows)} draft messages passed limited content checks; independent review is pending. No emails sent.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'results/evaluation')
    run(parser.parse_args().directory)
