# Grounded message review

The three example profiles are actual held-out customers selected by the fixed
logistic uplift policy at 20%. They were chosen from attributes and recommendations,
without using their observed purchase outcomes. The nine drafts were generated
by the Codex assistant in this working session. No email was sent.

The repository can prepare prompts, accept another generator through
`generate_draft(context, generator)`, and audit parsed JSON. There is no hidden
provider call or fabricated large-scale generation experiment.

An independent reviewer should score each message from 1 to 5 on these criteria:

- Relevance: Does the message use the documented customer's purchase context
  naturally? A generic category mention merits at most 3; unsupported
  personalization fails regardless of fluency.
- Factual grounding: Can every factual statement be traced to the supplied
  attributes? Discounts, product specifications, customer gender, inventory,
  and personal names are unavailable and must not be invented.
- Treatment alignment: Does the copy follow the recommended merchandise
  category? An email recommendation need not match past category purchases,
  and purchase flags do not describe gender.
- Clarity: Is the copy readable, useful, and free of model implementation detail?
- Sequence suitability: Would the proposed later stage add value rather than
  repeat the first? Do not treat these ratings as evidence of conversion lift.

Collect two blinded reviewers per draft, record scores and disagreements,
and resolve grounding violations before sending. No independent reviewer scores
have been collected. The author review identifies deliberately modest
personalization and repetition across later stages. Detailed product copy needs
an approved product catalog and offers; these are not available in Hillstrom.

`message_audit.csv` checks declared facts against source values, explicit
category alignment, a relevant attribute, and a small prohibited-claim list.
These are limited automated checks: omission of a claim from the declared-fact
ledger, paraphrased inventions, and semantic mismatch can evade them. Passing
all nine messages does not establish excellent relevance or complete factuality.

Only stage 1 is compatible with the evaluated one-email policy. Stages 2 and 3
are unsent research concepts. A future sequence experiment must specify timing,
eligibility, consent, control assignment, total contacts, and the outcome window.
Hillstrom's two-week outcome window cannot establish that these drafts or a
multi-touch sequence improve conversion.
