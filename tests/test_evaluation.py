"""Validate combined budgets, paired uncertainty, stability, and grounding."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in ('data', 'models', 'evaluation', 'segmentation', 'targeting', 'genai'):
    sys.path.insert(0, str(ROOT/'code'/directory))
from clusters import cluster_pipeline, wilson
from policies import assign_policy
from policy_value import paired_bootstrap_totals, policy_scores
from reliability import agreement, training_bootstrap
from messages import audit_draft, generate_draft, make_prompt


class EvaluationTests(unittest.TestCase):
    def test_opposite_emails_share_one_cap_and_nonpositive_uplift_excluded(self):
        p = np.array([[.1,.3,.2],[.1,.2,.4],[.1,.1,.1],[.2,.1,.1],[.1,.2,.15]])
        pi = assign_policy(p, 'uplift', .4)
        np.testing.assert_array_equal(pi, [1,2,0,0,0])
        self.assertEqual(np.count_nonzero(assign_policy(p,'uplift',1)),3)

    def test_ipw_recovers_known_heterogeneous_policy_in_balanced_trial(self):
        # Two equal-sized covariate groups, each observed equally under all actions.
        a = np.tile([0,1,2],20)
        group = np.repeat([0,1],30)
        pi = np.where(group == 0,1,2)
        y = (a == pi).astype(int)
        _, score = policy_scores(a,y,pi,[1/3]*3)
        self.assertEqual(score.sum(),len(a))
        for action in (1,2):
            self.assertEqual((score*(pi == action)).sum(),len(a)/2)

    def test_bootstrap_is_paired_and_reproducible(self):
        s = np.array([0.,3.,-3.,0.,3.])
        scores = np.column_stack([s,s,2*s,np.zeros(len(s))])
        draws = paired_bootstrap_totals(scores,repeats=100,seed=8)
        np.testing.assert_array_equal(draws[:,0],draws[:,1])
        np.testing.assert_array_equal(draws[:,2],2*draws[:,0])
        np.testing.assert_array_equal(draws[:,3],0)
        np.testing.assert_array_equal(draws,paired_bootstrap_totals(scores,repeats=100,seed=8))
        self.assertGreater(draws[:,0].std(),0)
        with self.assertRaises(ValueError):
            paired_bootstrap_totals([[np.nan]],repeats=10)

    def test_agreement_distinguishes_selection_from_email(self):
        row=agreement([1,2,0,0],[2,0,1,0])
        self.assertEqual(row['selection_overlap'],.5)
        self.assertAlmostEqual(row['selection_jaccard'],1/3)
        self.assertEqual(row['treatment_agreement_shared'],0)
        self.assertEqual(row['action_agreement_all'],.25)
        self.assertTrue(np.isnan(agreement([0,0],[0,0])['selection_overlap']))

    def test_training_resampling_preserves_arms_and_varies_purchases(self):
        frame=pd.DataFrame({'treatment':np.repeat(['a','b','c'],100),
                            'conversion':np.tile(np.r_[np.ones(10),np.zeros(90)],3)})
        samples=[training_bootstrap(frame,seed) for seed in range(10)]
        for sample in samples:
            self.assertEqual(sample.groupby('treatment').size().to_dict(),{'a':100,'b':100,'c':100})
        self.assertGreater(np.std([s.conversion.sum() for s in samples]),0)

    def test_cluster_transform_never_reads_outcomes(self):
        n=60
        frame=pd.DataFrame(dict(recency=np.arange(n)%12+1,history=np.arange(n)+10.,
                                mens=np.arange(n)%2,womens=(np.arange(n)+1)%2,newbie=np.arange(n)%2,
                                zip_code=['Urban']*n,channel=['Web']*n,conversion=np.arange(n)%2))
        model=cluster_pipeline(2).fit(frame)
        altered=frame.assign(conversion=999,treatment='unseen',spend=1e9)
        np.testing.assert_array_equal(model.predict(frame),model.predict(altered))
        lower,upper=wilson(0,50)
        self.assertAlmostEqual(lower,0)
        self.assertGreater(upper,0)


class MessageTests(unittest.TestCase):
    def context(self):
        return dict(source_row=1,recency=3,history=100.,mens=1,womens=0,newbie=0,
                    zip_code='Urban',channel='Web',recommended_action='Mens E-Mail')

    def draft(self):
        return dict(source_row=1,recommended_action='Mens E-Mail',messages=[
            dict(stage=i,subject="Browse men's merchandise",body='Your past purchases include this category.',
                 declared_customer_facts={'mens':1},rationale='Past merchandise history.') for i in (1,2,3)])

    def test_wrong_facts_and_email_alignment_fail(self):
        draft=self.draft()
        self.assertTrue(all(r['content_checks_pass'] for r in audit_draft(draft,self.context())))
        draft['messages'][0]['declared_customer_facts']={'mens':0}
        self.assertFalse(audit_draft(draft,self.context())[0]['content_checks_pass'])
        draft['recommended_action']='Womens E-Mail'
        self.assertFalse(any(r['category_alignment'] for r in audit_draft(draft,self.context())))

    def test_no_email_is_not_sent_to_generator(self):
        context=self.context(); context['recommended_action']='No E-Mail'
        called=[]
        with self.assertRaises(ValueError):
            generate_draft(context,lambda prompt:called.append(prompt))
        self.assertEqual(called,[])
        self.assertIn('never gender',make_prompt(self.context()))


if __name__=='__main__':
    unittest.main()
