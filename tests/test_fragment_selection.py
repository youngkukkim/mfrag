import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from model.ac import SFSPolicy


class ConstantEncoder(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, batch):
        # Deliberately different prediction and embedding: guidance uses the latter.
        return torch.tensor([[-99.0]]), torch.tensor([[self.value]])


def make_policy(mode='hybrid', constrained=False):
    policy = SFSPolicy.__new__(SFSPolicy)
    nn.Module.__init__(policy)
    policy.device = 'cpu'
    policy.fragment_selection_mode = mode
    policy.region_guidance_enabled = mode != 'sac_only'
    policy.constraint_guidance_enabled = constrained
    policy.constraint_region_guides = {}
    policy.gumbel_noise_scale = 0.001
    policy.motif_type_num = 3
    policy.env = SimpleNamespace(counter=0, get_final_smiles_mol=lambda: (None, 'CC'))
    policy.mfrag = ConstantEncoder(0.4)
    policy.cand_mfrag = torch.tensor([[0.8], [0.6], [0.2]])
    policy.region_ref_embeddings = torch.tensor([[0.0], [1.0]])
    policy.region_ref_scores = torch.tensor([0.0, 1.0])
    policy.query_region_scores = lambda embeddings: embeddings[:, 0]
    policy.get_fragment_viability_mask = lambda _: torch.tensor([[True, True, True]])
    if constrained:
        for name, cutoff in (('qed', 0.75), ('sa', 0.70)):
            policy.constraint_region_guides[name] = {
                'model': ConstantEncoder(0.8),
                'candidate_embeddings': torch.tensor([[0.6], [0.8], [0.8]]),
                'reference_embeddings': torch.tensor([[0.0], [1.0]]),
                'reference_scores': torch.tensor([0.0, 1.0]),
                'knn_k': 2, 'cutoff': cutoff,
            }
        policy._query_reference_scores = lambda embeddings, *_: embeddings[:, 0]
    return policy


class FragmentSelectionTest(unittest.TestCase):
    def test_generic_guidance_overrides_sac_preferences(self):
        logits = make_policy().get_region_guided_logits(1, torch.tensor([1.0, 9.0, 3.0]))
        torch.testing.assert_close(logits, torch.tensor([0.2, 0.1, -1e9]))

    def test_sac_only_leaves_logits_unchanged(self):
        logits = torch.tensor([1.0, 9.0, 3.0])
        torch.testing.assert_close(make_policy('sac_only').get_region_guided_logits(1, logits), logits)

    def test_benchmark_requires_all_three_conditions(self):
        logits = make_policy(constrained=True).get_region_guided_logits(1, torch.tensor([9., 1., 2.]))
        torch.testing.assert_close(logits, torch.tensor([-1e9, 0.1, -1e9]))

    def test_region_cutoff_is_inclusive(self):
        policy = make_policy(constrained=True)
        for name in ('qed', 'sa'):
            policy.constraint_region_guides[name]['cutoff'] = 0.75
            policy.constraint_region_guides[name]['candidate_embeddings'][:] = 0.7
        logits = policy.get_region_guided_logits(1, torch.tensor([9., 1., 2.]))
        torch.testing.assert_close(logits, torch.tensor([0.2, 0.1, -1e9]))

    def test_no_eligible_hybrid_uses_sac_mfrag_only_uses_geometry(self):
        for mode in ('hybrid', 'mfrag_only'):
            policy = make_policy(mode, constrained=True)
            policy.constraint_region_guides['qed']['cutoff'] = 0.95
            policy.cand_mfrag[:] = torch.tensor([[0.2], [0.1], [0.0]])
            fallback = torch.tensor([9., 2., 1.])
            result = policy.get_region_guided_logits(1, fallback)
            expected = fallback if mode == 'hybrid' else torch.tensor([-0.1, -0.15, -0.2])
            torch.testing.assert_close(result, expected)

    def test_encoding_failure_never_uses_sac_in_mfrag_only(self):
        policy = make_policy('mfrag_only')
        policy.env.get_final_smiles_mol = Mock(side_effect=ValueError('Invalid molecule'))
        policy.get_fragment_viability_mask = lambda _: torch.tensor([[True, False, True]])
        result = policy.get_region_guided_logits(1, torch.tensor([9., 2., 1.]))
        torch.testing.assert_close(result, torch.tensor([0., -1e9, 0.]))

    def test_missing_or_stale_guides_rejected(self):
        policy = make_policy(constrained=True)
        policy.constraint_region_guides['qed']['candidate_embeddings'] = torch.zeros(1, 1)
        with self.assertRaisesRegex(RuntimeError, 'stale qed'):
            policy.get_region_guided_logits(1, torch.zeros(3))
        policy = make_policy('mfrag_only')
        policy.region_ref_embeddings = None
        with self.assertRaisesRegex(RuntimeError, 'initialized region'):
            policy.get_region_guided_logits(1, torch.zeros(3))

    def test_vocab_update_refreshes_each_property(self):
        policy = make_policy(constrained=True)
        policy.create_candidate_motifs = Mock(return_value=[{'att': [0, 1]}] * 4)
        policy.get_candidate_ecfp = Mock(return_value=torch.zeros(4, 1024))
        policy.get_candidate_mfrag_embedding = Mock(return_value=torch.zeros(4, 1))
        policy.update_vocab({'FRAG_MOL': [None] * 4})
        self.assertEqual(policy.get_candidate_mfrag_embedding.call_count, 3)
        for guide in policy.constraint_region_guides.values():
            self.assertEqual(guide['candidate_embeddings'].shape, (4, 1))

    def test_noise_default_matches_previous_formula(self):
        policy = make_policy()
        logits = torch.tensor([[0.2, 0.8, 0.3]])
        for scale in (0.001, 1.0):
            policy.gumbel_noise_scale = scale
            torch.manual_seed(9)
            noise = -torch.empty_like(logits).exponential_().log()
            expected = ((logits + scale * noise) / 0.1).softmax(-1)
            torch.manual_seed(9)
            actual = policy.gumbel_softmax(logits, tau=0.1)
            torch.testing.assert_close(actual, expected)

    def test_knn_region_score(self):
        value = SFSPolicy._query_reference_scores(
            torch.tensor([[0.25]]), torch.tensor([[0.0], [1.0]]), torch.tensor([0., 1.]), 40)
        torch.testing.assert_close(value, torch.tensor([0.25]), atol=1e-5, rtol=1e-5)
