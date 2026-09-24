import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from model.sac import SAC
from utils_mfrag.benchmark import file_sha256, load_benchmark_reference


class BenchmarkTest(unittest.TestCase):
    def population(self, enabled=True):
        sac = SAC.__new__(SAC)
        sac.args = SimpleNamespace(ga_qed_sa_gate=enabled, ga_qed_threshold=0.5,
                                   ga_sa_threshold=5 / 9)
        sac.population = []
        sac.population_score = []
        sac.population_size = 2
        return sac

    def test_ga_uses_strict_actual_property_thresholds(self):
        sac = self.population()
        mask = sac._ga_parent_mask([.5, .6, .7, np.nan, .8], [.9, 5 / 9, .9, .9, np.inf])
        np.testing.assert_array_equal(mask, [False, False, True, False, False])

    def test_parent_filter_does_not_mutate_generated_inputs_and_ranks_by_docking(self):
        sac = self.population()
        molecules, scores = ['A', 'B', 'C', 'D'], [.99, .6, .7, .8]
        sac._update_ga_population(molecules, scores, [.1, .9, .7, .6], [.9] * 4)
        self.assertEqual(sac.population, ['D', 'C'])
        self.assertEqual(sac.population_score, [.8, .7])
        self.assertEqual(molecules, ['A', 'B', 'C', 'D'])
        self.assertEqual(scores, [.99, .6, .7, .8])

    def test_generic_parent_ranking_unchanged(self):
        sac = self.population(enabled=False)
        sac._update_ga_population(['A', 'B', 'C'], [.9, .6, .8], [.1] * 3, [.1] * 3)
        self.assertEqual(sac.population, ['A', 'C'])

    def test_no_eligible_parents_is_valid(self):
        sac = self.population()
        sac._update_ga_population(['A'], [.9], [.1], [.9])
        self.assertEqual(sac.population, [])

    def test_reference_property_and_checkpoint_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.pt'
            checkpoint.write_bytes(b'test checkpoint')
            path = Path(directory) / 'qed.npz'
            np.savez(str(path), checkpoint_sha256=file_sha256(checkpoint), property='qed',
                     split=['train', 'train'], molecule_embeddings=[[0.0], [1.0]], true_values=[.1, .9])
            embeddings, scores = load_benchmark_reference(path, checkpoint, 'qed')
            self.assertEqual(embeddings.shape, (2, 1))
            self.assertEqual(scores.shape, (2,))
            with self.assertRaisesRegex(ValueError, 'Wrong property'):
                load_benchmark_reference(path, checkpoint, 'sa')
            checkpoint.write_bytes(b'different checkpoint')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                load_benchmark_reference(path, checkpoint, 'qed')
