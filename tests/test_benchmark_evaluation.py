import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from eval_benchmark import (
    annotate_novelty, evaluate_stream, internal_diversity, main, prepare_stream,
    summarize_stream,
)
from eval_generation_batch import load_result_csv, load_train_novelty


COLUMNS = ['SMILES', 'DOCKING', 'QED', 'SA', 'TOTAL']


def fingerprint(smiles):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, 1024)


class BenchmarkEvaluationTest(unittest.TestCase):
    def test_joint_criteria_full_denominator_and_docking_sign(self):
        rows = [
            ['CC', 11, .6, .8, .5],
            ['CCC', 19, .5, .8, .5],
            ['CCCC', 20, .8, 5 / 9, .5],
            ['CCCCC', 18, .9, .9, .5],
            ['CCO', 10, .6, .8, .5],
            ['CCN', 12, .6, .8, .5],
            ['CCCl', np.nan, .6, .8, .5],
        ]
        raw = pd.DataFrame(rows + [rows[0]] * 13, columns=COLUMNS)
        prepared = prepare_stream(raw, 20)
        prepared['SIM'] = [.2, .2, .2, .4, .2, .2]
        result = summarize_stream(prepared, 'parp1', 20)
        self.assertEqual(result['rows'], 20)
        self.assertEqual(result['unique_molecules'], 6)
        self.assertEqual(result['qualified_novel_molecules'], 3)
        self.assertEqual(result['novel_hit_count'], 2)
        self.assertEqual(result['nhr_pct'], 10)
        self.assertEqual(result['novel_top5_ds_kcal_mol'], -12)
        self.assertEqual(result['top5_requested'], 1)
        fps = prepared['FPS'].tolist()
        expected = np.mean([1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
                            for i in range(len(fps)) for j in range(i + 1, len(fps))])
        self.assertAlmostEqual(result['internal_diversity'], expected)

    def test_deduplication_matches_recorded_smiles_and_prefix(self):
        raw = pd.DataFrame([
            ['CCO', 11, .6, .8, .5], ['OCC', 12, .6, .8, .5],
            ['CCO', 20, .9, .9, .9], ['CCN', 30, .9, .9, .9],
        ], columns=COLUMNS)
        prepared = prepare_stream(raw, 3)
        self.assertEqual(prepared['SMILES'].tolist(), ['CCO', 'OCC'])
        self.assertEqual(prepared['DOCKING'].tolist(), [11, 12])
        self.assertEqual(prepared['CANONICAL_SMILES'].nunique(), 1)

    def test_similarity_is_ecfp4_not_exact_smiles_only(self):
        raw = pd.DataFrame([['CCC', 12, .9, .9, .6]], columns=COLUMNS)
        reference = [fingerprint('CC')]
        result = annotate_novelty(prepare_stream(raw, 1), reference)
        expected = DataStructs.TanimotoSimilarity(fingerprint('CCC'), reference[0])
        self.assertAlmostEqual(result['SIM'].iloc[0], expected)
        self.assertGreater(expected, 0)

    def test_training_match_is_not_novel(self):
        raw = pd.DataFrame([['CC', 12, .9, .9, .6]], columns=COLUMNS)
        result = evaluate_stream(raw, 'parp1', [fingerprint('CC')], budget=1)
        self.assertEqual(result['nhr_pct'], 0)
        self.assertEqual(result['top5_used'], 0)
        self.assertTrue(np.isnan(result['novel_top5_ds_kcal_mol']))
        self.assertTrue(np.isnan(result['internal_diversity']))

    def test_fewer_than_requested_eligible_molecules_is_explicit(self):
        raw = pd.DataFrame([['CC', 12, .9, .9, .6]] * 40, columns=COLUMNS)
        prepared = prepare_stream(raw, 40)
        prepared['SIM'] = .1
        result = summarize_stream(prepared, 'parp1', 40)
        self.assertEqual(result['top5_requested'], 2)
        self.assertEqual(result['top5_used'], 1)
        self.assertEqual(result['novel_top5_ds_kcal_mol'], -12)

    def test_invalid_budget_and_missing_similarity_rejected(self):
        raw = pd.DataFrame([['CC', 12, .9, .9, .6]], columns=COLUMNS)
        for budget in (-1, 0, 2):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                prepare_stream(raw, budget)
        prepared = prepare_stream(raw, 1)
        prepared['SIM'] = np.nan
        with self.assertRaisesRegex(ValueError, 'novelty similarity'):
            summarize_stream(prepared, 'parp1', 1)
        with self.assertRaisesRegex(ValueError, 'empty'):
            annotate_novelty(prepared, [])

    def test_all_invalid_outputs_keep_denominator(self):
        raw = pd.DataFrame([['CC', np.nan, .9, .9, .6]] * 20, columns=COLUMNS)
        result = evaluate_stream(raw, 'parp1', [fingerprint('CC')], 20)
        self.assertEqual(result['rows'], 20)
        self.assertEqual(result['unique_molecules'], 0)
        self.assertEqual(result['nhr_pct'], 0)
        self.assertTrue(np.isnan(result['internal_diversity']))

    def test_five_and_six_column_csv_and_command_output(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = pd.DataFrame([['CC', 12, .9, .9, .6]] * 20, columns=COLUMNS)
            for with_source in (False, True):
                frame = raw.copy()
                if with_source:
                    frame.insert(1, 'TYPE', ['sac', 'ga'] * 10)
                path = Path(directory) / ('{}.csv'.format(with_source))
                frame.to_csv(path, header=False, index=False)
                self.assertEqual(len(load_result_csv(path)), 20)
                output = Path(directory) / 'metrics.csv'
                with patch('eval_benchmark.load_train_novelty',
                           return_value=(set(), [fingerprint('c1ccccc1')])):
                    with contextlib.redirect_stdout(io.StringIO()) as text:
                        main([str(path), '-t', 'parp1', '--num_mols', '20',
                              '--output', str(output)])
                self.assertIn('Novel Hit Ratio (%)', text.getvalue())
                metrics = pd.read_csv(output).iloc[0]
                self.assertEqual(metrics['nhr_pct'], 5)
                self.assertEqual(metrics['novel_top5_ds_kcal_mol'], -12)

    def test_missing_novelty_cache_can_be_built_from_public_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data'
            data.mkdir()
            pd.DataFrame({'smiles': ['CC', 'CCO']}).to_csv(data / 'zinc250k.csv', index=False)
            (data / 'valid_idx_zinc250k.json').write_text(json.dumps([1]))
            previous = os.getcwd()
            try:
                os.chdir(directory)
                smiles, fps = load_train_novelty()
                self.assertEqual(smiles, {'CC'})
                self.assertEqual(len(fps), 1)
                self.assertEqual(fps[0].GetNumBits(), 1024)
                self.assertTrue((data / 'zinc250k_novelty.pt').is_file())
            finally:
                os.chdir(previous)

    def test_diversity_with_zero_or_one_molecule(self):
        self.assertTrue(np.isnan(internal_diversity([])))
        self.assertTrue(np.isnan(internal_diversity([fingerprint('CC')])))
