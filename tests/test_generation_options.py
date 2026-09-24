import contextlib
import io
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from model.sac import SAC
from run import parse_args
from run_benchmark import parse_args as benchmark_args
from train_mfrag import parse_args as training_args


class GenerationOptionsTest(unittest.TestCase):
    def test_training_loss_options(self):
        with patch('sys.argv', ['train_mfrag.py']):
            args = training_args()
        self.assertEqual((args.regression_loss, args.delta), ('huber', 1.0))
        for loss in ('mse', 'huber'):
            with patch('sys.argv', ['train_mfrag.py', '--regression_loss', loss,
                                    '--delta', '0.5']):
                args = training_args()
            self.assertEqual((args.regression_loss, args.delta), (loss, 0.5))

    def test_invalid_training_loss_options_rejected(self):
        cases = [['--delta', value] for value in ('0', '-1', 'nan', 'inf')]
        cases.append(['--regression_loss', 'unknown'])
        for flags in cases:
            with self.subTest(flags=flags), patch('sys.argv', ['train_mfrag.py'] + flags):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    training_args()

    @patch('run_benchmark.os.path.isfile', return_value=True)
    def test_finetuning_loss_options_in_both_entrypoints(self, _):
        for parse in (parse_args, benchmark_args):
            args = parse(['-v', 'vocab.txt'])
            self.assertEqual((args.mfrag_finetune_loss, args.mfrag_finetune_delta),
                             ('huber', 1.0))
            for loss in ('mse', 'huber'):
                args = parse(['-v', 'vocab.txt', '--enable_mfrag_finetune',
                              '--mfrag_finetune_loss', loss, '--mfrag_finetune_delta', '0.5'])
                self.assertEqual((args.mfrag_finetune_loss, args.mfrag_finetune_delta),
                                 (loss, 0.5))
            cases = [['--mfrag_finetune_delta', value] for value in ('0', '-1', 'nan', 'inf')]
            cases.append(['--mfrag_finetune_loss', 'unknown'])
            for flags in cases:
                with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse(['-v', 'vocab.txt'] + flags)

    def test_documented_training_schedule(self):
        command = [
            'train_mfrag.py', '-t', 'parp1', '--model_arch', 'shared',
            '--train_mode', 'joint', '--epochs', '50', '--batch_size', '512',
            '--early_stop_patience', '5',
        ]
        with patch('sys.argv', command):
            args = training_args()
        self.assertEqual((args.epochs, args.batch_size, args.early_stop_patience),
                         (50, 512, 5))
        self.assertEqual((args.model_arch, args.train_mode), ('shared', 'joint'))

    def test_training_checkpoint_root_matches_generation(self):
        with patch('sys.argv', ['train_mfrag.py']):
            args = training_args()
        self.assertEqual(args.out_root, parse_args(['-v', 'vocab.txt']).mfrag_root)
        with patch('sys.argv', ['train_mfrag.py', '--out_root', 'custom_ckpt']):
            self.assertEqual(training_args().out_root, 'custom_ckpt')

    def test_generic_defaults_unchanged(self):
        args = parse_args(['-v', 'vocab.txt'])
        self.assertEqual(args.fragment_selection_mode, 'hybrid')
        self.assertEqual(args.gumbel_noise_scale, 0.001)
        self.assertEqual(args.frag_desc_mode, 'ecfp')
        self.assertEqual(args.num_mols, 6000)
        self.assertEqual(args.mfrag_root, 'ckpt')
        self.assertFalse(args.disable_region_guidance)
        self.assertFalse(hasattr(args, 'enable_constrained_mfrag_guidance'))

    def test_legacy_disable_and_explicit_sac_only_agree(self):
        for flags in (['--disable_region_guidance'], ['--fragment_selection_mode', 'sac_only']):
            args = parse_args(['-v', 'vocab.txt'] + flags)
            self.assertEqual(args.fragment_selection_mode, 'sac_only')
            self.assertTrue(args.disable_region_guidance)

    def test_invalid_options_rejected(self):
        cases = [['--gumbel_noise_scale', value] for value in ('-1', 'nan', 'inf')]
        cases.append(['--fragment_selection_mode', 'mfrag_only', '--disable_region_guidance'])
        for flags in cases:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(['-v', 'vocab.txt'] + flags)

    @patch('run_benchmark.os.path.isfile', return_value=True)
    def test_benchmark_defaults_and_controls(self, _):
        for mode in ('hybrid', 'sac_only', 'mfrag_only'):
            args = benchmark_args(['-v', 'vocab.txt', '--fragment_selection_mode', mode])
            self.assertEqual(args.qed_region_cutoff, 0.75)
            self.assertEqual(args.sa_region_cutoff, 0.70)
            self.assertEqual(args.ga_qed_threshold, 0.5)
            self.assertEqual(args.ga_sa_threshold, 5 / 9)
            self.assertEqual(args.population_size, 100)
            self.assertEqual(args.frag_desc_mode, 'raw_ecfp')
            self.assertEqual(args.region_knn_k, 40)
            self.assertEqual(args.mfrag_root, 'ckpt')
            self.assertEqual(args.qed_mfrag_ckpt, 'ckpt/reg/qed/best.pt')
            self.assertEqual(args.sa_mfrag_ckpt, 'ckpt/reg/sa/best.pt')
            self.assertTrue(args.ga_qed_sa_gate)
            self.assertEqual(args.enable_constrained_mfrag_guidance, mode != 'sac_only')
            self.assertFalse(args.enable_mfrag_finetune)
        args = benchmark_args(['-v', 'vocab.txt', '--gumbel_noise_scale', '1.0'])
        self.assertEqual(args.gumbel_noise_scale, 1.0)

    def test_checkpoint_resolution_defaults_and_overrides(self):
        sac = SAC.__new__(SAC)
        self.assertEqual(sac._resolve_mfrag_ckpt(SimpleNamespace(target='parp1')),
                         'ckpt/reg/parp1/best.pt')
        args = parse_args(['-v', 'vocab.txt', '--mfrag_root', 'custom_ckpt'])
        self.assertEqual(sac._resolve_mfrag_ckpt(args), 'custom_ckpt/reg/parp1/best.pt')
        args.mfrag_ckpt = 'explicit.pt'
        self.assertEqual(sac._resolve_mfrag_ckpt(args), 'explicit.pt')

    def test_sac_only_needs_no_auxiliary_files(self):
        with patch('run_benchmark.os.path.isfile', return_value=False):
            benchmark_args(['-v', 'vocab.txt', '--fragment_selection_mode', 'sac_only'])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark_args(['-v', 'vocab.txt'])

    def test_benchmark_rejects_invalid_criteria(self):
        for flags in (['-t', 'qed'], ['--qed_region_cutoff', 'nan'],
                      ['--sa_region_cutoff', '1.1'], ['--constraint_knn_k', '0'],
                      ['--region_score_mode', 'min_distance']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    benchmark_args(['-v', 'vocab.txt'] + flags)
