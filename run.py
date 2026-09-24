import os
import torch
import gym
import argparse
import pickle
import math
import json

from model.sac import SAC
from utils_sac.utils import set_seed, get_vocab

import warnings
warnings.filterwarnings(action='ignore')


DOCKING_TARGETS = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
AUX_TARGETS = ['qed', 'sa']
TARGETS = DOCKING_TARGETS + AUX_TARGETS


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=int, default=-1)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-t', '--target', type=str, default='parp1',
                        choices=TARGETS)
    parser.add_argument('-v', '--vocab_path', type=str, required=True)
    
    parser.add_argument('--num_mols', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--start_steps', type=int, default=4000)
    parser.add_argument('--update_after', type=int, default=3000)
    parser.add_argument('--update_every', type=int, default=256)
    
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--num_layer', type=int, default=3)
    
    parser.add_argument('--tau', type=float, default=1e-1)
    parser.add_argument('--target_entropy', type=float, default=1.)
    parser.add_argument('--init_alpha', type=float, default=1.)
    parser.add_argument('--init_pi_lr', type=float, default=1e-4)
    parser.add_argument('--init_q_lr', type=float, default=1e-4)
    parser.add_argument('--init_alpha_lr', type=float, default=5e-4)
    parser.add_argument('--alpha_max', type=float, default=20.)
    parser.add_argument('--alpha_min', type=float, default=.05)

    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--mutation_rate', type=float, default=0.1)

    parser.add_argument('--max_vocab_update', type=int, default=50)
    parser.add_argument('--max_vocab_size', type=int, default=10000)
    parser.add_argument('--reward_mode', type=str, default='composite',
                        choices=['composite', 'docking'])
    parser.add_argument('--frag_desc_mode', type=str, default='ecfp',
                        choices=['raw_ecfp', 'ecfp', 'sp', 'concat', 'sum'])
    parser.add_argument('--frag_desc_dim', type=int, default=128)
    parser.add_argument('--ecfp_weight', type=float, default=1.0)
    parser.add_argument('--sp_weight', type=float, default=1.0)
    parser.add_argument('--disable_region_guidance', action='store_true')
    parser.add_argument('--fragment_selection_mode', default='hybrid',
                        choices=['hybrid', 'sac_only', 'mfrag_only'],
                        help='Fragment identity after random warmup; attachment sites still use SAC.')
    parser.add_argument('--gumbel_noise_scale', type=float, default=1e-3,
                        help='Gumbel noise multiplier at all three action stages (1.0: unscaled).')
    parser.add_argument('--region_train_sample_size', type=int, default=1000)
    parser.add_argument('--region_high_score_quantile', type=float, default=0.95)
    parser.add_argument('--region_knn_k', type=int, default=40)
    parser.add_argument('--region_score_mode', type=str, default='knn',
                        choices=['knn', 'min_distance'])
    parser.add_argument('--region_generated_max', type=int, default=5000)
    parser.add_argument('--mfrag_ckpt', type=str, default='',
                        help='Explicit MFRAG checkpoint path. Overrides mfrag_root/label/ckpt_name.')
    parser.add_argument('--mfrag_root', type=str, default='ckpt')
    parser.add_argument('--mfrag_label_mode', type=str, default='reg', choices=['reg', 'cls'])
    parser.add_argument('--mfrag_ckpt_name', type=str, default='best.pt')
    parser.add_argument('--mfrag_model_arch', type=str, default='auto',
                        choices=['auto', 'shared', 'dual'])
    parser.add_argument('--enable_mfrag_finetune', action='store_true')
    parser.add_argument('--mfrag_finetune_trigger', type=int, default=1000)
    parser.add_argument('--mfrag_finetune_interval', type=int, default=1000)
    parser.add_argument('--mfrag_finetune_max_generated', type=int, default=5000)
    parser.add_argument('--mfrag_finetune_epochs', type=int, default=3)
    parser.add_argument('--mfrag_finetune_batch_size', type=int, default=512)
    parser.add_argument('--mfrag_finetune_lr', type=float, default=1e-4)
    parser.add_argument('--mfrag_finetune_loss', default='huber', choices=['huber', 'mse'],
                        help='Property-prediction loss for optional M-FRAG fine-tuning.')
    parser.add_argument('--mfrag_finetune_delta', type=float, default=1.0,
                        help='Positive Huber transition threshold; ignored for MSE.')
    parser.add_argument('--mfrag_finetune_generated_top_frac', type=float, default=0.4)
    parser.add_argument('--mfrag_finetune_generated_random_frac', type=float, default=0.2)
    parser.add_argument('--mfrag_finetune_zinc_top_frac', type=float, default=0.2)
    parser.add_argument('--mfrag_finetune_zinc_random_frac', type=float, default=0.2)
    return parser


def validate_args(parser, args):
    if not math.isfinite(args.mfrag_finetune_delta) or args.mfrag_finetune_delta <= 0:
        parser.error('--mfrag_finetune_delta must be finite and positive')
    if not math.isfinite(args.gumbel_noise_scale) or args.gumbel_noise_scale < 0:
        parser.error('--gumbel_noise_scale must be finite and non-negative')
    if args.disable_region_guidance:
        if args.fragment_selection_mode == 'mfrag_only':
            parser.error('--disable_region_guidance is incompatible with mfrag_only')
        args.fragment_selection_mode = 'sac_only'
    args.disable_region_guidance = args.fragment_selection_mode == 'sac_only'
    return args


def parse_args(argv=None):
    parser = build_parser()
    return validate_args(parser, parser.parse_args(argv))


def main(args=None):
    if args is None:
        args = parse_args()
    print(args)
    
    if args.gpu_id >= 0:
        args.device = torch.device(f'cuda:{args.gpu_id}')
    else:
        args.device = torch.device('cpu')
        torch.set_num_threads(256)
    
    if not os.path.exists('results'):
        os.makedirs('results')

    gym.envs.registration.register(id='molecule-v0', entry_point='utils_sac.env:MoleculeEnv')
    set_seed(args.seed)

    vocab = get_vocab(args.vocab_path)

    env = gym.make('molecule-v0')
    env.init(vocab=vocab, target=args.target, reward_mode=args.reward_mode)
    env.seed(args.seed)

    sac = SAC(args, vocab, env)
    with open(sac.fname[:-4] + '_settings.json', 'w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)
    with open(sac.fname[:-4]+".pkl", 'wb') as f:
    	pickle.dump(sac.vocab['FRAG_QUEUE'], f)
    sac.run()
    with open(sac.fname[:-4]+"_after.pkl", 'wb') as f:
    	pickle.dump(sac.vocab['FRAG_QUEUE'], f)
    env.close()


if __name__ == '__main__':
    main()
