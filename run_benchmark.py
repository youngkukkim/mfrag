"""Docking benchmark: property-conditioned selection and GA parent eligibility."""
import os

from run import DOCKING_TARGETS, build_parser, main, validate_args


def parse_args(argv=None):
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(reward_mode='docking', frag_desc_mode='raw_ecfp')
    parser.add_argument('--constraint_embedding_dir', default='mfrag_region_cache/benchmark')
    parser.add_argument('--qed_mfrag_ckpt', default='ckpt/reg/qed/best.pt')
    parser.add_argument('--sa_mfrag_ckpt', default='ckpt/reg/sa/best.pt')
    parser.add_argument('--constraint_knn_k', type=int, default=40)
    parser.add_argument('--qed_region_cutoff', type=float, default=0.75)
    parser.add_argument('--sa_region_cutoff', type=float, default=0.70)
    parser.add_argument('--ga_qed_threshold', type=float, default=0.5)
    parser.add_argument('--ga_sa_threshold', type=float, default=5.0 / 9.0)
    args = validate_args(parser, parser.parse_args(argv))
    if args.target not in DOCKING_TARGETS:
        parser.error('The docking benchmark requires a docking target')
    if args.region_score_mode != 'knn':
        parser.error('Benchmark region thresholds require --region_score_mode knn')
    if args.constraint_knn_k <= 0 or args.region_knn_k <= 0:
        parser.error('Region neighbor counts must be positive')
    for name in ('qed_region_cutoff', 'sa_region_cutoff',
                 'ga_qed_threshold', 'ga_sa_threshold'):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error('--{} must be between 0 and 1'.format(name))
    args.enable_constrained_mfrag_guidance = args.fragment_selection_mode != 'sac_only'
    args.ga_qed_sa_gate = True
    if args.enable_constrained_mfrag_guidance:
        for name in ('qed', 'sa'):
            for path in (getattr(args, name + '_mfrag_ckpt'),
                         os.path.join(args.constraint_embedding_dir, name + '.npz')):
                if not os.path.isfile(path):
                    parser.error('Missing benchmark input: {}. See prepare_benchmark_references.py.'.format(path))
    return args


if __name__ == '__main__':
    main(parse_args())
