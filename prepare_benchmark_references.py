"""Encode the fixed ZINC training references for QED/SA benchmark selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from utils_mfrag.benchmark import file_sha256, load_scalar_mfrag
from utils_mfrag.data import get_graph_from_frag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference_indices', default='data/benchmark_reference_indices.json')
    parser.add_argument('--data_csv', default='data/zinc250k.csv')
    parser.add_argument('--valid_indices', default='data/valid_idx_zinc250k.json')
    parser.add_argument('--qed_mfrag_ckpt', default='ckpt/reg/qed/best.pt')
    parser.add_argument('--sa_mfrag_ckpt', default='ckpt/reg/sa/best.pt')
    parser.add_argument('--out_dir', default='mfrag_region_cache/benchmark')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')
    with open(args.reference_indices) as handle:
        indices = json.load(handle)
    with open(args.valid_indices) as handle:
        validation = set(json.load(handle))
    frame = pd.read_csv(args.data_csv)
    if (not indices or len(set(indices)) != len(indices)
            or any(not isinstance(i, int) or i < 0 or i >= len(frame) for i in indices)
            or validation.intersection(indices)):
        raise ValueError('Reference indices must be unique ZINC training-row indices')
    selected = frame.iloc[indices]
    graphs = [get_graph_from_frag(smiles) for smiles in selected['smiles']]
    if any(graph is None for graph in graphs):
        raise ValueError('Invalid graph in benchmark references')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ('qed', 'sa'):
        checkpoint_path = getattr(args, name + '_mfrag_ckpt')
        model = load_scalar_mfrag(checkpoint_path, args.device)
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(graphs), args.batch_size):
                batch = Batch.from_data_list(graphs[start:start + args.batch_size]).to(args.device)
                _, encoded = model(batch)
                embeddings.append(encoded.cpu().numpy())
        output = out_dir / (name + '.npz')
        np.savez_compressed(
            str(output), molecule_embeddings=np.concatenate(embeddings),
            true_values=selected[name].to_numpy(dtype=np.float32),
            data_indices=np.asarray(indices, dtype=np.int64),
            split=np.full(len(indices), 'train'), property=np.asarray(name),
            checkpoint_sha256=np.asarray(file_sha256(checkpoint_path)),
            source_sha256=np.asarray(file_sha256(args.data_csv)),
        )
        print('{}: {} training references -> {}'.format(name, len(indices), output))


if __name__ == '__main__':
    main()
