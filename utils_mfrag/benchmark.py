"""Checkpoint and reference loading for the docking benchmark."""
import hashlib

import numpy as np
import torch

from model.mfrag import MFRAG


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_scalar_mfrag(path, device, model_arch='auto'):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get('state_dict', checkpoint)
    if model_arch == 'auto':
        model_arch = 'dual' if any(
            key.startswith(('mol_encoder.', 'frag_encoder.')) for key in state
        ) else 'shared'
    model = MFRAG(device=device, model_arch=model_arch).to(device)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_benchmark_reference(path, checkpoint_path, property_name):
    with np.load(path, allow_pickle=False) as payload:
        if ('checkpoint_sha256' not in payload
                or payload['checkpoint_sha256'].item() != file_sha256(checkpoint_path)):
            raise ValueError('Reference/checkpoint mismatch; rerun prepare_benchmark_references.py')
        if payload['property'].item() != property_name:
            raise ValueError('Wrong property in {}'.format(path))
        if not np.all(payload['split'].astype(str) == 'train'):
            raise ValueError('Benchmark references must contain training molecules only')
        embeddings = payload['molecule_embeddings'].copy()
        scores = payload['true_values'].copy()
    if (embeddings.ndim != 2 or scores.ndim != 1 or len(scores) == 0
            or len(embeddings) != len(scores) or not np.isfinite(embeddings).all()
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError('Invalid benchmark reference: {}'.format(path))
    return embeddings, scores
