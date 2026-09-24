"""Evaluate docking-benchmark outputs using the controlled-comparison metrics."""
import argparse
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from eval_generation_batch import (
    DOCKING_TARGETS, get_hit_threshold, load_result_csv, load_train_novelty,
)


def prepare_stream(raw, budget):
    if budget <= 0:
        raise ValueError('The evaluation budget must be positive')
    if len(raw) < budget:
        raise ValueError(
            'Requested {} outputs but found {}; use --num_mols to select a shorter prefix'.format(
                budget, len(raw)))
    frame = raw.iloc[:budget].copy()
    for column in ('DOCKING', 'QED', 'SA'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['MOL'] = [Chem.MolFromSmiles(str(smiles)) for smiles in frame['SMILES']]
    frame = frame[frame['MOL'].notna() & np.isfinite(frame['DOCKING'])].copy()
    # Preserve the generated-SMILES deduplication used for the comparison tables.
    frame = frame.drop_duplicates('SMILES').copy()
    frame['CANONICAL_SMILES'] = [
        Chem.MolToSmiles(mol, isomericSmiles=False) for mol in frame['MOL']]
    frame['FPS'] = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
                    for mol in frame['MOL']]
    return frame


def annotate_novelty(frame, train_fps):
    if not train_fps:
        raise ValueError('The ZINC training fingerprint reference is empty')
    if any(fp.GetNumBits() != 1024 for fp in train_fps):
        raise ValueError('Benchmark novelty requires 1,024-bit ECFP4 fingerprints')
    frame = frame.copy()
    similarities = {}
    for smiles, fingerprint in zip(frame['CANONICAL_SMILES'], frame['FPS']):
        if smiles not in similarities:
            similarities[smiles] = max(
                DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps))
    frame['SIM'] = frame['CANONICAL_SMILES'].map(similarities).astype(float)
    return frame


def internal_diversity(fingerprints):
    if len(fingerprints) < 2:
        return float('nan')
    total, pairs = 0.0, 0
    for index, fingerprint in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, fingerprints[index + 1:])
        total += float(np.sum(similarities))
        pairs += len(similarities)
    return 1.0 - total / pairs


def summarize_stream(frame, target, budget):
    if budget <= 0 or len(frame) > budget:
        raise ValueError('Invalid evaluation budget')
    if not np.isfinite(frame['SIM']).all() or not frame['SIM'].between(0, 1).all():
        raise ValueError('Every valid unique output needs a finite novelty similarity in [0, 1]')
    eligible = frame[
        (frame['SIM'] < 0.4) & (frame['QED'] > 0.5) & (frame['SA'] > 5.0 / 9.0)
        & np.isfinite(frame['QED']) & np.isfinite(frame['SA'])]
    hits = int((eligible['DOCKING'] > get_hit_threshold(target)).sum())
    top_count = max(int(budget * 0.05), 1)
    top = eligible.nlargest(top_count, 'DOCKING')
    return {
        'target': target,
        'rows': budget,
        'unique_molecules': len(frame),
        'qualified_novel_molecules': len(eligible),
        'novel_hit_count': hits,
        'nhr_pct': 100.0 * hits / budget,
        # Result CSVs store -DS; manuscript tables report the negative DS.
        'novel_top5_ds_kcal_mol': -float(top['DOCKING'].mean()),
        'top5_requested': top_count,
        'top5_used': len(top),
        'internal_diversity': internal_diversity(frame['FPS'].tolist()),
    }


def evaluate_stream(raw, target, train_fps, budget=3000):
    frame = annotate_novelty(prepare_stream(raw, budget), train_fps)
    return summarize_stream(frame, target, budget)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file_path', help='Headerless generation CSV (five or six columns)')
    parser.add_argument('-t', '--target', required=True, choices=DOCKING_TARGETS)
    parser.add_argument('--num_mols', type=int, default=3000,
                        help='Number of initial outputs to evaluate, before filtering')
    parser.add_argument('--output', help='Optional one-row metric CSV')
    args = parser.parse_args(argv)
    raw = load_result_csv(args.file_path)
    frame = prepare_stream(raw, args.num_mols)
    _, train_fps = load_train_novelty()
    result = summarize_stream(annotate_novelty(frame, train_fps), args.target, args.num_mols)
    print('Generated outputs: {}'.format(result['rows']))
    print('Valid unique SMILES: {}'.format(result['unique_molecules']))
    print('Qualified novel molecules: {}'.format(result['qualified_novel_molecules']))
    print('Novel Hit Ratio (%): {:.6f}'.format(result['nhr_pct']))
    print('Novel Top-5% DS (kcal/mol; lower is better): {:.6f}'.format(
        result['novel_top5_ds_kcal_mol']))
    print('Top-5% molecules used/requested: {}/{}'.format(
        result['top5_used'], result['top5_requested']))
    print('Internal diversity: {:.6f}'.format(result['internal_diversity']))
    if result['top5_used'] < result['top5_requested']:
        warnings.warn('Fewer qualified novel molecules than the requested top-5% count; '
                      'the score averages only the available molecules')
    if args.output:
        pd.DataFrame([result]).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
