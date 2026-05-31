import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


DOCKING_TARGETS = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
AUX_TARGETS = ['qed', 'sa']
TARGETS = DOCKING_TARGETS + AUX_TARGETS


def get_hit_threshold(target):
    if target == 'parp1':
        return 10.0
    if target == 'fa7':
        return 8.5
    if target == '5ht1b':
        return 8.7845
    if target == 'braf':
        return 10.3
    if target == 'jak2':
        return 9.1
    raise ValueError(target)


def load_result_csv(path):
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] == 5:
        raw.columns = ['SMILES', 'DOCKING', 'QED', 'SA', 'TOTAL']
    elif raw.shape[1] == 6:
        raw.columns = ['SMILES', 'TYPE', 'DOCKING', 'QED', 'SA', 'TOTAL']
    else:
        raise ValueError(f'Unsupported CSV shape {raw.shape} for {path}')
    if 'TYPE' not in raw:
        raw['TYPE'] = 'unknown'
    return raw


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    return Chem.MolToSmiles(mol, isomericSmiles=False), mol


def prepare_df(df):
    df = df.copy()
    payload = df['SMILES'].apply(canonicalize)
    df['CANONICAL_SMILES'] = [item[0] for item in payload]
    df['MOL'] = [item[1] for item in payload]
    df = df.dropna(subset=['CANONICAL_SMILES', 'MOL']).copy()
    return df.drop_duplicates(subset=['CANONICAL_SMILES']).copy()


def load_train_novelty():
    novelty_path = 'data/zinc250k_novelty.pt'
    if os.path.exists(novelty_path):
        return torch.load(novelty_path)
    train_df = pd.read_csv('data/zinc250k.csv')
    with open('data/valid_idx_zinc250k.json') as handle:
        test_idx = set(json.load(handle))
    train_idx = [idx for idx in range(len(train_df)) if idx not in test_idx]
    train_smiles = train_df.iloc[train_idx]['smiles'].tolist()
    train_mols = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]
    train_mols = [mol for mol in train_mols if mol is not None]
    train_smiles = set(Chem.MolToSmiles(mol, isomericSmiles=False) for mol in train_mols)
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in train_mols]
    torch.save((train_smiles, train_fps), novelty_path)
    return train_smiles, train_fps


def load_train_smiles_only():
    train_df = pd.read_csv('data/zinc250k.csv')
    with open('data/valid_idx_zinc250k.json') as handle:
        test_idx = set(json.load(handle))
    train_idx = [idx for idx in range(len(train_df)) if idx not in test_idx]
    return set(train_df.iloc[train_idx]['smiles'].astype(str).tolist())


def add_similarity(df, train_fps):
    df = df.copy()
    if len(df) == 0:
        df['SIM'] = []
        return df
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in df['MOL']]
    df['SIM'] = [max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in fps]
    return df


def top_mean(values, k):
    values = pd.to_numeric(values, errors='coerce').dropna().to_numpy(dtype=np.float32)
    if len(values) == 0:
        return float('nan')
    values = np.sort(values)[::-1]
    return float(values[:min(k, len(values))].mean())


def auc_top_n(mol_dict, top_n=10, finish=True, freq_log=100, max_oracle_calls=10000):
    if len(mol_dict) == 0:
        return float('nan')
    ordered = sorted(
        [(float(value[0]), int(value[1])) for value in mol_dict.values() if np.isfinite(float(value[0]))],
        key=lambda item: item[1],
    )
    if len(ordered) == 0:
        return float('nan')
    scores_seen = []
    ptr = 0
    prev_x = 0
    prev_y = 0.0
    area = 0.0
    for x in range(freq_log, max_oracle_calls + 1, freq_log):
        while ptr < len(ordered) and ordered[ptr][1] < x:
            scores_seen.append(ordered[ptr][0])
            ptr += 1
        if scores_seen:
            top_scores = np.sort(np.asarray(scores_seen, dtype=np.float32))[::-1]
            y = float(top_scores[:min(top_n, len(top_scores))].mean())
        else:
            y = 0.0
        area += (x - prev_x) * (prev_y + y) / 2.0
        prev_x = x
        prev_y = y
        if not finish and ptr >= len(ordered):
            return area / max(prev_x, 1)
    return area / max_oracle_calls


def build_auc_buffer(df, score_col='DOCKING'):
    if len(df) == 0:
        return {}
    df = df.copy()
    df['CALL_INDEX'] = np.arange(len(df))
    df = prepare_df(df)
    df = df.sort_values('CALL_INDEX').drop_duplicates(subset=['CANONICAL_SMILES'], keep='first')
    df = df.iloc[:10000].copy()
    return {
        row['CANONICAL_SMILES']: [float(row[score_col]), int(row['CALL_INDEX'])]
        for _, row in df.iterrows()
        if np.isfinite(float(row[score_col]))
    }


def summarize_docking(target, df, train_smiles, train_fps, novelty_mode):
    total_n = len(df)
    df = prepare_df(df)
    valid_unique_n = len(df)
    if novelty_mode == 'soft':
        df = add_similarity(df, train_fps)
        novel_n = len(df[df['SIM'] < 0.4])
    else:
        novel_n = len(df[~df['CANONICAL_SMILES'].isin(train_smiles)])
    hit_thr = get_hit_threshold(target)
    top5_n = max(int(total_n * 0.05), 1)
    row = {
        'target': target,
        'kind': 'docking',
        'novelty_mode': novelty_mode,
        'file': '',
        'total_rows': total_n,
        'valid_unique': valid_unique_n,
        'novel_unique': novel_n,
        'novel_ratio': novel_n / total_n if total_n > 0 else float('nan'),
        'hit_threshold': hit_thr,
        'hit_ratio': len(df[df['DOCKING'] > hit_thr]) / total_n if total_n > 0 else float('nan'),
        'top5_ds': float(df.sort_values('DOCKING', ascending=False)['DOCKING'].iloc[:top5_n].mean()) if len(df) > 0 else float('nan'),
        'top10_score_mean': float('nan'),
    }
    for mol_type in ['sac', 'ga']:
        part_total_n = len(df[df['TYPE'] == mol_type])
        part_df = df[df['TYPE'] == mol_type]
        row[f'{mol_type}_hit_ratio'] = (
            len(part_df[part_df['DOCKING'] > hit_thr]) / part_total_n
            if part_total_n > 0 else float('nan')
        )
        row[f'{mol_type}_top5_ds'] = (
            float(part_df.sort_values('DOCKING', ascending=False)['DOCKING'].iloc[:max(int(part_total_n * 0.05), 1)].mean())
            if len(part_df) > 0 and part_total_n > 0 else float('nan')
        )
    return row


def summarize_score_target(target, df):
    total_n = len(df)
    full_auc_buffer = build_auc_buffer(df, 'DOCKING')
    sac_auc_buffer = build_auc_buffer(df[df['TYPE'] == 'sac'], 'DOCKING')
    ga_auc_buffer = build_auc_buffer(df[df['TYPE'] == 'ga'], 'DOCKING')
    df = prepare_df(df)
    score_col = 'DOCKING'
    return {
        'target': target,
        'kind': 'score',
        'novelty_mode': '',
        'file': '',
        'total_rows': total_n,
        'valid_unique': len(df),
        'novel_unique': '',
        'novel_ratio': '',
        'hit_threshold': '',
        'hit_ratio': '',
        'top5_ds': '',
        'sac_hit_ratio': '',
        'sac_top5_ds': '',
        'ga_hit_ratio': '',
        'ga_top5_ds': '',
        'top10_score_mean': top_mean(df[score_col], 10),
        'top100_score_mean': top_mean(df[score_col], 100),
        'auc_top10': auc_top_n(full_auc_buffer, 10, True, 100, 10000),
        'auc_top100': auc_top_n(full_auc_buffer, 100, True, 100, 10000),
        'sac_top10_score_mean': top_mean(df[df['TYPE'] == 'sac'][score_col], 10),
        'sac_top100_score_mean': top_mean(df[df['TYPE'] == 'sac'][score_col], 100),
        'sac_auc_top10': auc_top_n(sac_auc_buffer, 10, True, 100, 10000),
        'sac_auc_top100': auc_top_n(sac_auc_buffer, 100, True, 100, 10000),
        'ga_top10_score_mean': top_mean(df[df['TYPE'] == 'ga'][score_col], 10),
        'ga_top100_score_mean': top_mean(df[df['TYPE'] == 'ga'][score_col], 100),
        'ga_auc_top10': auc_top_n(ga_auc_buffer, 10, True, 100, 10000),
        'ga_auc_top100': auc_top_n(ga_auc_buffer, 100, True, 100, 10000),
    }


def save_summary(rows, out_csv, out_txt):
    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False)
    with open(out_txt, 'w') as handle:
        handle.write(summary.to_string(index=False))
        handle.write('\n')
    print(f'Summary saved: {out_csv}')
    print(f'Summary saved: {out_txt}')


def find_files(prefix):
    files = {}
    for target in TARGETS:
        matches = sorted(glob.glob(os.path.join('results', f'{prefix}*_{target}_*_rwtarget.csv')))
        matches = [path for path in matches if not path.endswith('_attempts.csv')]
        if matches:
            files[target] = matches[-1]
    return files


def build_summary_rows(files, train_smiles, train_fps, novelty_mode):
    rows = []
    for target in TARGETS:
        path = files.get(target)
        if path is None:
            continue
        df = load_result_csv(path)
        if target in DOCKING_TARGETS:
            row = summarize_docking(target, df, train_smiles, train_fps, novelty_mode)
        else:
            row = summarize_score_target(target, df)
        row['file'] = path
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='2026-05-13-07')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--novelty_mode', type=str, default='hard', choices=['hard', 'soft'])
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join('result_eval_generation_batch', args.prefix)
    os.makedirs(out_dir, exist_ok=True)
    files = find_files(args.prefix)
    missing = [target for target in TARGETS if target not in files]
    if missing:
        print(f'[WARN] missing targets: {missing}', flush=True)
    if args.novelty_mode == 'soft':
        train_smiles, train_fps = load_train_novelty()
    else:
        train_smiles = load_train_smiles_only()
        train_fps = None
    for target in TARGETS:
        path = files.get(target)
        if path is not None:
            print(f'[OK] {target}: {path}', flush=True)
    rows = build_summary_rows(files, train_smiles, train_fps, args.novelty_mode)
    save_summary(
        rows,
        os.path.join(out_dir, f'{args.prefix}_summary.csv'),
        os.path.join(out_dir, f'{args.prefix}_summary.txt'),
    )


if __name__ == '__main__':
    main()
