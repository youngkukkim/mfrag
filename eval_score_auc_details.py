import argparse
import os

import numpy as np
import pandas as pd

from eval_generation_batch import AUX_TARGETS, find_files, load_result_csv, prepare_df


SCORE_TARGETS = AUX_TARGETS


def get_top10_rows(target, df, score_col='DOCKING'):
    if len(df) == 0:
        return []
    df = prepare_df(df).copy()
    df['CALL_INDEX'] = np.arange(len(df))
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df = df.dropna(subset=[score_col]).copy()
    df = df.sort_values(score_col, ascending=False).head(10)
    rows = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        rows.append({
            'target': target,
            'rank': rank,
            'score': float(row[score_col]),
            'call_index': int(row['CALL_INDEX']),
            'type': row.get('TYPE', ''),
            'smiles': row['CANONICAL_SMILES'],
        })
    return rows


def get_auc_trace_rows(target, df, score_col='DOCKING', top_n=10, freq_log=100, max_oracle_calls=10000):
    if len(df) == 0:
        return []
    df = df.copy()
    df['CALL_INDEX'] = np.arange(len(df))
    df = prepare_df(df)
    df = df.sort_values('CALL_INDEX').drop_duplicates(subset=['CANONICAL_SMILES'], keep='first').iloc[:max_oracle_calls].copy()
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df = df.dropna(subset=[score_col]).copy()
    rows = []
    prev_x = 0
    prev_y = 0.0
    auc_area = 0.0
    for checkpoint in range(freq_log, max_oracle_calls + 1, freq_log):
        seen = df[df['CALL_INDEX'] < checkpoint].copy()
        if len(seen) > 0:
            top_seen = seen.sort_values(score_col, ascending=False).head(top_n)
            top_scores = top_seen[score_col].astype(float).tolist()
            top_smiles = top_seen['CANONICAL_SMILES'].astype(str).tolist()
            top_call_indices = top_seen['CALL_INDEX'].astype(int).tolist()
            mean_top = float(np.mean(top_scores))
        else:
            top_scores = []
            top_smiles = []
            top_call_indices = []
            mean_top = 0.0
        auc_area += (checkpoint - prev_x) * (prev_y + mean_top) / 2.0
        auc_top10 = auc_area / max_oracle_calls
        prev_x = checkpoint
        prev_y = mean_top
        row = {
            'target': target,
            'checkpoint': checkpoint,
            'n_seen_unique': int(len(seen)),
            'top10_mean': mean_top,
            'auc_top10_so_far': float(auc_top10),
        }
        for i in range(top_n):
            row[f'top{i + 1}_score'] = float(top_scores[i]) if i < len(top_scores) else np.nan
            row[f'top{i + 1}_call_index'] = int(top_call_indices[i]) if i < len(top_call_indices) else ''
            row[f'top{i + 1}_smiles'] = top_smiles[i] if i < len(top_smiles) else ''
        rows.append(row)
    return rows


def summarize_from_trace(target, df, top10_rows, trace_rows):
    final_auc = trace_rows[-1]['auc_top10_so_far'] if trace_rows else np.nan
    top10_mean = float(np.mean([row['score'] for row in top10_rows])) if top10_rows else np.nan
    unique_n = len(prepare_df(df)) if len(df) > 0 else 0
    return {
        'target': target,
        'total_rows': int(len(df)),
        'valid_unique': int(unique_n),
        'top10_score_mean': top10_mean,
        'auc_top10': float(final_auc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='2026-05-13-07')
    parser.add_argument('--out_dir', type=str, default='')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join('result_eval_generation_batch', args.prefix)
    os.makedirs(out_dir, exist_ok=True)
    files = find_files(args.prefix)
    summary_rows = []
    top10_rows = []
    trace_rows = []
    for target in SCORE_TARGETS:
        path = files.get(target)
        if path is None:
            print(f'[WARN] missing target={target}', flush=True)
            continue
        df = load_result_csv(path)
        target_top10_rows = get_top10_rows(target, df)
        target_trace_rows = get_auc_trace_rows(target, df)
        summary_rows.append(summarize_from_trace(target, df, target_top10_rows, target_trace_rows))
        top10_rows.extend(target_top10_rows)
        trace_rows.extend(target_trace_rows)
        print(f'[OK] target={target} n={len(df)}', flush=True)
    base = os.path.join(out_dir, args.prefix)
    summary_path = f'{base}_score_auc_detail_summary.csv'
    top10_path = f'{base}_score_auc_detail_top10_values.csv'
    trace_path = f'{base}_score_auc_detail_trace.csv'
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(top10_rows).to_csv(top10_path, index=False)
    pd.DataFrame(trace_rows).to_csv(trace_path, index=False)
    print(f'Saved: {summary_path}')
    print(f'Saved: {top10_path}')
    print(f'Saved: {trace_path}')


if __name__ == '__main__':
    main()
