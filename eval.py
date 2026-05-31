import argparse

import pandas as pd
from rdkit import Chem

from utils_sac.utils_eval import get_ncircle, get_novelty


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
    return raw


def top_mean(values, ratio):
    values = pd.to_numeric(values, errors='coerce').dropna().sort_values(ascending=False)
    if len(values) == 0:
        return float('nan')
    n = max(int(len(values) * ratio), 1)
    return float(values.iloc[:n].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('-t', '--target', type=str, default='parp1',
                        choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2'])
    args = parser.parse_args()

    df = load_result_csv(args.file_path).iloc[:3000].copy()
    total_rows = len(df)
    df['MOL'] = df['SMILES'].astype(str).apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['MOL']).copy()
    df['CANONICAL_SMILES'] = df['MOL'].apply(lambda mol: Chem.MolToSmiles(mol, isomericSmiles=False))
    df = df.drop_duplicates(subset=['CANONICAL_SMILES']).copy()

    hit_threshold = get_hit_threshold(args.target)
    hit_df = df[df['DOCKING'] > hit_threshold].copy()

    get_novelty(df)

    print(f'Number of rows:\t\t{total_rows}')
    print(f'Valid unique:\t\t{len(df)}')
    print(f'Novelty:\t\t{len(df[df["SIM"] < 0.4]) / total_rows if total_rows else float("nan")}')
    print(f'Hit ratio:\t\t{len(hit_df) / total_rows if total_rows else float("nan")}')
    print(f'Top 5% DS:\t\t{top_mean(df["DOCKING"], 0.05)}')
    print(f'#Circle:\t\t{get_ncircle(hit_df)}')


if __name__ == '__main__':
    main()
