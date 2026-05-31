import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Batch

from model.mfrag import MFRAG
from utils_mfrag.data import get_graph_from_frag
from eval_generation_batch import DOCKING_TARGETS, AUX_TARGETS, TARGETS, load_result_csv, prepare_df


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


def target_value(target, raw_value):
    raw_value = float(raw_value)
    if target in DOCKING_TARGETS:
        return float(np.clip(raw_value, 0.0, 20.0)) / 20.0
    return raw_value


def cls_value(target, raw_value):
    raw_value = float(raw_value)
    if target in DOCKING_TARGETS:
        return float(raw_value > get_hit_threshold(target))
    if target == 'qed':
        return float(raw_value > 0.7)
    if target == 'sa':
        return float(raw_value > (7.0 / 9.0))
    return float(raw_value > 0.5)


def safe_corr(fn, y_true, y_pred):
    if len(y_true) < 2:
        return float('nan')
    try:
        value = fn(y_true, y_pred)[0]
    except Exception:
        value = np.nan
    return float(value)


def load_mfrag_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    model_arch = 'shared' if any(key.startswith('gather') for key in state_dict) else 'dual'
    model = MFRAG(device=device, model_arch=model_arch).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt, model_arch


def mean_pool_frag_embeddings(frag_embeddings, group_sizes):
    group_sizes = [int(x) for x in group_sizes]
    if len(group_sizes) == 0:
        return frag_embeddings.new_zeros((0, frag_embeddings.size(-1)))
    splits = torch.split(frag_embeddings, group_sizes, dim=0)
    return torch.stack([split.mean(dim=0) for split in splits], dim=0)


def batched_predict(model, graphs, device, batch_size):
    preds = []
    embeddings = []
    for start in range(0, len(graphs), batch_size):
        batch = Batch.from_data_list(graphs[start:start + batch_size]).to(device)
        with torch.no_grad():
            pred, embedding = model(batch)
        preds.append(pred.detach().cpu())
        embeddings.append(embedding.detach().cpu())
    if not preds:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)
    return (
        torch.cat(preds, dim=0).numpy().reshape(-1),
        torch.cat(embeddings, dim=0).numpy(),
    )


def batched_align_l2(model, graph_frag_items, device, batch_size):
    distances = []
    valid_n = 0
    for start in range(0, len(graph_frag_items), batch_size):
        items = graph_frag_items[start:start + batch_size]
        graphs = [x[0] for x in items]
        frag_lists = [x[1] for x in items]
        group_sizes = [len(frags) for frags in frag_lists]
        flat_frags = [frag for frags in frag_lists for frag in frags]
        if len(flat_frags) == 0:
            continue

        graph_batch = Batch.from_data_list(graphs).to(device)
        frag_batch = Batch.from_data_list(flat_frags).to(device)
        with torch.no_grad():
            _, graph_embedding = model(graph_batch)
            frag_embedding = model.encode_frag(frag_batch)
            pooled = mean_pool_frag_embeddings(frag_embedding, group_sizes)
            distance = F.pairwise_distance(graph_embedding, pooled)
        distances.extend(distance.detach().cpu().numpy().astype(float).tolist())
        valid_n += len(items)

    if len(distances) == 0:
        return {
            'align_n': 0,
            'align_l2_mean': float('nan'),
            'align_l2_median': float('nan'),
            'align_l2_std': float('nan'),
        }
    values = np.asarray(distances, dtype=np.float32)
    return {
        'align_n': int(valid_n),
        'align_l2_mean': float(values.mean()),
        'align_l2_median': float(np.median(values)),
        'align_l2_std': float(values.std()),
    }


def prediction_metrics(y_true, y_pred, label_mode='reg'):
    row = {'n': int(len(y_true))}
    if len(y_true) == 0:
        row.update({
            'rmse': float('nan'),
            'mae': float('nan'),
            'r2': float('nan'),
            'pearson': float('nan'),
            'spearman': float('nan'),
            'accuracy': float('nan'),
        })
        return row

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    row.update({
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float('nan'),
        'pearson': safe_corr(pearsonr, y_true, y_pred),
        'spearman': safe_corr(spearmanr, y_true, y_pred),
    })
    if label_mode == 'cls':
        pred_label = (1.0 / (1.0 + np.exp(-y_pred)) >= 0.5).astype(np.float32)
        row['accuracy'] = float((pred_label == y_true).mean())
    else:
        row['accuracy'] = float('nan')
    return row


def load_zinc_split(target, split='test', max_molecules=0, seed=0):
    train_frag, test_frag = torch.load('data/zinc250k_frag.pt')
    dataset = test_frag if split == 'test' else train_frag
    if max_molecules and len(dataset) > max_molecules:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=max_molecules, replace=False)
    else:
        indices = np.arange(len(dataset))

    graphs = []
    graph_frag_items = []
    y_reg = []
    y_cls = []
    smiles = []
    for idx in indices:
        graph, frag_list, value = dataset[int(idx)]
        graphs.append(graph)
        graph_frag_items.append((graph, frag_list))
        raw = float(value[target])
        y_reg.append(target_value(target, raw))
        y_cls.append(cls_value(target, raw))
        smiles.append(value.get('smiles', ''))
    return {
        'graphs': graphs,
        'graph_frag_items': graph_frag_items,
        'y_reg': np.asarray(y_reg, dtype=np.float32),
        'y_cls': np.asarray(y_cls, dtype=np.float32),
        'smiles': smiles,
    }


def load_generated(prefix, target, max_molecules=0, seed=0):
    matches = sorted(glob.glob(os.path.join('results', f'{prefix}*_{target}_*_rwtarget.csv')))
    matches = [path for path in matches if not path.endswith('_attempts.csv')]
    if not matches:
        return None, ''
    path = matches[-1]
    df = load_result_csv(path)
    df = prepare_df(df)
    if max_molecules and len(df) > max_molecules:
        df = df.sample(n=max_molecules, random_state=seed).copy()

    graphs = []
    y_reg = []
    y_cls = []
    smiles = []
    for _, row in df.iterrows():
        graph = get_graph_from_frag(row['CANONICAL_SMILES'])
        if graph is None:
            continue
        raw = float(row['DOCKING'])
        graphs.append(graph)
        y_reg.append(target_value(target, raw))
        y_cls.append(cls_value(target, raw))
        smiles.append(row['CANONICAL_SMILES'])
    return {
        'graphs': graphs,
        'graph_frag_items': [],
        'y_reg': np.asarray(y_reg, dtype=np.float32),
        'y_cls': np.asarray(y_cls, dtype=np.float32),
        'smiles': smiles,
    }, path


def runtime_ckpts(target, prefix, mode='latest'):
    pattern = os.path.join('ckpt', 'runtime_mfrag', target, f'{prefix}*', 'finetune_*.pt')
    paths = sorted(glob.glob(pattern))
    if mode == 'none':
        return []
    if mode == 'latest' and paths:
        return [paths[-1]]
    return paths


def finetune_index(path):
    match = re.search(r'finetune_(\d+)_n(\d+)\.pt$', os.path.basename(path))
    if not match:
        return '', ''
    return int(match.group(1)), int(match.group(2))


def build_ckpt_list(target, prefix, ckpt_mode):
    rows = []
    base = os.path.join('ckpt', 'only2', 'reg', target, 'best.pt')
    if os.path.exists(base):
        rows.append(('base_only2', base, '', ''))
    for path in runtime_ckpts(target, prefix, ckpt_mode):
        idx, bank_n = finetune_index(path)
        rows.append((f'finetune_{idx:03d}', path, idx, bank_n))
    return rows


def evaluate_ckpt(model, target, dataset_name, payload, label_mode, device, batch_size):
    y_true = payload['y_cls'] if label_mode == 'cls' else payload['y_reg']
    preds, _ = batched_predict(model, payload['graphs'], device, batch_size)
    row = prediction_metrics(y_true, preds, label_mode=label_mode)
    if payload['graph_frag_items']:
        row.update(batched_align_l2(model, payload['graph_frag_items'], device, batch_size))
    else:
        row.update({
            'align_n': 0,
            'align_l2_mean': float('nan'),
            'align_l2_median': float('nan'),
            'align_l2_std': float('nan'),
        })
    row['target'] = target
    row['dataset'] = dataset_name
    return row


def parse_targets(raw_targets):
    if raw_targets == ['all']:
        return TARGETS
    return raw_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='2026-05-13-07')
    parser.add_argument('--targets', nargs='+', default=['all'], choices=['all'] + TARGETS)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--ckpt_mode', type=str, default='latest', choices=['latest', 'all', 'none'])
    parser.add_argument('--datasets', nargs='+', default=['zinc_test', 'generated'],
                        choices=['zinc_test', 'generated'])
    parser.add_argument('--max_zinc_test', type=int, default=0)
    parser.add_argument('--max_generated', type=int, default=0)
    parser.add_argument('--label_mode', type=str, default='reg', choices=['reg', 'cls'])
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    targets = parse_targets(args.targets)
    out_dir = args.out_dir or os.path.join('result_eval_generation_batch', args.prefix)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for target in targets:
        dataset_payloads = {}
        if 'zinc_test' in args.datasets:
            dataset_payloads['zinc_test'] = load_zinc_split(
                target,
                split='test',
                max_molecules=args.max_zinc_test,
                seed=args.seed,
            )
        if 'generated' in args.datasets:
            generated_payload, generated_path = load_generated(
                args.prefix,
                target,
                max_molecules=args.max_generated,
                seed=args.seed,
            )
            if generated_payload is not None:
                dataset_payloads['generated'] = generated_payload
            else:
                print(f'[WARN] missing generated csv target={target}', flush=True)

        for ckpt_name, ckpt_path, ft_idx, bank_n in build_ckpt_list(target, args.prefix, args.ckpt_mode):
            if not os.path.exists(ckpt_path):
                print(f'[WARN] missing ckpt target={target} path={ckpt_path}', flush=True)
                continue
            model, ckpt, model_arch = load_mfrag_model(ckpt_path, device)
            for dataset_name, payload in dataset_payloads.items():
                row = evaluate_ckpt(
                    model,
                    target,
                    dataset_name,
                    payload,
                    args.label_mode,
                    device,
                    args.batch_size,
                )
                row.update({
                    'checkpoint': ckpt_name,
                    'checkpoint_path': ckpt_path,
                    'model_arch': model_arch,
                    'finetune_idx': ft_idx,
                    'generated_bank_size': bank_n,
                    'source_ckpt': ckpt.get('source_ckpt', '') if isinstance(ckpt, dict) else '',
                })
                rows.append(row)
                print(
                    f"[OK] target={target} ckpt={ckpt_name} dataset={dataset_name} "
                    f"n={row['n']} rmse={row['rmse']:.6f} spearman={row['spearman']:.6f} "
                    f"align_l2={row['align_l2_mean']}",
                    flush=True,
                )
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, f'{args.prefix}_runtime_mfrag_eval.csv')
    out_txt = os.path.join(out_dir, f'{args.prefix}_runtime_mfrag_eval.txt')
    summary.to_csv(out_csv, index=False)
    with open(out_txt, 'w') as f:
        f.write(summary.to_string(index=False))
        f.write('\n')
    print(f'Saved: {out_csv}')
    print(f'Saved: {out_txt}')


if __name__ == '__main__':
    main()
