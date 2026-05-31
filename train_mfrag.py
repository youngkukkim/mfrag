import argparse
import gc
import json
import os
import random
import time
from typing import Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.data import Batch

from model.mfrag import MFRAG
from utils_sac.utils import set_seed


DOCKING_TARGETS = ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']
MPO_TARGETS = [
    'amlodipine_mpo',
    'fexofenadine_mpo',
    'osimertinib_mpo',
    'perindopril_mpo',
    'ranolazine_mpo',
    'sitagliptin_mpo',
    'zaleplon_mpo',
]
AUX_TARGETS = ['qed', 'sa']
TARGETS = DOCKING_TARGETS + MPO_TARGETS + AUX_TARGETS


def get_hit_threshold(target: str) -> float:
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


def build_targets(target_name: str, raw_target: float) -> Tuple[float, float]:
    if target_name in DOCKING_TARGETS:
        reg_target = np.clip(raw_target, 0.0, 20.0) / 20.0
        cls_target = float(raw_target > get_hit_threshold(target_name))
    elif target_name == 'qed':
        reg_target = raw_target
        cls_target = float(raw_target > 0.7)
    elif target_name == 'sa':
        reg_target = raw_target
        cls_target = float(raw_target > (7.0 / 9.0))
    else:
        reg_target = raw_target
        cls_target = float(raw_target > 0.5)
    return float(reg_target), float(cls_target)


class DockingDataset(Dataset):
    def __init__(self, dataset, target: str):
        self.dataset = dataset
        self.target = target
        self.reg_target_values = []
        self.cls_target_values = []
        for _, _, value in self.dataset:
            raw_target = float(value[self.target])
            reg_target, cls_target = build_targets(self.target, raw_target)
            self.reg_target_values.append(reg_target)
            self.cls_target_values.append(cls_target)
        self.reg_target_values = np.asarray(self.reg_target_values, dtype=np.float32)
        self.cls_target_values = np.asarray(self.cls_target_values, dtype=np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        graph, frag_list, value = self.dataset[idx]
        frags_num = len(frag_list)
        raw_target = float(value[self.target])
        reg_target, cls_target = build_targets(self.target, raw_target)
        return (
            graph,
            frag_list,
            frags_num,
            torch.tensor(reg_target, dtype=torch.float32),
            torch.tensor(cls_target, dtype=torch.float32),
        )


def collate_mfrag_batch(samples):
    graphs, frag_lists, frags_num, reg_target, cls_target = zip(*samples)
    flat_frags = [frag for frag_list in frag_lists for frag in frag_list]
    graph_batch = Batch.from_data_list(list(graphs))
    frag_batch = Batch.from_data_list(flat_frags) if len(flat_frags) > 0 else None
    return (
        graph_batch,
        frag_batch,
        torch.tensor(frags_num, dtype=torch.long),
        torch.stack(reg_target, dim=0),
        torch.stack(cls_target, dim=0),
    )


def mean_pool_frag_embeddings(frag_embeddings: torch.Tensor, frag_group_sizes) -> torch.Tensor:
    if torch.is_tensor(frag_group_sizes):
        group_sizes = [int(x) for x in frag_group_sizes.view(-1).tolist()]
    else:
        group_sizes = [int(x) for x in frag_group_sizes]

    if len(group_sizes) == 0:
        return frag_embeddings.new_zeros((0, frag_embeddings.size(-1)))

    splits = torch.split(frag_embeddings, group_sizes, dim=0)
    pooled = [split.mean(dim=0) for split in splits]
    return torch.stack(pooled, dim=0)


class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim_matrix = torch.clamp(sim_matrix, max=78.0)

        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float()
        eye = torch.eye(labels.size(0), device=embeddings.device)
        positive_mask = positive_mask * (1.0 - eye)

        exp_sim = torch.exp(sim_matrix) * (1.0 - eye)
        denom = exp_sim.sum(dim=1).clamp_min(1e-12)
        pos = (exp_sim * positive_mask).sum(dim=1)

        valid = pos > 0
        if not torch.any(valid):
            return embeddings.new_tensor(0.0)
        loss = -torch.log((pos[valid] / denom[valid]).clamp_min(1e-12))
        return loss.mean()


class ContinuousContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.1, label_temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.label_temperature = label_temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim_matrix = torch.clamp(sim_matrix, max=78.0)

        labels = labels.view(-1, 1).float()
        label_diff = torch.abs(labels - labels.T)
        pos_weights = torch.exp(-label_diff / self.label_temperature)
        eye = torch.eye(labels.size(0), device=embeddings.device)
        pos_weights = pos_weights * (1.0 - eye)

        exp_sim = torch.exp(sim_matrix) * (1.0 - eye)
        denom = exp_sim.sum(dim=1).clamp_min(1e-12)
        pos = (exp_sim * pos_weights).sum(dim=1)

        valid = pos > 0
        if not torch.any(valid):
            return embeddings.new_tensor(0.0)
        loss = -torch.log((pos[valid] / denom[valid]).clamp_min(1e-12))
        return loss.mean()


def build_contrastive_bin_edges(values: np.ndarray, num_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if num_bins <= 1 or len(values) == 0:
        return np.zeros((0,), dtype=np.float32)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    edges = np.quantile(values, quantiles).astype(np.float32)
    return np.unique(edges)


def bucketize_targets(targets: torch.Tensor, bin_edges: np.ndarray) -> torch.Tensor:
    targets = targets.view(-1)
    if len(bin_edges) == 0:
        return torch.zeros(targets.size(0), dtype=torch.long, device=targets.device)
    edges = torch.as_tensor(bin_edges, dtype=targets.dtype, device=targets.device)
    return torch.bucketize(targets, edges)


def safe_corr(fn, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float('nan')
    try:
        return float(fn(y_true, y_pred)[0])
    except Exception:
        return float('nan')


def safe_binary_metric(fn, y_true, y_score):
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return float('nan')


def build_plot_indices(dataset_size: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size <= 0 or dataset_size == 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.RandomState(seed)
    size = min(sample_size, dataset_size)
    return rng.choice(dataset_size, size=size, replace=False)


def build_nonzero_plot_indices(dataset: DockingDataset, sample_size: int, seed: int) -> np.ndarray:
    values = dataset.reg_target_values
    valid_indices = np.nonzero(values != 0)[0]
    if sample_size <= 0 or len(valid_indices) == 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.RandomState(seed)
    size = min(sample_size, len(valid_indices))
    return rng.choice(valid_indices, size=size, replace=False)


def release_cuda_cache(device: torch.device):
    if device.type == 'cuda':
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def resolve_mol_ckpt_path(args) -> str:
    if args.mol_ckpt:
        return args.mol_ckpt
    if args.mol_ckpt_root:
        return os.path.join(args.mol_ckpt_root, args.label_mode, args.target, 'best.pt')
    return ''


def load_mol_predictor_checkpoint(model: MFRAG, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('mol_encoder.') or key.startswith('value_predictor.'):
            remapped[key] = value
        elif key.startswith('gather.') or key.startswith('embed.'):
            remapped[f'mol_encoder.{key}'] = value
    load_result = model.load_state_dict(remapped, strict=False)
    return load_result


def freeze_mol_predictor(model: MFRAG):
    if getattr(model, 'model_arch', 'dual') == 'shared':
        for param in model.gather.parameters():
            param.requires_grad = False
        for param in model.embed.parameters():
            param.requires_grad = False
        for param in model.value_predictor.parameters():
            param.requires_grad = False
        return
    for param in model.mol_encoder.parameters():
        param.requires_grad = False
    for param in model.value_predictor.parameters():
        param.requires_grad = False


def freeze_mol_gather(model: MFRAG):
    if getattr(model, 'model_arch', 'dual') == 'shared':
        for param in model.gather.parameters():
            param.requires_grad = False
        return
    for param in model.mol_encoder.gather.parameters():
        param.requires_grad = False


def collect_plot_embeddings(model: MFRAG, dataset: DockingDataset, indices: np.ndarray,
                            device: torch.device, batch_size: int,
                            fragment_sample_size: int):
    model.eval()
    graph_embeddings = []
    graph_scores = []
    frag_embeddings = []
    frag_scores = []
    frag_keys = []
    fallback_key_id = 0

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        samples = [dataset[int(i)] for i in batch_indices]
        batch_frag_keys = []
        for _, frag_list, _, _, _ in samples:
            for frag in frag_list:
                smiles = getattr(frag, 'smiles', None)
                if smiles is None:
                    smiles = f"__frag_{fallback_key_id}"
                    fallback_key_id += 1
                batch_frag_keys.append(str(smiles))
        graph_batch, frag_batch, frags_num, reg_target, _ = collate_mfrag_batch(samples)
        graph_batch = graph_batch.to(device, non_blocking=True)
        if frag_batch is not None:
            frag_batch = frag_batch.to(device, non_blocking=True)
        with torch.inference_mode():
            _, graph_embedding = model(graph_batch)
            if frag_batch is not None:
                frag_embedding = model.encode_frag(frag_batch)
            else:
                frag_embedding = graph_embedding.new_zeros((0, graph_embedding.size(-1)))
        graph_embeddings.append(graph_embedding.cpu())
        frag_embeddings.append(frag_embedding.cpu())
        graph_scores.append(reg_target.cpu())
        repeated_scores = torch.repeat_interleave(reg_target.cpu(), frags_num.cpu())
        frag_scores.append(repeated_scores)
        frag_keys.extend(batch_frag_keys)

    if not graph_embeddings:
        return None

    graph_embeddings = torch.cat(graph_embeddings, dim=0).numpy()
    graph_scores = torch.cat(graph_scores, dim=0).numpy().reshape(-1)
    frag_embeddings = torch.cat(frag_embeddings, dim=0).numpy()
    frag_scores = torch.cat(frag_scores, dim=0).numpy().reshape(-1)

    if len(frag_keys) == len(frag_scores):
        key_to_idx = {}
        agg_embeddings = []
        agg_scores = []
        agg_counts = []
        for emb, score, key in zip(frag_embeddings, frag_scores, frag_keys):
            idx = key_to_idx.get(key)
            if idx is None:
                key_to_idx[key] = len(agg_embeddings)
                agg_embeddings.append(emb.copy())
                agg_scores.append(float(score))
                agg_counts.append(1)
            else:
                agg_embeddings[idx] += emb
                agg_scores[idx] += float(score)
                agg_counts[idx] += 1
        counts = np.asarray(agg_counts, dtype=np.float32).reshape(-1, 1)
        frag_embeddings = np.asarray(agg_embeddings, dtype=np.float32) / counts
        frag_scores = np.asarray(agg_scores, dtype=np.float32) / np.asarray(agg_counts, dtype=np.float32)

    if fragment_sample_size > 0 and len(frag_scores) > fragment_sample_size:
        rng = np.random.RandomState(0)
        keep = rng.choice(len(frag_scores), size=fragment_sample_size, replace=False)
        frag_embeddings = frag_embeddings[keep]
        frag_scores = frag_scores[keep]

    return graph_embeddings, graph_scores, frag_embeddings, frag_scores


def plot_embedding_panels(axes, fig, split_name: str, graph_embeddings: np.ndarray, graph_scores: np.ndarray,
                          frag_embeddings: np.ndarray, frag_scores: np.ndarray):
    all_embeddings = np.concatenate([graph_embeddings, frag_embeddings], axis=0)
    coords = PCA(n_components=2, random_state=0).fit_transform(all_embeddings)
    graph_coords = coords[:len(graph_embeddings)]
    frag_coords = coords[len(graph_embeddings):]
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_pad = max((x_max - x_min) * 0.05, 1e-6)
    y_pad = max((y_max - y_min) * 0.05, 1e-6)

    mol_ax, frag_ax = axes
    mol_sc = mol_ax.scatter(
        graph_coords[:, 0],
        graph_coords[:, 1],
        c=graph_scores,
        s=16,
        alpha=1.0,
        cmap='coolwarm',
        marker='o',
        edgecolors='black',
        linewidths=0.25,
    )
    mol_ax.set_title(f"{split_name} molecules (n={len(graph_scores)})")
    mol_ax.set_xlabel('PC1')
    mol_ax.set_ylabel('PC2')
    mol_ax.set_xlim(x_min - x_pad, x_max + x_pad)
    mol_ax.set_ylim(y_min - y_pad, y_max + y_pad)
    fig.colorbar(mol_sc, ax=mol_ax, label='molecule score', fraction=0.046, pad=0.04)

    frag_sc = frag_ax.scatter(
        frag_coords[:, 0],
        frag_coords[:, 1],
        c=frag_scores,
        s=14,
        alpha=1.0,
        cmap='coolwarm',
        marker='^',
        edgecolors='black',
        linewidths=0.15,
    )
    frag_ax.set_title(f"{split_name} fragments (n={len(frag_scores)})")
    frag_ax.set_xlabel('PC1')
    frag_ax.set_ylabel('PC2')
    frag_ax.set_xlim(x_min - x_pad, x_max + x_pad)
    frag_ax.set_ylim(y_min - y_pad, y_max + y_pad)
    fig.colorbar(frag_sc, ax=frag_ax, label='fragment mean parent score', fraction=0.046, pad=0.04)


def save_embedding_pca_plot(model: MFRAG, train_dataset: DockingDataset, train_indices: np.ndarray,
                            test_dataset: DockingDataset, test_indices: np.ndarray,
                            device: torch.device, batch_size: int, fragment_sample_size: int,
                            out_path: str) -> float:
    start_time = time.time()
    train_payload = collect_plot_embeddings(
        model,
        train_dataset,
        train_indices,
        device,
        batch_size,
        fragment_sample_size,
    )
    test_payload = collect_plot_embeddings(
        model,
        test_dataset,
        test_indices,
        device,
        batch_size,
        fragment_sample_size,
    )
    if train_payload is None or test_payload is None:
        return 0.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    plot_embedding_panels(axes[0], fig, 'Train', *train_payload)
    plot_embedding_panels(axes[1], fig, 'Test', *test_payload)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return time.time() - start_time


def evaluate(model: MFRAG, loader: DataLoader, device: torch.device, pred_loss_fn,
             label_mode: str, distance_mode: str, frag_distance_weight: float,
             frag_ctr_weight: float, ctr_mode: str, contrastive_loss_fn,
             contrastive_bin_edges: np.ndarray, train_mode: str) -> Tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    total_pred_loss = 0.0
    total_distance_loss = 0.0
    total_ctr_loss = 0.0
    preds = []
    reg_targets = []
    cls_targets = []

    with torch.inference_mode():
        for graph_batch, frag_batch, frags_num_list, reg_target, cls_target in loader:
            graph_batch = graph_batch.to(device, non_blocking=True)
            if frag_batch is not None:
                frag_batch = frag_batch.to(device, non_blocking=True)
            reg_target = reg_target.to(device, non_blocking=True).view(-1, 1)
            cls_target = cls_target.to(device, non_blocking=True).view(-1, 1)
            pred, graph_embedding = model(graph_batch)
            if label_mode == 'reg':
                pred_loss = pred_loss_fn(pred, reg_target)
                ctr_source = reg_target.view(-1)
            else:
                pred_loss = pred_loss_fn(pred, cls_target)
                ctr_source = cls_target.view(-1)

            use_frag_loss = (distance_mode == 'l2' or ctr_mode != 'none') and frag_batch is not None
            if use_frag_loss:
                frag_embeddings = model.encode_frag(frag_batch)
                pooled_frag_embeddings = mean_pool_frag_embeddings(frag_embeddings, frags_num_list)
                graph_embedding_target = graph_embedding.detach()
            if distance_mode == 'l2' and use_frag_loss:
                distance_loss = F.pairwise_distance(graph_embedding_target, pooled_frag_embeddings).mean()
            else:
                distance_loss = graph_embedding.new_tensor(0.0)
            if ctr_mode == 'cls' and use_frag_loss:
                ctr_labels = bucketize_targets(ctr_source, contrastive_bin_edges)
                contrastive_embeddings = torch.cat([graph_embedding_target, pooled_frag_embeddings], dim=0)
                contrastive_targets = torch.cat([ctr_labels, ctr_labels], dim=0)
                ctr_loss = contrastive_loss_fn(contrastive_embeddings, contrastive_targets)
            elif ctr_mode == 'reg' and use_frag_loss:
                contrastive_embeddings = torch.cat([graph_embedding_target, pooled_frag_embeddings], dim=0)
                contrastive_targets = torch.cat([ctr_source, ctr_source], dim=0)
                ctr_loss = contrastive_loss_fn(contrastive_embeddings, contrastive_targets)
            else:
                ctr_loss = graph_embedding.new_tensor(0.0)
            frag_loss = frag_distance_weight * distance_loss + frag_ctr_weight * ctr_loss
            if train_mode == 'frag_only':
                loss = frag_loss
            else:
                loss = pred_loss + frag_loss
            total_loss += float(loss.item())
            total_pred_loss += float(pred_loss.item())
            total_distance_loss += float(distance_loss.item())
            total_ctr_loss += float(ctr_loss.item())
            preds.append(pred.view(-1).cpu())
            reg_targets.append(reg_target.view(-1).cpu())
            cls_targets.append(cls_target.view(-1).cpu())

    preds = torch.cat(preds).numpy() if preds else np.zeros((0,), dtype=np.float32)
    reg_targets = torch.cat(reg_targets).numpy() if reg_targets else np.zeros((0,), dtype=np.float32)
    cls_targets = torch.cat(cls_targets).numpy() if cls_targets else np.zeros((0,), dtype=np.float32)
    metrics = {
        'pred_loss': total_pred_loss / max(len(loader), 1),
        'distance_loss': total_distance_loss / max(len(loader), 1),
        'ctr_loss': total_ctr_loss / max(len(loader), 1),
    }
    if label_mode == 'reg':
        mse = float(np.mean((preds - reg_targets) ** 2)) if len(preds) > 0 else float('nan')
        metrics.update({
            'rmse': float(np.sqrt(mse)) if len(preds) > 0 else float('nan'),
            'mae': float(np.mean(np.abs(preds - reg_targets))) if len(preds) > 0 else float('nan'),
            'pearson': safe_corr(pearsonr, reg_targets, preds),
            'spearman': safe_corr(spearmanr, reg_targets, preds),
        })
    else:
        probs = 1.0 / (1.0 + np.exp(-preds))
        pred_labels = (probs > 0.5).astype(np.int64)
        true_labels = cls_targets.astype(np.int64)
        metrics.update({
            'accuracy': float(accuracy_score(true_labels, pred_labels)),
            'precision': float(precision_score(true_labels, pred_labels, zero_division=0)),
            'recall': float(recall_score(true_labels, pred_labels, zero_division=0)),
            'f1': float(f1_score(true_labels, pred_labels, zero_division=0)),
            'auroc': safe_binary_metric(roc_auc_score, true_labels, probs),
            'auprc': safe_binary_metric(average_precision_score, true_labels, probs),
        })
    return total_loss / max(len(loader), 1), metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=int, default=0)
    parser.add_argument('-t', '--target', type=str, default='parp1', choices=TARGETS)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=1e-5)
    parser.add_argument('--label_mode', type=str, default='reg', choices=['reg', 'cls'])
    parser.add_argument('--model_arch', type=str, default='dual', choices=['shared', 'dual'])
    parser.add_argument('--train_mode', type=str, default='joint', choices=['joint', 'frag_only'])
    parser.add_argument('--delta', type=float, default=1.0,
                        help='Huber loss delta.')
    parser.add_argument('--distance_mode', type=str, default='l2', choices=['none', 'l2'])
    parser.add_argument('--frag_distance_weight', type=float, default=0.1)
    parser.add_argument('--frag_ctr_weight', type=float, default=0.1)
    parser.add_argument('--ctr_mode', type=str, default='reg', choices=['none', 'reg', 'cls'])
    parser.add_argument('--ctr_bins', type=int, default=2)
    parser.add_argument('--ctr_temperature', type=float, default=0.1)
    parser.add_argument('--ctr_reg_label_temperature', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--plot_embedding', action='store_true')
    parser.add_argument('--plot_every', type=int, default=5)
    parser.add_argument('--plot_sample_size', type=int, default=3000)
    parser.add_argument('--plot_test_sample_size', type=int, default=3000)
    parser.add_argument('--plot_fragment_sample_size', type=int, default=10000)
    parser.add_argument('--out_root', type=str, default='ckpt/only')
    parser.add_argument('--mol_ckpt', type=str, default='')
    parser.add_argument('--mol_ckpt_root', type=str, default='')
    parser.add_argument('--freeze_mol_encoder', action='store_true')
    parser.add_argument('--freeze_mol_gather', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.eval_every < 1:
        raise ValueError('--eval_every must be >= 1')
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    target_dir = os.path.join(args.out_root, args.label_mode, args.target)
    os.makedirs(target_dir, exist_ok=True)

    data_load_start = time.time()
    data_start_line = "[Data] loading data/zinc250k_frag.pt"
    print(data_start_line, flush=True)
    with open(os.path.join(target_dir, 'log.log'), 'a') as f:
        f.write(data_start_line + '\n')
    train_frag, test_frag = torch.load('data/zinc250k_frag.pt')
    train_data = DockingDataset(train_frag, args.target)
    test_data = DockingDataset(test_frag, args.target)
    data_elapsed = time.time() - data_load_start
    data_done_line = (
        f"[Data] loaded train={len(train_data)} test={len(test_data)} "
        f"elapsed={data_elapsed // 60:.0f}m {data_elapsed % 60:.1f}s"
    )
    print(data_done_line, flush=True)
    with open(os.path.join(target_dir, 'log.log'), 'a') as f:
        f.write(data_done_line + '\n')

    loader_kwargs = {
        'num_workers': args.num_workers,
        'collate_fn': collate_mfrag_batch,
        'pin_memory': device.type == 'cuda',
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    if args.train_mode == 'frag_only' and args.model_arch == 'shared':
        raise ValueError('frag_only requires --model_arch dual')
    if args.train_mode == 'frag_only' and args.distance_mode == 'none' and args.ctr_mode == 'none':
        raise ValueError('frag_only needs --distance_mode l2 or --ctr_mode reg/cls')

    model = MFRAG(device=device, model_arch=args.model_arch).to(device)
    mol_ckpt_path = resolve_mol_ckpt_path(args)
    if mol_ckpt_path:
        load_result = load_mol_predictor_checkpoint(model, mol_ckpt_path, device)
        load_line = (
            f"[Mol ckpt] loaded={mol_ckpt_path} "
            f"missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
        )
        print(load_line, flush=True)
        with open(os.path.join(target_dir, 'log.log'), 'a') as f:
            f.write(load_line + '\n')
    if args.freeze_mol_encoder or args.train_mode == 'frag_only':
        freeze_mol_predictor(model)
        freeze_line = "[Freeze] mol_encoder=True value_predictor=True"
        print(freeze_line, flush=True)
        with open(os.path.join(target_dir, 'log.log'), 'a') as f:
            f.write(freeze_line + '\n')
    elif args.freeze_mol_gather:
        freeze_mol_gather(model)
        freeze_line = "[Freeze] mol_encoder.gather=True mol_encoder.embed=False value_predictor=False"
        print(freeze_line, flush=True)
        with open(os.path.join(target_dir, 'log.log'), 'a') as f:
            f.write(freeze_line + '\n')

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError('No trainable parameters.')
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, mode='min', verbose=True)
    if args.label_mode == 'reg':
        pred_loss_fn = torch.nn.HuberLoss(delta=args.delta)
        contrastive_source_values = train_data.reg_target_values
    else:
        pred_loss_fn = torch.nn.BCEWithLogitsLoss()
        contrastive_source_values = train_data.cls_target_values

    if args.ctr_mode == 'cls':
        contrastive_loss_fn = SupervisedContrastiveLoss(temperature=args.ctr_temperature)
        contrastive_bin_edges = build_contrastive_bin_edges(contrastive_source_values, args.ctr_bins)
    elif args.ctr_mode == 'reg':
        contrastive_loss_fn = ContinuousContrastiveLoss(
            temperature=args.ctr_temperature,
            label_temperature=args.ctr_reg_label_temperature,
        )
        contrastive_bin_edges = np.zeros((0,), dtype=np.float32)
    else:
        contrastive_loss_fn = None
        contrastive_bin_edges = np.zeros((0,), dtype=np.float32)

    with open(os.path.join(target_dir, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    train_plot_indices = build_nonzero_plot_indices(train_data, args.plot_sample_size, args.seed)
    test_plot_indices = build_nonzero_plot_indices(test_data, args.plot_test_sample_size, args.seed)

    start_time = time.time()
    best_test_loss = float('inf')
    stale_eval_count = 0
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.plot_embedding:
        plot_path = os.path.join(target_dir, 'plots', 'epoch_000_pca.png')
        plot_elapsed = save_embedding_pca_plot(
            model,
            train_data,
            train_plot_indices,
            test_data,
            test_plot_indices,
            device,
            args.batch_size,
            args.plot_fragment_sample_size,
            plot_path,
        )
        plot_line = f"[Plot] epoch=000 path={plot_path} elapsed={plot_elapsed:.1f}s"
        print(plot_line, flush=True)
        with open(os.path.join(target_dir, 'log.log'), 'a') as f:
            f.write(plot_line + '\n')
        release_cuda_cache(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_pred_loss = 0.0
        train_distance_loss = 0.0
        train_ctr_loss = 0.0
        for graph_batch, frag_batch, frags_num_list, reg_target, cls_target in train_loader:
            graph_batch = graph_batch.to(device, non_blocking=True)
            if frag_batch is not None:
                frag_batch = frag_batch.to(device, non_blocking=True)
            reg_target = reg_target.to(device, non_blocking=True).view(-1, 1)
            cls_target = cls_target.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                if args.train_mode == 'frag_only':
                    with torch.no_grad():
                        pred, graph_embedding = model(graph_batch)
                else:
                    pred, graph_embedding = model(graph_batch)
                if args.label_mode == 'reg':
                    pred_loss = pred_loss_fn(pred, reg_target)
                    ctr_source = reg_target.view(-1)
                else:
                    pred_loss = pred_loss_fn(pred, cls_target)
                    ctr_source = cls_target.view(-1)

                use_frag_loss = (args.distance_mode == 'l2' or args.ctr_mode != 'none') and frag_batch is not None
                if use_frag_loss:
                    frag_embeddings = model.encode_frag(frag_batch)
                    pooled_frag_embeddings = mean_pool_frag_embeddings(frag_embeddings, frags_num_list)
                    graph_embedding_target = graph_embedding.detach()
                if args.distance_mode == 'l2' and use_frag_loss:
                    distance_loss = F.pairwise_distance(graph_embedding_target, pooled_frag_embeddings).mean()
                else:
                    distance_loss = graph_embedding.new_tensor(0.0)
                if args.ctr_mode == 'cls' and use_frag_loss:
                    contrastive_labels = bucketize_targets(ctr_source, contrastive_bin_edges)
                    contrastive_embeddings = torch.cat([graph_embedding_target, pooled_frag_embeddings], dim=0)
                    contrastive_targets = torch.cat([contrastive_labels, contrastive_labels], dim=0)
                    ctr_loss = contrastive_loss_fn(contrastive_embeddings, contrastive_targets)
                elif args.ctr_mode == 'reg' and use_frag_loss:
                    contrastive_embeddings = torch.cat([graph_embedding_target, pooled_frag_embeddings], dim=0)
                    contrastive_targets = torch.cat([ctr_source, ctr_source], dim=0)
                    ctr_loss = contrastive_loss_fn(contrastive_embeddings, contrastive_targets)
                else:
                    ctr_loss = graph_embedding.new_tensor(0.0)
                frag_loss = args.frag_distance_weight * distance_loss + args.frag_ctr_weight * ctr_loss
                if args.train_mode == 'frag_only':
                    loss = frag_loss
                else:
                    loss = pred_loss + frag_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_pred_loss += float(pred_loss.item())
            train_distance_loss += float(distance_loss.item())
            train_ctr_loss += float(ctr_loss.item())

        train_loss /= max(len(train_loader), 1)
        train_pred_loss /= max(len(train_loader), 1)
        train_distance_loss /= max(len(train_loader), 1)
        train_ctr_loss /= max(len(train_loader), 1)
        should_eval = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            test_loss, test_metrics = evaluate(
                model,
                test_loader,
                device,
                pred_loss_fn,
                args.label_mode,
                args.distance_mode,
                args.frag_distance_weight,
                args.frag_ctr_weight,
                args.ctr_mode,
                contrastive_loss_fn,
                contrastive_bin_edges,
                args.train_mode,
            )
            scheduler.step(test_loss)
            release_cuda_cache(device)

            if args.label_mode == 'reg':
                tail_metrics = (
                    f"rmse={test_metrics['rmse']:.6f} "
                    f"mae={test_metrics['mae']:.6f} "
                    f"pearson={test_metrics['pearson']:.6f} "
                    f"spearman={test_metrics['spearman']:.6f}"
                )
            else:
                tail_metrics = (
                    f"acc={test_metrics['accuracy']:.6f} "
                    f"precision={test_metrics['precision']:.6f} "
                    f"recall={test_metrics['recall']:.6f} "
                    f"f1={test_metrics['f1']:.6f} "
                    f"auroc={test_metrics['auroc']:.6f} "
                    f"auprc={test_metrics['auprc']:.6f}"
                )
        else:
            test_loss = float('nan')
            test_metrics = {'pred_loss': float('nan'), 'distance_loss': float('nan'), 'ctr_loss': float('nan')}
            tail_metrics = "eval=skipped"
        will_early_stop = (
            should_eval
            and not (test_loss < best_test_loss - args.min_delta)
            and args.early_stop_patience > 0
            and stale_eval_count + 1 >= args.early_stop_patience
        )
        log_line = (
            f"[Epoch {epoch:03d} | {time.time() - start_time:.1f}s] "
            f"label_mode={args.label_mode} train_mode={args.train_mode} "
            f"distance_mode={args.distance_mode} ctr_mode={args.ctr_mode} "
            f"train_loss={train_loss:.6f} "
            f"train_pred_loss={train_pred_loss:.6f} "
            f"train_distance_loss={train_distance_loss:.6f} "
            f"train_ctr_loss={train_ctr_loss:.6f} "
            f"test_loss={test_loss:.6f} "
            f"test_pred_loss={test_metrics['pred_loss']:.6f} "
            f"test_distance_loss={test_metrics['distance_loss']:.6f} "
            f"test_ctr_loss={test_metrics['ctr_loss']:.6f} "
            f"{tail_metrics}"
        )
        print(log_line, flush=True)
        with open(os.path.join(target_dir, 'log.log'), 'a') as f:
            f.write(log_line + '\n')

        if epoch % args.save_epoch == 0:
            torch.save(
                {'state_dict': model.state_dict(), 'args': args},
                os.path.join(target_dir, f'epoch_{epoch}.pt'),
            )

        should_plot = (
            args.plot_embedding
            and args.plot_every > 0
            and (epoch == 1 or epoch == args.epochs or epoch % args.plot_every == 0 or will_early_stop)
        )
        if should_plot:
            plot_path = os.path.join(target_dir, 'plots', f'epoch_{epoch:03d}_pca.png')
            plot_elapsed = save_embedding_pca_plot(
                model,
                train_data,
                train_plot_indices,
                test_data,
                test_plot_indices,
                device,
                args.batch_size,
                args.plot_fragment_sample_size,
                plot_path,
            )
            plot_line = f"[Plot] epoch={epoch:03d} path={plot_path} elapsed={plot_elapsed:.1f}s"
            print(plot_line, flush=True)
            with open(os.path.join(target_dir, 'log.log'), 'a') as f:
                f.write(plot_line + '\n')
            release_cuda_cache(device)

        if should_eval and test_loss < best_test_loss - args.min_delta:
            best_test_loss = test_loss
            stale_eval_count = 0
            torch.save(
                {'state_dict': model.state_dict(), 'args': args},
                os.path.join(target_dir, 'best.pt'),
            )
            if args.label_mode == 'reg':
                best_tail = (
                    f"rmse={test_metrics['rmse']:.6f} "
                    f"mae={test_metrics['mae']:.6f} "
                    f"pearson={test_metrics['pearson']:.6f} "
                    f"spearman={test_metrics['spearman']:.6f}"
                )
            else:
                best_tail = (
                    f"acc={test_metrics['accuracy']:.6f} "
                    f"f1={test_metrics['f1']:.6f} "
                    f"auroc={test_metrics['auroc']:.6f} "
                    f"auprc={test_metrics['auprc']:.6f}"
                )
            best_line = (
                f"[Best] epoch={epoch:03d} "
                f"test_loss={test_loss:.6f} "
                f"pred_loss={test_metrics['pred_loss']:.6f} "
                f"distance_loss={test_metrics['distance_loss']:.6f} "
                f"ctr_loss={test_metrics['ctr_loss']:.6f} "
                f"{best_tail}"
            )
            print(best_line, flush=True)
            with open(os.path.join(target_dir, 'log.log'), 'a') as f:
                f.write(best_line + '\n')
        elif should_eval:
            stale_eval_count += 1
            if args.early_stop_patience > 0 and stale_eval_count >= args.early_stop_patience:
                stop_line = (
                    f"[Early stop] epoch={epoch:03d} "
                    f"best_test_loss={best_test_loss:.6f} "
                    f"stale_evals={stale_eval_count}"
                )
                print(stop_line, flush=True)
                with open(os.path.join(target_dir, 'log.log'), 'a') as f:
                    f.write(stop_line + '\n')
                release_cuda_cache(device)
                break
        release_cuda_cache(device)


if __name__ == '__main__':
    main()
