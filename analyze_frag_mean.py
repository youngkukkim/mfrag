import argparse
import json
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Batch

from model.mfrag import MFRAG
from utils_mfrag.data import get_frag_batch_brics, get_graph_from_frag


PLOT_CMAP = "coolwarm"


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def get_fragment_smiles(smiles: str) -> Tuple[str, List[str], str]:
    canonical = canonicalize_smiles(smiles)
    mol = Chem.MolFromSmiles(canonical)
    frag_smiles = get_frag_batch_brics(mol, get_frag_only=True)

    if frag_smiles is None:
        return canonical, [], "no_brics_cut"

    frag_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(frag), isomericSmiles=False) for frag in frag_smiles]
    return canonical, frag_smiles, "brics"


def load_mfrag(target: str, device: str):
    ckpt = f"./ckpt/only/reg/{target}/epoch_10.pt"
    model = MFRAG(device=device).to(device)
    state_dict = torch.load(ckpt, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def embed_graphs(graphs, model, device: str, batch_size: int) -> np.ndarray:
    embeddings = []
    for start in range(0, len(graphs), batch_size):
        batch = Batch.from_data_list(graphs[start:start + batch_size]).to(device)
        with torch.no_grad():
            _, embedding = model(batch)
        embeddings.append(embedding.cpu())
    if not embeddings:
        return np.zeros((0, 128), dtype=np.float32)
    return torch.cat(embeddings, dim=0).numpy()


def embed_smiles_list(smiles_list: List[str], model, device: str, batch_size: int) -> np.ndarray:
    graphs = [get_graph_from_frag(smiles) for smiles in smiles_list]
    return embed_graphs(graphs, model, device, batch_size)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def get_cumulative_mean_stats(frag_embeddings: np.ndarray, mol_embedding: np.ndarray):
    cumulative_mean_embeddings = []
    cumulative_l2 = []
    cumulative_cosine = []

    for idx in range(len(frag_embeddings)):
        mean_embedding = frag_embeddings[:idx + 1].mean(axis=0)
        cumulative_mean_embeddings.append(mean_embedding)
        cumulative_l2.append(float(np.linalg.norm(mol_embedding - mean_embedding)))
        cumulative_cosine.append(cosine_similarity(mol_embedding, mean_embedding))

    return (
        np.stack(cumulative_mean_embeddings, axis=0),
        np.array(cumulative_l2, dtype=np.float32),
        np.array(cumulative_cosine, dtype=np.float32),
    )


def sample_train_graphs_for_region(target, sample_size, high_score_quantile, random_seed):
    train_df = pd.read_csv("data/zinc250k.csv")
    with open("data/valid_idx_zinc250k.json") as f:
        test_idx = set(json.load(f))
    train_idx = [i for i in range(len(train_df)) if i not in test_idx]
    train_df = train_df.iloc[train_idx].reset_index(drop=True)
    train_df = train_df[np.abs(train_df[target].to_numpy(dtype=np.float32)) > 1e-8].reset_index(drop=True)

    train_scores = train_df[target].to_numpy(dtype=np.float32)
    high_thr = np.quantile(train_scores, high_score_quantile)
    high_idx = np.where(train_scores >= high_thr)[0]
    low_idx = np.where(train_scores < high_thr)[0]
    rng = np.random.default_rng(random_seed)

    if sample_size is None or sample_size <= 0 or sample_size >= len(train_df):
        selected_idx = np.arange(len(train_df))
    else:
        high_take = min(len(high_idx), max(sample_size // 4, 1))
        low_take = min(len(low_idx), max(sample_size - high_take, 0))
        selected_high = rng.choice(high_idx, size=high_take, replace=False) if high_take > 0 else np.array([], dtype=int)
        selected_low = rng.choice(low_idx, size=low_take, replace=False) if low_take > 0 else np.array([], dtype=int)
        selected_idx = np.concatenate([selected_low, selected_high])

    selected_smiles = train_df.iloc[selected_idx]["smiles"].tolist()
    selected_graphs = [get_graph_from_frag(smiles) for smiles in selected_smiles]
    selected_scores = train_scores[selected_idx]
    return selected_graphs, selected_scores, high_thr


def weighted_average_from_neighbors(values, distances, eps=1e-6):
    weights = 1.0 / (distances + eps)
    return (weights * values).sum(axis=1) / weights.sum(axis=1)


def compute_train_local_scores(train_embeddings, train_scores, k):
    if len(train_embeddings) == 0:
        return np.zeros((0,), dtype=np.float32)
    n_neighbors = min(max(k + 1, 2), len(train_embeddings))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(train_embeddings)
    distances, indices = nn.kneighbors(train_embeddings)
    if distances.shape[1] > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    neighbor_scores = train_scores[indices]
    return weighted_average_from_neighbors(neighbor_scores, distances).astype(np.float32)


def compute_good_region_grid(train_2d, train_local_scores, xlim, ylim, grid_size, grid_k, density_quantile):
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    n_neighbors = min(max(grid_k, 1), len(train_2d))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(train_2d)
    distances, indices = nn.kneighbors(grid_points)

    weights = 1.0 / (distances + 1e-6)
    smoothed_scores = (weights * train_local_scores[indices]).sum(axis=1) / weights.sum(axis=1)
    density = weights.sum(axis=1)
    density_threshold = np.quantile(density, density_quantile)
    smoothed_scores[density < density_threshold] = np.nan
    return xx, yy, smoothed_scores.reshape(xx.shape)


def get_axis_limits(points_2d: np.ndarray):
    x_min, x_max = float(points_2d[:, 0].min()), float(points_2d[:, 0].max())
    y_min, y_max = float(points_2d[:, 1].min()), float(points_2d[:, 1].max())
    x_pad = max((x_max - x_min) * 0.05, 1e-3)
    y_pad = max((y_max - y_min) * 0.05, 1e-3)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def build_plot_context(model, args):
    if args.plot_method != "pca":
        return None

    train_graphs, train_scores, high_thr = sample_train_graphs_for_region(
        args.target,
        args.train_sample_size,
        args.high_score_quantile,
        args.random_seed,
    )
    train_embeddings = embed_graphs(train_graphs, model, args.device, args.batch_size)
    reducer = PCA(n_components=2, random_state=args.random_seed)
    train_2d = reducer.fit_transform(train_embeddings)
    train_local_scores = compute_train_local_scores(train_embeddings, train_scores, args.good_region_k)
    xlim, ylim = get_axis_limits(train_2d)
    xx, yy, good_region = compute_good_region_grid(
        train_2d,
        train_local_scores,
        xlim,
        ylim,
        args.good_region_grid_size,
        args.good_region_grid_k,
        args.good_region_density_quantile,
    )
    return {
        "train_2d": train_2d,
        "train_scores": train_scores,
        "train_local_scores": train_local_scores,
        "high_thr": high_thr,
        "reducer": reducer,
        "xlim": xlim,
        "ylim": ylim,
        "xx": xx,
        "yy": yy,
        "good_region": good_region,
        "region_vmin": float(np.nanmin(good_region)),
        "region_vmax": float(np.nanmax(good_region)),
    }


def summarize_single(smiles: str, model, args):
    mol_smiles, frag_smiles, frag_mode = get_fragment_smiles(smiles)
    if len(frag_smiles) == 0:
        return {
            "input_smiles": smiles,
            "mol_smiles": mol_smiles,
            "frag_mode": frag_mode,
            "num_frags": 0,
            "frag_smiles": "",
            "l2_to_frag_mean": np.nan,
            "cosine_to_frag_mean": np.nan,
            "mean_frag_norm": np.nan,
            "mol_norm": np.nan,
        }, None

    all_smiles = [mol_smiles] + frag_smiles
    embeddings = embed_smiles_list(all_smiles, model, args.device, args.batch_size)
    mol_embedding = embeddings[0]
    frag_embeddings = embeddings[1:]
    frag_mean_embedding = frag_embeddings.mean(axis=0)
    cumulative_mean_embeddings, cumulative_l2, cumulative_cosine = get_cumulative_mean_stats(
        frag_embeddings,
        mol_embedding,
    )

    row = {
        "input_smiles": smiles,
        "mol_smiles": mol_smiles,
        "frag_mode": frag_mode,
        "num_frags": len(frag_smiles),
        "frag_smiles": "|".join(frag_smiles),
        "l2_to_frag_mean": float(np.linalg.norm(mol_embedding - frag_mean_embedding)),
        "cosine_to_frag_mean": cosine_similarity(mol_embedding, frag_mean_embedding),
        "mean_frag_norm": float(np.linalg.norm(frag_mean_embedding)),
        "mol_norm": float(np.linalg.norm(mol_embedding)),
        "cumulative_l2_path": "|".join(f"{value:.6f}" for value in cumulative_l2.tolist()),
        "cumulative_cosine_path": "|".join(f"{value:.6f}" for value in cumulative_cosine.tolist()),
    }

    payload = {
        "mol_smiles": mol_smiles,
        "frag_smiles": frag_smiles,
        "mol_embedding": mol_embedding,
        "frag_embeddings": frag_embeddings,
        "frag_mean_embedding": frag_mean_embedding,
        "cumulative_mean_embeddings": cumulative_mean_embeddings,
        "cumulative_l2": cumulative_l2,
        "cumulative_cosine": cumulative_cosine,
    }
    return row, payload


def reduce_points(points: np.ndarray, method: str, random_seed: int) -> np.ndarray:
    if len(points) <= 2:
        padded = np.zeros((len(points), 2), dtype=np.float32)
        if len(points) == 2:
            padded[1, 0] = 1.0
        return padded
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_seed, init="random", learning_rate="auto")
    else:
        reducer = PCA(n_components=2, random_state=random_seed)
    return reducer.fit_transform(points)


def make_safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:120]


def get_output_stem(args) -> str:
    if args.out_prefix is not None:
        stem = args.out_prefix.rstrip("_")
        out_dir = os.path.dirname(stem)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return stem

    if args.file is not None:
        base = os.path.splitext(os.path.basename(args.file))[0]
    else:
        base = make_safe_name(args.smiles)

    out_dir = os.path.join("results", "frag_mean", base)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)


def plot_single(payload, row, args, out_path: str, plot_context=None):
    mol_embedding = payload["mol_embedding"]
    frag_embeddings = payload["frag_embeddings"]
    frag_mean_embedding = payload["frag_mean_embedding"]
    cumulative_mean_embeddings = payload["cumulative_mean_embeddings"]
    cumulative_l2 = payload["cumulative_l2"]
    cumulative_cosine = payload["cumulative_cosine"]

    points = np.vstack([
        mol_embedding[None, :],
        frag_embeddings,
        cumulative_mean_embeddings,
        frag_mean_embedding[None, :],
    ])
    if plot_context is not None:
        points_2d = plot_context["reducer"].transform(points)
    else:
        points_2d = reduce_points(points, args.plot_method, args.random_seed)

    mol_2d = points_2d[0]
    frag_2d = points_2d[1:1 + len(frag_embeddings)]
    cumulative_mean_2d = points_2d[1 + len(frag_embeddings):1 + len(frag_embeddings) + len(cumulative_mean_embeddings)]
    frag_mean_2d = points_2d[-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    frag_color = "steelblue"
    mean_color = "darkorange"

    if plot_context is not None:
        contour = ax.contourf(
            plot_context["xx"],
            plot_context["yy"],
            plot_context["good_region"],
            levels=18,
            cmap=PLOT_CMAP,
            vmin=plot_context["region_vmin"],
            vmax=plot_context["region_vmax"],
            alpha=0.75,
        )
        ax.scatter(
            plot_context["train_2d"][:, 0],
            plot_context["train_2d"][:, 1],
            s=8,
            c="black",
            alpha=0.08,
            edgecolors="none",
            zorder=1,
        )
        ax.set_xlim(*plot_context["xlim"])
        ax.set_ylim(*plot_context["ylim"])
    else:
        contour = None

    if len(frag_embeddings) > 0:
        ax.scatter(
            frag_2d[:, 0],
            frag_2d[:, 1],
            c=frag_color,
            s=48,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3,
            label="Fragments",
        )
        for idx, (x, y) in enumerate(frag_2d):
            ax.text(x, y, f"f{idx + 1}", fontsize=8, ha="left", va="bottom", color=frag_color)

    if len(cumulative_mean_embeddings) > 0:
        ax.scatter(
            cumulative_mean_2d[:, 0],
            cumulative_mean_2d[:, 1],
            c=mean_color,
            s=60,
            marker="D",
            edgecolors="black",
            linewidths=0.3,
            zorder=4,
            label="Cumulative means",
        )
        for idx, (x, y) in enumerate(cumulative_mean_2d, start=1):
            ax.text(x, y, f"m{idx}", fontsize=8, ha="right", va="top", color=mean_color)

        for idx in range(len(cumulative_mean_2d)):
            ax.annotate(
                "",
                xy=(cumulative_mean_2d[idx, 0], cumulative_mean_2d[idx, 1]),
                xytext=(frag_2d[idx, 0], frag_2d[idx, 1]),
                arrowprops=dict(
                    arrowstyle="->",
                    linestyle="--",
                    color="gray",
                    linewidth=1.0,
                    alpha=0.9,
                ),
                zorder=3,
            )

        for idx in range(len(cumulative_mean_2d) - 1):
            ax.annotate(
                "",
                xy=(cumulative_mean_2d[idx + 1, 0], cumulative_mean_2d[idx + 1, 1]),
                xytext=(cumulative_mean_2d[idx, 0], cumulative_mean_2d[idx, 1]),
                arrowprops=dict(
                    arrowstyle="->",
                    linestyle="--",
                    color=mean_color,
                    linewidth=1.2,
                    alpha=0.95,
                ),
                zorder=3,
            )

    ax.scatter(
        mol_2d[0],
        mol_2d[1],
        s=180,
        c="black",
        marker="*",
        label="Whole molecule",
        zorder=5,
    )
    ax.scatter(
        frag_mean_2d[0],
        frag_mean_2d[1],
        s=120,
        c=mean_color,
        marker="X",
        edgecolors="black",
        linewidths=0.5,
        label="Mean(fragment embeddings)",
        zorder=6,
    )
    ax.set_title(
        f"Mol vs Fragment / Cumulative Means\n"
        f"frags={row['num_frags']} | l2={row['l2_to_frag_mean']:.4f} | cos={row['cosine_to_frag_mean']:.4f}"
    )
    ax.set_xlabel(f"{args.plot_method.upper()} 1")
    ax.set_ylabel(f"{args.plot_method.upper()} 2")
    ax.legend(loc="best")
    if contour is not None:
        cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"{args.target} local train-region score")

    ax_trend = axes[1]
    x = np.arange(1, len(cumulative_l2) + 1, dtype=np.int32)
    ax_trend.plot(x, cumulative_l2, color="black", marker="o", linewidth=1.8, label="L2 to molecule")
    ax_trend.set_xlabel("Number of joined fragments")
    ax_trend.set_ylabel("L2 distance", color="black")
    ax_trend.tick_params(axis="y", labelcolor="black")

    ax_cos = ax_trend.twinx()
    ax_cos.plot(x, cumulative_cosine, color="crimson", marker="s", linewidth=1.6, label="Cosine to molecule")
    ax_cos.set_ylabel("Cosine similarity", color="crimson")
    ax_cos.tick_params(axis="y", labelcolor="crimson")

    ax_trend.set_title("Cumulative mean change")
    handles_left, labels_left = ax_trend.get_legend_handles_labels()
    handles_right, labels_right = ax_cos.get_legend_handles_labels()
    ax_trend.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_input_smiles(args) -> List[str]:
    if args.smiles is not None:
        return [args.smiles]

    smiles_col_idx = None
    try:
        smiles_col_idx = int(args.smiles_col)
    except ValueError:
        smiles_col_idx = None

    df = pd.read_csv(args.file)
    smiles_series = None

    if smiles_col_idx is not None:
        if smiles_col_idx >= df.shape[1]:
            raise ValueError(f"Column index '{smiles_col_idx}' out of range for {args.file}")
        smiles_series = df.iloc[:, smiles_col_idx]
    elif args.smiles_col in df.columns:
        smiles_series = df[args.smiles_col]
    else:
        df_no_header = pd.read_csv(args.file, header=None)
        if smiles_col_idx is not None:
            if smiles_col_idx >= df_no_header.shape[1]:
                raise ValueError(f"Column index '{smiles_col_idx}' out of range for {args.file}")
            smiles_series = df_no_header.iloc[:, smiles_col_idx]
        elif args.smiles_col.upper() == "SMILES":
            smiles_series = df_no_header.iloc[:, 0]
        else:
            raise ValueError(
                f"Column '{args.smiles_col}' not found in {args.file}. "
                f"If this is a headerless result CSV, use '--smiles_col 0'."
            )

    smiles_list = smiles_series.dropna().astype(str).tolist()
    if args.max_molecules is not None:
        smiles_list = smiles_list[:args.max_molecules]
    return smiles_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", type=str, required=True,
                        choices=["parp1", "fa7", "5ht1b", "braf", "jak2"])
    parser.add_argument("--smiles", type=str, default=None)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--smiles_col", type=str, default="SMILES")
    parser.add_argument("--max_molecules", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--out_prefix", type=str, default=None)
    parser.add_argument("--train_sample_size", type=int, default=3000)
    parser.add_argument("--high_score_quantile", type=float, default=0.95)
    parser.add_argument("--good_region_k", type=int, default=40)
    parser.add_argument("--good_region_grid_k", type=int, default=30)
    parser.add_argument("--good_region_grid_size", type=int, default=180)
    parser.add_argument("--good_region_density_quantile", type=float, default=0.20)
    args = parser.parse_args()

    if (args.smiles is None) == (args.file is None):
        raise ValueError("Provide exactly one of --smiles or --file")

    smiles_list = load_input_smiles(args)
    model = load_mfrag(args.target, args.device)
    out_stem = get_output_stem(args)
    plot_context = build_plot_context(model, args) if args.plot else None

    rows = []
    for idx, smiles in enumerate(smiles_list):
        try:
            row, payload = summarize_single(smiles, model, args)
        except Exception as exc:
            row = {
                "input_smiles": smiles,
                "mol_smiles": "",
                "frag_mode": "error",
                "num_frags": 0,
                "frag_smiles": "",
                "l2_to_frag_mean": np.nan,
                "cosine_to_frag_mean": np.nan,
                "mean_frag_norm": np.nan,
                "mol_norm": np.nan,
                "error": str(exc),
            }
            payload = None
        row["row_id"] = idx
        rows.append(row)

        if args.plot and payload is not None:
            if args.file is None:
                plot_path = f"{out_stem}_plot.png"
            else:
                plot_path = f"{out_stem}_{idx:04d}_plot.png"
            plot_single(payload, row, args, plot_path, plot_context=plot_context)

    df_out = pd.DataFrame(rows)
    csv_path = f"{out_stem}_summary.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved summary:\t{csv_path}")
    print(df_out[[
        "row_id",
        "mol_smiles",
        "frag_mode",
        "num_frags",
        "l2_to_frag_mean",
        "cosine_to_frag_mean",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
