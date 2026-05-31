import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Batch

from model.mfrag import MFRAG


DOCKING_TARGETS = ["parp1", "fa7", "5ht1b", "braf", "jak2"]
AUX_TARGETS = ["qed", "sa"]
TARGETS = DOCKING_TARGETS + AUX_TARGETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        nargs="+",
        default=TARGETS,
        choices=TARGETS,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--out_dir", type=str, default="results/mfrag_eval")
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def load_model(target: str, device: str):
    ckpt = f"./ckpt/only/reg/{target}/epoch_10.pt"
    model = MFRAG(device=device).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def load_test_frag():
    _, test_frag = torch.load("data/zinc250k_frag.pt")
    return test_frag


def safe_corr(fn, y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    try:
        value = fn(y_true, y_pred)[0]
    except Exception:
        value = np.nan
    return float(value)


def batched_predict(model, graphs: List, device: str, batch_size: int):
    reg_preds = []
    embeddings = []
    for start in range(0, len(graphs), batch_size):
        batch = Batch.from_data_list(graphs[start:start + batch_size]).to(device)
        with torch.no_grad():
            reg_pred, embedding = model(batch)
        reg_preds.append(reg_pred.cpu())
        embeddings.append(embedding.cpu())
    reg_preds = torch.cat(reg_preds, dim=0).numpy().reshape(-1)
    embeddings = torch.cat(embeddings, dim=0).numpy()
    return reg_preds, embeddings


def build_target_arrays(test_frag, target: str):
    graphs = []
    docking_reg = []
    smiles = []

    for graph, frag_list, value in test_frag:
        graphs.append(graph)
        raw_target = float(value[target])
        if target in DOCKING_TARGETS:
            target_value = float(np.clip(raw_target, 0, 20)) / 20.0
        else:
            target_value = raw_target
        docking_reg.append(target_value)
        smiles.append(value.get("smiles", None))

    return {
        "graphs": graphs,
        "docking_reg": np.asarray(docking_reg, dtype=np.float32),
        "smiles": smiles,
    }


def evaluate_target(model, test_frag, target: str, device: str, batch_size: int):
    payload = build_target_arrays(test_frag, target)
    reg_pred, embedding = batched_predict(model, payload["graphs"], device, batch_size)

    reg_true = payload["docking_reg"]
    row = {
        "target": target,
        "n_test": int(len(reg_true)),
        "reg_rmse": float(np.sqrt(mean_squared_error(reg_true, reg_pred))),
        "reg_mae": float(mean_absolute_error(reg_true, reg_pred)),
        "reg_r2": float(r2_score(reg_true, reg_pred)),
        "reg_pearson": safe_corr(pearsonr, reg_true, reg_pred),
        "reg_spearman": safe_corr(spearmanr, reg_true, reg_pred),
    }

    prediction_df = pd.DataFrame({
        "smiles": payload["smiles"],
        "docking_reg_true": reg_true,
        "docking_reg_pred": reg_pred,
    })
    return row, prediction_df


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    test_frag = load_test_frag()
    summary_rows = []

    for target in args.targets:
        print(f"[eval_mfrag] target={target}")
        model = load_model(target, args.device)
        row, prediction_df = evaluate_target(model, test_frag, target, args.device, args.batch_size)
        summary_rows.append(row)

        if args.save_predictions:
            pred_path = os.path.join(args.out_dir, f"{target}_predictions.csv")
            prediction_df.to_csv(pred_path, index=False)
            print(f"[eval_mfrag] saved predictions: {pred_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, "mfrag_eval_summary.csv")
    summary_json = os.path.join(args.out_dir, "mfrag_eval_summary.json")
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(summary_rows, f, indent=2)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df)
    print(f"[eval_mfrag] summary csv: {summary_csv}")
    print(f"[eval_mfrag] summary json: {summary_json}")


if __name__ == "__main__":
    main()
