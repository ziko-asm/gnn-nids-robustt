"""
Explain a detection with GNNExplainer (Ying et al., NeurIPS 2019) and measure
how faithful the explanation is.

    python -m src.explain                 # explains the first correctly detected bot
    python -m src.explain --node 42 --model gcn

Reported
  * top features by importance (named, e.g. syn_ratio, unique_dst_ports)
  * top links by importance (with the label of the neighbour)
  * Fidelity+ : prediction change when the important part is REMOVED (higher = better)
  * Fidelity- : prediction change when ONLY the important part is kept (lower = better)
    (Amara et al., 2022 / PyG `fidelity`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.metric import fidelity

from src.dataset import generate_iot_graph_dataset
from src.utils import build_model, set_seed, train_model


def build_explainer(model, epochs: int = 200) -> Explainer:
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node",
                          return_type="log_probs"),
    )


def explain_node(model, data, node_idx: int, top_k: int = 5, epochs: int = 200) -> dict:
    model.eval()
    explainer = build_explainer(model, epochs)
    exp = explainer(data.x, data.edge_index, index=node_idx)

    feat_imp = exp.node_mask.sum(dim=0)
    feat_imp = feat_imp / (feat_imp.sum() + 1e-12)
    order = torch.argsort(feat_imp, descending=True)[:top_k]
    names = data.feature_names

    em = exp.edge_mask
    touching = (data.edge_index[0] == node_idx) | (data.edge_index[1] == node_idx)
    em_local = torch.where(touching, em, torch.zeros_like(em))
    top_e = torch.argsort(em_local, descending=True)[:top_k]

    try:
        fid_pos, fid_neg = fidelity(explainer, exp)
        fid_pos, fid_neg = float(fid_pos), float(fid_neg)
    except Exception:  # older PyG versions
        fid_pos, fid_neg = float("nan"), float("nan")

    with torch.no_grad():
        prob = model(data.x, data.edge_index).exp()[node_idx, 1].item()

    return {
        "node": int(node_idx),
        "true_label": int(data.y[node_idx]),
        "p_compromised": round(prob, 4),
        "top_features": [{"feature": names[i], "importance": round(float(feat_imp[i]), 4)}
                         for i in order.tolist()],
        "top_links": [{"src": int(data.edge_index[0, e]), "dst": int(data.edge_index[1, e]),
                       "importance": round(float(em[e]), 4),
                       "neighbour_label": int(data.y[data.edge_index[1, e]
                                                     if data.edge_index[0, e] == node_idx
                                                     else data.edge_index[0, e]])}
                      for e in top_e.tolist() if em_local[e] > 0],
        "fidelity_plus": round(fid_pos, 4),
        "fidelity_minus": round(fid_neg, 4),
    }


def main():
    p = argparse.ArgumentParser(description="GNNExplainer for GNN-NIDS")
    p.add_argument("--model", default="sage", choices=["gcn", "gat", "sage"])
    p.add_argument("--node", type=int, default=None)
    p.add_argument("--nodes", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    data = generate_iot_graph_dataset(num_nodes=args.nodes, seed=args.seed)
    model = train_model(build_model(args.model, data.num_features), data)

    node = args.node
    if node is None:
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(-1)
        hits = torch.where(data.test_mask & (data.y == 1) & (pred == 1))[0]
        node = int(hits[0]) if len(hits) else int(torch.where(data.test_mask)[0][0])

    print(f"[*] Explaining node {node} with GNNExplainer ({args.model.upper()})...")
    res = explain_node(model, data, node)
    print(f"    true label: {'compromised' if res['true_label'] else 'benign'} | "
          f"P(compromised) = {res['p_compromised']:.3f}")
    print("    top features:")
    for f in res["top_features"]:
        print(f"      {f['feature']:18s} {f['importance']:.3f}")
    print("    top links:")
    for e in res["top_links"]:
        print(f"      {e['src']:4d} - {e['dst']:4d}  importance {e['importance']:.3f}  "
              f"neighbour={'bot' if e['neighbour_label'] else 'benign'}")
    print(f"    Fidelity+ {res['fidelity_plus']:.3f} (higher is better) | "
          f"Fidelity- {res['fidelity_minus']:.3f} (lower is better)")

    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / f"explanation_node_{node}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"[*] Saved -> results/explanation_node_{node}.json")


if __name__ == "__main__":
    main()
