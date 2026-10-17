"""
Defences against structural and feature attacks.

  none         - plain training on the clean graph (reference).
  cosine_prune - GCN-Jaccard style pre-processing (Wu et al., IJCAI 2019),
                 adapted to continuous features: drop links whose endpoint
                 feature vectors have cosine similarity below a threshold.
                 Adversarial links tend to connect dissimilar devices. The
                 threshold is chosen on the *validation* split, never on test.
  dropedge     - DropEdge augmentation (Rong et al., ICLR 2020): random link
                 removal at every training step. This is a robustness
                 regulariser, NOT adversarial training, and is named as such.
  adv_train    - PGD adversarial training (Madry et al., 2018) in feature space:
                 after a short warm-up, compromised training devices are
                 perturbed with PGD (eps=0.5, the evaluation budget) at every
                 step. The checkpoint is selected on ROBUST validation F1
                 (mean of clean and PGD-attacked), not on clean F1 alone.

Run:  python -m src.defend --model sage
"""

from __future__ import annotations

import argparse
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge

from src.attack_feature import pgd_evasion
from src.attack_smart import gradient_attack
from src.dataset import generate_iot_graph_dataset
from src.utils import build_model, compute_metrics, fmt, set_seed, train_model

DEFENSES = ("none", "cosine_prune", "dropedge", "adv_train")
Preprocess = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def cosine_prune(x: torch.Tensor, edge_index: torch.Tensor, threshold: float) -> torch.Tensor:
    xn = F.normalize(x, dim=1)
    sim = (xn[edge_index[0]] * xn[edge_index[1]]).sum(1)
    return edge_index[:, sim >= threshold]


def _identity(x, ei):
    return x, ei


def train_defended(defense: str, model_type: str, data, seed: int = 0,
                   dropedge_p: float = 0.2, adv_eps: float = 0.5,
                   prune_grid=(-0.3, -0.2, -0.1, 0.0, 0.1)):
    """Returns (trained_model, preprocess) where `preprocess(x, edge_index)`
    must be applied to (possibly attacked) inputs before inference."""
    if defense not in DEFENSES:
        raise ValueError(f"Unknown defence '{defense}'. Choose one of {DEFENSES}")

    if defense == "none":
        set_seed(seed)
        return train_model(build_model(model_type, data.num_features), data), _identity

    if defense == "cosine_prune":
        best = None
        for t in prune_grid:
            set_seed(seed)
            ei = cosine_prune(data.x, data.edge_index, t)
            model = train_model(build_model(model_type, data.num_features), data, edge_index=ei)
            val_f1 = compute_metrics(model, data.x, ei, data.y, data.val_mask)["f1"]
            if best is None or val_f1 > best[0]:
                best = (val_f1, t, model)
        _, t, model = best
        model.prune_threshold = t
        return model, (lambda x, ei, t=t: (x, cosine_prune(x, ei, t)))

    if defense == "dropedge":
        def aug(epoch, model, x, ei):
            ei_d, _ = dropout_edge(ei, p=dropedge_p, force_undirected=True)
            return x, ei_d
        set_seed(seed)
        return train_model(build_model(model_type, data.num_features), data, augment=aug), _identity

    # adv_train
    train_bots = data.train_mask & (data.y == 1)
    val_bots = data.val_mask & (data.y == 1)

    def aug(epoch, model, x, ei):
        if epoch <= 10:                      # short clean warm-up
            return x, ei
        x_adv = pgd_evasion(model, x, ei, train_bots, eps=adv_eps, steps=10)
        model.train()
        return x_adv.detach(), ei

    def robust_val(model):
        # checkpoint = best average of clean and PGD-attacked validation F1
        clean = compute_metrics(model, data.x, data.edge_index, data.y, data.val_mask)["f1"]
        x_adv = pgd_evasion(model, data.x, data.edge_index, val_bots, eps=adv_eps, steps=20)
        robust = compute_metrics(model, x_adv, data.edge_index, data.y, data.val_mask)["f1"]
        return 0.5 * clean + 0.5 * robust

    set_seed(seed)
    model = build_model(model_type, data.num_features)
    return train_model(model, data, augment=aug, select_fn=robust_val, patience=60), _identity


def main():
    parser = argparse.ArgumentParser(description="Compare defences under attack")
    parser.add_argument("--model", default="gcn", choices=["gcn", "gat", "sage"])
    parser.add_argument("--rate", type=float, default=0.05, help="structural budget")
    parser.add_argument("--eps", type=float, default=0.5, help="feature PGD budget")
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    data = generate_iot_graph_dataset(num_nodes=args.nodes, seed=args.seed)
    ei_att = gradient_attack(data, args.rate, seed=args.seed)

    print(f"Model {args.model.upper()} | gradient attack {args.rate:.0%} | PGD eps {args.eps}\n")
    for d in DEFENSES:
        model, prep = train_defended(d, args.model, data, seed=args.seed)
        x0, ei0 = prep(data.x, data.edge_index)
        x1, ei1 = prep(data.x, ei_att)
        x_pgd = pgd_evasion(model, data.x, ei0, data.y == 1, args.eps)
        x2, ei2 = prep(x_pgd, data.edge_index)
        print(f"[{d}]")
        print(f"  clean       {fmt(compute_metrics(model, x0, ei0, data.y, data.test_mask))}")
        print(f"  structure   {fmt(compute_metrics(model, x1, ei1, data.y, data.test_mask))}")
        print(f"  feature PGD {fmt(compute_metrics(model, x2, ei2, data.y, data.test_mask))}\n")


if __name__ == "__main__":
    main()
