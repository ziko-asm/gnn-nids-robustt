"""
Feature-space attacks.

1. gaussian_noise  - random measurement noise on every device (sanity baseline,
                     not an adversary).
2. pgd_evasion     - white-box L-infinity PGD (Madry et al., 2018) where ONLY the
                     compromised devices change ONLY their own traffic statistics,
                     trying to be classified as benign. This is the realistic
                     IoT threat: a bot can shape its own traffic, but cannot
                     rewrite a neighbour's telemetry.

`eps` is measured in standard deviations of each (standardised) feature.

Run:  python -m src.attack_feature --eps 0.5
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from src.dataset import generate_iot_graph_dataset
from src.utils import build_model, compute_metrics, fmt, set_seed, train_model


def gaussian_noise(x: torch.Tensor, sigma: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return x + sigma * torch.randn(x.shape, generator=g)


def pgd_evasion(model, x, edge_index, controlled: torch.Tensor, eps: float,
                steps: int = 40, alpha: float | None = None, edge_weight=None) -> torch.Tensor:
    """Perturb rows of `x` selected by boolean mask `controlled` so that those
    nodes are pushed toward class 0 (benign). Returns the perturbed features."""
    if eps <= 0 or controlled.sum() == 0:
        return x
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    model.eval()
    row_mask = controlled.float().unsqueeze(1)
    target = torch.zeros(int(controlled.sum()), dtype=torch.long)

    delta = torch.zeros_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        out = model(x + delta * row_mask, edge_index, edge_weight)
        # attacker minimises the loss of the "benign" label on its own nodes
        loss = F.nll_loss(out[controlled], target)
        (grad,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = (delta - alpha * grad.sign()).clamp(-eps, eps)
    return (x + delta.detach() * row_mask)


def main():
    parser = argparse.ArgumentParser(description="Feature-space attacks on GNN-NIDS")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    data = generate_iot_graph_dataset(num_nodes=args.nodes, seed=args.seed)
    attackers = data.y == 1

    for m in ["mlp", "gcn", "gat", "sage"]:
        set_seed(args.seed)
        model = train_model(build_model(m, data.num_features), data)
        x_noise = gaussian_noise(data.x, args.eps, seed=args.seed)
        x_pgd = pgd_evasion(model, data.x, data.edge_index, attackers, args.eps)
        print(f"\n[{m.upper()}]")
        print(f"  clean                 {fmt(compute_metrics(model, data.x, data.edge_index, data.y, data.test_mask))}")
        print(f"  gaussian sigma={args.eps:<4}   {fmt(compute_metrics(model, x_noise, data.edge_index, data.y, data.test_mask))}")
        print(f"  PGD evasion eps={args.eps:<4}  {fmt(compute_metrics(model, x_pgd, data.edge_index, data.y, data.test_mask))}")


if __name__ == "__main__":
    main()
