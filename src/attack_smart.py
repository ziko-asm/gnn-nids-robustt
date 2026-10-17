"""
Gradient-based structural attack (greedy adjacency-gradient, FGA / Metattack family).

Threat model (stated explicitly so results can be interpreted):
  * Grey-box: the attacker does NOT know the victim architecture or weights.
    It trains its own linearised 2-hop GCN surrogate (SGC, Wu et al. 2019)
    on the public training labels and transfers the perturbed graph.
  * Evasion: the victim is trained on the clean graph; the attack is applied
    at inference time.
  * The attacker knows which devices are its own bots (it controls them) and
    wants them to be classified as benign. It does not know other labels.
  * Budget: `rate` x (number of undirected links) edge flips (add or remove).

Algorithm (per step):
  1. A is the dense adjacency, relaxed to a continuous variable.
  2. L = CE( softmax( S(A)^2 X W )[bots], "compromised" ),  S(A) = D^-1/2 (A+I) D^-1/2
     (targeted: the attacker wants its own bots to be classified as benign)
  3. score_ij = (dL/dA_ij + dL/dA_ji) * (1 - 2 A_ij)
     -> positive score means flipping (i,j) increases the victim's loss.
  4. Flip the top-k admissible pairs, never touching the same pair twice.
Repeating in several small steps re-linearises the loss and is markedly
stronger than a single one-shot gradient.

Run:  python -m src.attack_smart --rate 0.05
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attack import budget, random_remove
from src.dataset import generate_iot_graph_dataset
from src.utils import (build_model, class_weights, compute_metrics, fmt,
                       set_seed, train_model)


def to_dense(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    A = torch.zeros(n, n)
    A[edge_index[0], edge_index[1]] = 1.0
    A = ((A + A.t()) > 0).float()
    A.fill_diagonal_(0.0)
    return A


def to_edge_index(A: torch.Tensor) -> torch.Tensor:
    upper = torch.triu(A, diagonal=1).nonzero().t()
    return torch.cat([upper, upper.flip(0)], dim=1)


def normalize(A: torch.Tensor) -> torch.Tensor:
    A_hat = A + torch.eye(A.size(0))
    d_inv_sqrt = A_hat.sum(1).pow(-0.5)
    return d_inv_sqrt[:, None] * A_hat * d_inv_sqrt[None, :]


def train_surrogate(data, epochs: int = 200, lr: float = 0.05, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    A = to_dense(data.edge_index, data.num_nodes)
    S = normalize(A)
    SSX = S @ (S @ data.x)
    W = nn.Linear(data.num_features, 2)
    opt = torch.optim.Adam(W.parameters(), lr=lr, weight_decay=5e-4)
    w = class_weights(data.y, data.train_mask)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(W(SSX)[data.train_mask], data.y[data.train_mask], weight=w)
        loss.backward()
        opt.step()
    W.requires_grad_(False)
    return W


def gradient_attack(data, rate: float, steps: int = 20, surrogate: nn.Linear | None = None,
                    seed: int = 0) -> torch.Tensor:
    """Returns the perturbed edge_index."""
    if rate <= 0:
        return data.edge_index
    n = data.num_nodes
    W = surrogate if surrogate is not None else train_surrogate(data, seed=seed)
    A = to_dense(data.edge_index, n)

    # Threat model: the botmaster knows which devices it controls (its bots)
    # and wants them classified as benign. It does NOT know the victim model.
    bots = data.y == 1
    target = torch.ones(int(bots.sum()), dtype=torch.long)

    B = budget(data.edge_index, rate)
    per_step = max(1, math.ceil(B / steps))
    admissible = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)

    remaining = B
    while remaining > 0:
        A_var = A.clone().requires_grad_(True)
        S = normalize(A_var)
        loss = F.cross_entropy(W(S @ (S @ data.x))[bots], target)
        (grad,) = torch.autograd.grad(loss, A_var)
        with torch.no_grad():
            grad = grad + grad.t()
            score = grad * (1.0 - 2.0 * A)
            score = score.masked_fill(~admissible, float("-inf"))
            k = min(per_step, remaining, int(admissible.sum()))
            if k == 0:
                break
            idx = torch.topk(score.flatten(), k).indices
            r, c = idx // n, idx % n
            new_val = 1.0 - A[r, c]
            A[r, c] = new_val
            A[c, r] = new_val
            admissible[r, c] = False
        remaining -= k

    return to_edge_index(A)


def main():
    parser = argparse.ArgumentParser(description="Gradient-based structural attack")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    data = generate_iot_graph_dataset(num_nodes=args.nodes, seed=args.seed)
    B = budget(data.edge_index, args.rate)
    print(f"[*] Budget: {B} edge flips ({args.rate:.0%} of links). Training surrogate...")
    ei_grad = gradient_attack(data, args.rate, seed=args.seed)
    ei_rand = random_remove(data.edge_index, data.num_nodes, args.rate, seed=args.seed)

    for m in ["gcn", "gat", "sage"]:
        set_seed(args.seed)
        model = train_model(build_model(m, data.num_features), data)
        print(f"\n[{m.upper()}]")
        print(f"  clean            {fmt(compute_metrics(model, data.x, data.edge_index, data.y, data.test_mask))}")
        print(f"  random removal   {fmt(compute_metrics(model, data.x, ei_rand, data.y, data.test_mask))}")
        print(f"  gradient attack  {fmt(compute_metrics(model, data.x, ei_grad, data.y, data.test_mask))}")


if __name__ == "__main__":
    main()
