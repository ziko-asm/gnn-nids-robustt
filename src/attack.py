"""
Non-gradient structural attacks (test-time / evasion setting).

Budget convention used across the whole repo:
    rate = fraction of the clean graph's *undirected* edges that the attacker
           may flip (remove or add). rate=0.10 on a graph with 1,000 links
           means 100 flipped links.

Attacks
    random_remove : delete random links (models link loss / jamming).
    random_add    : insert random links (models spoofed / injected flows).
    dice          : "Delete Internally, Connect Externally" (Waniek et al., 2018).
                    Uses label knowledge: removes links between same-class
                    devices and adds links between benign and compromised ones.
                    A strong, cheap heuristic baseline.

Run:  python -m src.attack --rate 0.1
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src.dataset import generate_iot_graph_dataset
from src.utils import build_model, compute_metrics, fmt, set_seed, train_model


# ---------------------------------------------------------------------------
# edge helpers
# ---------------------------------------------------------------------------
def undirected_pairs(edge_index: torch.Tensor) -> np.ndarray:
    """[2, E] (both directions) -> unique (u < v) pairs as [M, 2] numpy array."""
    ei = edge_index.cpu().numpy()
    u, v = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    keep = u != v
    pairs = np.unique(np.stack([u[keep], v[keep]], 1), axis=0)
    return pairs


def pairs_to_edge_index(pairs: np.ndarray) -> torch.Tensor:
    if len(pairs) == 0:
        return torch.empty(2, 0, dtype=torch.long)
    p = torch.as_tensor(pairs, dtype=torch.long).t()
    return torch.cat([p, p.flip(0)], dim=1)


def budget(edge_index: torch.Tensor, rate: float) -> int:
    return int(round(rate * len(undirected_pairs(edge_index))))


def _sample_non_edges(n: int, k: int, existing: set, rng: np.random.Generator,
                      allowed=None) -> list:
    out, tries = [], 0
    while len(out) < k and tries < 50 * max(k, 1):
        tries += 1
        if allowed is None:
            u, v = rng.integers(0, n, 2)
        else:
            u, v = allowed(rng)
        if u == v:
            continue
        a, b = int(min(u, v)), int(max(u, v))
        if (a, b) in existing:
            continue
        existing.add((a, b))
        out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# attacks
# ---------------------------------------------------------------------------
def random_remove(edge_index, num_nodes, rate, seed=0, **_):
    rng = np.random.default_rng(seed)
    pairs = undirected_pairs(edge_index)
    k = int(round(rate * len(pairs)))
    keep = rng.permutation(len(pairs))[k:]
    return pairs_to_edge_index(pairs[np.sort(keep)])


def random_add(edge_index, num_nodes, rate, seed=0, **_):
    rng = np.random.default_rng(seed)
    pairs = undirected_pairs(edge_index)
    k = int(round(rate * len(pairs)))
    existing = set(map(tuple, pairs.tolist()))
    new = _sample_non_edges(num_nodes, k, existing, rng)
    return pairs_to_edge_index(np.concatenate([pairs, np.array(new, dtype=np.int64).reshape(-1, 2)]))


def dice(edge_index, num_nodes, rate, y=None, seed=0, **_):
    if y is None:
        raise ValueError("DICE needs labels")
    rng = np.random.default_rng(seed)
    yn = y.cpu().numpy()
    pairs = undirected_pairs(edge_index)
    k = int(round(rate * len(pairs)))
    k_del, k_add = k // 2, k - k // 2

    same = np.where(yn[pairs[:, 0]] == yn[pairs[:, 1]])[0]
    drop = rng.choice(same, size=min(k_del, len(same)), replace=False)
    kept = np.delete(pairs, drop, axis=0)

    benign, bad = np.where(yn == 0)[0], np.where(yn == 1)[0]
    existing = set(map(tuple, pairs.tolist()))
    new = _sample_non_edges(num_nodes, k_add, existing, rng,
                            allowed=lambda r: (r.choice(benign), r.choice(bad)))
    return pairs_to_edge_index(np.concatenate([kept, np.array(new, dtype=np.int64).reshape(-1, 2)]))


STRUCTURE_ATTACKS = {"random_remove": random_remove, "random_add": random_add, "dice": dice}


def apply_structure_attack(name, data, rate, seed=0):
    if rate <= 0:
        return data.edge_index
    return STRUCTURE_ATTACKS[name](data.edge_index, data.num_nodes, rate, y=data.y, seed=seed)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Heuristic structural attacks on GNN-NIDS")
    parser.add_argument("--rate", type=float, default=0.10, help="fraction of edges flipped")
    parser.add_argument("--nodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    data = generate_iot_graph_dataset(num_nodes=args.nodes, seed=args.seed)
    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.size(1)//2} links, "
          f"budget = {budget(data.edge_index, args.rate)} flips ({args.rate:.0%})\n")

    for m in ["mlp", "gcn", "gat", "sage"]:
        set_seed(args.seed)
        model = train_model(build_model(m, data.num_features), data)
        print(f"[{m.upper():4s}] clean          {fmt(compute_metrics(model, data.x, data.edge_index, data.y, data.test_mask))}")
        for name in STRUCTURE_ATTACKS:
            ei = apply_structure_attack(name, data, args.rate, seed=args.seed)
            print(f"       {name:14s} {fmt(compute_metrics(model, data.x, ei, data.y, data.test_mask))}")
        print()


if __name__ == "__main__":
    main()
