"""
Train a single model on the synthetic IoT graph and save a checkpoint.

    python -m src.train --model sage
    python -m src.train --model gcn --nodes 1000 --seed 3

The checkpoint (checkpoints/<model>.pt) is what src/api.py serves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dataset import generate_iot_graph_dataset
from src.model import MODEL_TYPES
from src.utils import DEFAULT_HPARAMS, build_model, compute_metrics, fmt, set_seed, train_model


def main():
    p = argparse.ArgumentParser(description="Train a GNN for IoT intrusion detection")
    p.add_argument("--model", default="sage", choices=MODEL_TYPES)
    p.add_argument("--nodes", type=int, default=600)
    p.add_argument("--epochs", type=int, default=DEFAULT_HPARAMS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULT_HPARAMS["lr"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    set_seed(args.seed)
    data = generate_iot_graph_dataset(num_nodes=args.nodes, seed=args.seed)
    print(f"[*] Graph: {data.num_nodes} devices, {data.edge_index.size(1)//2} links, "
          f"{int(data.y.sum())} compromised ({data.y.float().mean():.1%})")
    print(f"[*] Training {args.model.upper()} (max {args.epochs} epochs, early stopping on val F1)")

    model = build_model(args.model, data.num_features)
    model = train_model(model, data, lr=args.lr, epochs=args.epochs, verbose=True)

    for split in ("train_mask", "val_mask", "test_mask"):
        m = compute_metrics(model, data.x, data.edge_index, data.y, getattr(data, split))
        print(f"    {split.replace('_mask', ''):5s}  {fmt(m)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model}.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": {"model": args.model, "nodes": args.nodes, "seed": args.seed,
                           "in_channels": data.num_features,
                           "hidden_channels": DEFAULT_HPARAMS["hidden_channels"]}}, path)
    print(f"[*] Saved checkpoint -> {path}")


if __name__ == "__main__":
    main()
