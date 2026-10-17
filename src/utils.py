"""
Shared helpers: seeding, training with early stopping, and imbalance-aware metrics.

Every script in this repo goes through these functions so that numbers reported
by train.py, evaluate.py, the dashboard and the API are produced the same way.
"""

from __future__ import annotations

import copy
import os
import random
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.model import IoTGNN

DEFAULT_HPARAMS = dict(hidden_channels=64, lr=0.01, weight_decay=5e-4,
                       epochs=300, patience=50)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights so the minority (attack) class is not ignored."""
    counts = torch.bincount(y[mask], minlength=2).float().clamp(min=1)
    w = counts.sum() / (2.0 * counts)
    return w


def build_model(model_type: str, in_channels: int,
                hidden_channels: int = DEFAULT_HPARAMS["hidden_channels"]) -> IoTGNN:
    return IoTGNN(in_channels, hidden_channels, 2, model_type)


@torch.no_grad()
def compute_metrics(model, x, edge_index, y, mask, edge_weight=None) -> Dict[str, float]:
    """Accuracy alone is misleading at 85/15 class balance, so we report
    F1 / precision / recall for the *attack* class and ROC-AUC as well."""
    model.eval()
    out = model(x, edge_index, edge_weight)
    prob = out.exp()[:, 1]
    pred = out.argmax(-1)
    yt = y[mask].cpu().numpy()
    yp = pred[mask].cpu().numpy()
    ps = prob[mask].cpu().numpy()
    try:
        auc = roc_auc_score(yt, ps)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "roc_auc": float(auc),
    }


def train_model(
    model,
    data,
    lr: float = DEFAULT_HPARAMS["lr"],
    weight_decay: float = DEFAULT_HPARAMS["weight_decay"],
    epochs: int = DEFAULT_HPARAMS["epochs"],
    patience: int = DEFAULT_HPARAMS["patience"],
    edge_index=None,
    edge_weight=None,
    augment: Optional[Callable] = None,
    select_fn: Optional[Callable] = None,
    verbose: bool = False,
):
    """Train with weighted NLL and early stopping on validation F1.

    `augment(epoch, model, x, edge_index) -> (x, edge_index)` lets defences (DropEdge,
    adversarial training) change the inputs seen at each training step.
    `select_fn(model) -> float` replaces clean validation F1 as the checkpoint
    selection score (adversarial training selects on robust validation F1).
    """
    edge_index = data.edge_index if edge_index is None else edge_index
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = class_weights(data.y, data.train_mask)

    best_f1, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        x_in, ei_in, ew_in = data.x, edge_index, edge_weight
        if augment is not None:
            x_in, ei_in = augment(epoch, model, data.x, edge_index)
            ew_in = None if ei_in is not edge_index else edge_weight
        opt.zero_grad()
        out = model(x_in, ei_in, ew_in)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=w)
        loss.backward()
        opt.step()

        val = compute_metrics(model, data.x, edge_index, data.y, data.val_mask, edge_weight)
        if select_fn is not None:            # e.g. robust validation F1 for adversarial training
            score = select_fn(model)
        else:
            score = val["f1"] + 1e-3 * val["roc_auc"]  # tie-break on AUC
        if score > best_f1:
            best_f1, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and epoch % 25 == 0:
            print(f"  epoch {epoch:4d} | loss {loss.item():.4f} | val F1 {val['f1']:.3f}")
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def fmt(m: Dict[str, float]) -> str:
    return (f"Acc {m['accuracy']*100:5.1f}% | F1 {m['f1']:.3f} | "
            f"Recall {m['recall']:.3f} | AUC {m['roc_auc']:.3f}")
