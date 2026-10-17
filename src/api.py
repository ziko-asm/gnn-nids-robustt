"""
REST API that runs REAL experiments (no hard-coded numbers).

    uvicorn src.api:app --reload --port 8000
    open http://localhost:8000/docs

Endpoints
    GET  /health                 liveness
    GET  /results                benchmark produced by `python -m src.evaluate`
    GET  /graph                  nodes/links/labels/predictions for visualisation
    POST /analyze                train (cached) -> attack -> defend -> metrics
    POST /predict                score one device

Trained models are cached in memory per (model, defense, nodes, seed), so the
first request for a configuration takes a few seconds and later ones are fast.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.attack import STRUCTURE_ATTACKS, apply_structure_attack, undirected_pairs
from src.attack_feature import pgd_evasion
from src.attack_smart import gradient_attack
from src.dataset import DEVICE_TYPES, generate_iot_graph_dataset
from src.defend import train_defended
from src.utils import compute_metrics, set_seed

RESULTS_PATH = Path(os.environ.get("RESULTS_PATH", "results/results.json"))
MAX_NODES = int(os.environ.get("MAX_NODES", 1000))

app = FastAPI(
    title="GraphGuard IoT API",
    description="Adversarial robustness of GNN-based intrusion detection for IoT networks.",
    version="2.0.0",
)

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,  # wildcard origins and credentials are mutually exclusive
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ModelName = Literal["mlp", "gcn", "gat", "sage"]
DefenseName = Literal["none", "cosine_prune", "dropedge", "adv_train"]
AttackName = Literal["none", "random_remove", "random_add", "dice", "gradient", "feature_pgd"]


# --------------------------------------------------------------------- cache
@lru_cache(maxsize=8)
def get_data(nodes: int, seed: int):
    return generate_iot_graph_dataset(num_nodes=nodes, seed=seed)


@lru_cache(maxsize=24)
def get_model(model: str, defense: str, nodes: int, seed: int):
    if model == "mlp" and defense in ("cosine_prune", "dropedge"):
        raise HTTPException(400, "Graph defences do not apply to the feature-only MLP baseline")
    data = get_data(nodes, seed)
    return train_defended(defense, model, data, seed=seed)


# --------------------------------------------------------------------- schemas
class AnalyzeRequest(BaseModel):
    model: ModelName = "sage"
    defense: DefenseName = "none"
    attack: AttackName = "gradient"
    budget: float = Field(0.05, ge=0.0, le=1.0,
                          description="Structural: fraction of links flipped (<=0.3). "
                                      "feature_pgd: L-inf eps in feature std (<=1.0).")
    nodes: int = Field(300, ge=50)
    seed: int = Field(0, ge=0, le=99)


class PredictRequest(BaseModel):
    node_index: int = Field(..., ge=0)
    model: ModelName = "sage"
    nodes: int = Field(300, ge=50)
    seed: int = 0


# --------------------------------------------------------------------- routes
@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "benchmark_available": RESULTS_PATH.exists()}


@app.get("/")
def root():
    return {"service": app.title, "docs": "/docs", **health()}


@app.get("/results")
def results():
    if not RESULTS_PATH.exists():
        raise HTTPException(404, "No benchmark yet. Run: python -m src.evaluate")
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.get("/graph")
def graph(nodes: int = 300, seed: int = 0, model: ModelName = "sage"):
    nodes = min(max(nodes, 50), MAX_NODES)
    data = get_data(nodes, seed)
    net, prep = get_model(model, "none", nodes, seed)
    with torch.no_grad():
        prob = net(*prep(data.x, data.edge_index)).exp()[:, 1]
    split = torch.where(data.train_mask, 0, torch.where(data.val_mask, 1, 2))
    return {
        "nodes": [{"id": i, "label": int(data.y[i]), "p_compromised": round(float(prob[i]), 4),
                   "device_type": DEVICE_TYPES[int(data.device_type[i])],
                   "community": int(data.community[i]),
                   "split": ["train", "val", "test"][int(split[i])]}
                  for i in range(data.num_nodes)],
        "links": undirected_pairs(data.edge_index).tolist(),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    nodes = min(req.nodes, MAX_NODES)
    data = get_data(nodes, req.seed)
    net, prep = get_model(req.model, req.defense, nodes, req.seed)
    set_seed(req.seed)

    x, ei = data.x, data.edge_index
    if req.attack in STRUCTURE_ATTACKS or req.attack == "gradient":
        if req.model == "mlp":
            raise HTTPException(400, "Structural attacks cannot affect the feature-only MLP baseline")
        if req.budget > 0.3:
            raise HTTPException(400, "Structural budget must be <= 0.3")
        ei = (gradient_attack(data, req.budget, seed=req.seed) if req.attack == "gradient"
              else apply_structure_attack(req.attack, data, req.budget, seed=req.seed))
    elif req.attack == "feature_pgd":
        x0, ei0 = prep(data.x, data.edge_index)
        x = pgd_evasion(net, x0, ei0, data.y == 1, req.budget)

    xc, eic = prep(data.x, data.edge_index)
    xa, eia = prep(x, ei)
    clean = compute_metrics(net, xc, eic, data.y, data.test_mask)
    attacked = compute_metrics(net, xa, eia, data.y, data.test_mask)

    with torch.no_grad():
        p_clean = net(xc, eic).argmax(-1)
        p_att = net(xa, eia).argmax(-1)
    before = set(map(tuple, undirected_pairs(data.edge_index).tolist()))
    after = set(map(tuple, undirected_pairs(ei).tolist()))
    evaded = torch.where(data.test_mask & (data.y == 1) & (p_clean == 1) & (p_att == 0))[0]

    return {
        "config": req.model_dump() | {"nodes": nodes},
        "clean": clean,
        "attacked": attacked,
        "delta_f1": round(attacked["f1"] - clean["f1"], 4),
        "links_removed": sorted(before - after)[:500],
        "links_added": sorted(after - before)[:500],
        "evaded_test_bots": evaded.tolist(),
        "prune_threshold": getattr(net, "prune_threshold", None),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    nodes = min(req.nodes, MAX_NODES)
    data = get_data(nodes, req.seed)
    if req.node_index >= data.num_nodes:
        raise HTTPException(400, f"node_index must be < {data.num_nodes}")
    net, prep = get_model(req.model, "none", nodes, req.seed)
    with torch.no_grad():
        p = net(*prep(data.x, data.edge_index)).exp()[req.node_index]
    cls = int(p.argmax())
    return {"node_index": req.node_index, "prediction": ["benign", "compromised"][cls],
            "p_compromised": round(float(p[1]), 4), "true_label": int(data.y[req.node_index])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
