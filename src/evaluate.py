"""
Full robustness benchmark. Produces every number shown on the website.

    python -m src.evaluate            # full run (5 seeds)  ~10-20 min on a laptop CPU
    python -m src.evaluate --quick    # 2 seeds, fewer budgets, ~2-4 min

Outputs
    results/results.json       raw per-seed records + mean/std summary + metadata
    results/robustness_*.png   F1 vs attack budget, one curve per model
    results/defenses.png       defence comparison under the gradient attack

Protocol
    * Synthetic IoT graph (src/dataset.py), 60/20/20 stratified split.
    * Models trained on the clean graph with early stopping on validation F1.
    * All attacks are evasion (test-time). Metrics on the test split only.
    * Each configuration repeated over several seeds (new graph + new init);
      we report mean +/- std.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.attack import STRUCTURE_ATTACKS, apply_structure_attack
from src.attack_feature import pgd_evasion
from src.attack_smart import gradient_attack, train_surrogate
from src.dataset import FEATURE_NAMES, generate_iot_graph_dataset
from src.defend import train_defended
from src.utils import DEFAULT_HPARAMS, compute_metrics, set_seed

MODELS = ["mlp", "gcn", "gat", "sage"]
DEFENSES_FOR = {"mlp": ["none", "adv_train"],
                "gcn": ["none", "cosine_prune", "dropedge", "adv_train"],
                "gat": ["none", "cosine_prune", "dropedge", "adv_train"],
                "sage": ["none", "cosine_prune", "dropedge", "adv_train"]}


@torch.no_grad()
def latency_ms(model, x, ei, reps: int = 30) -> float:
    model.eval()
    model(x, ei)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(x, ei)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def graph_stats(data) -> dict:
    ei, y = data.edge_index, data.y
    return {
        "nodes": int(data.num_nodes),
        "links": int(ei.size(1) // 2),
        "features": int(data.num_features),
        "attack_share": float((y == 1).float().mean()),
        "edge_homophily": float((y[ei[0]] == y[ei[1]]).float().mean()),
        "mean_degree": float(ei.size(1) / data.num_nodes),
    }


def run(seeds, nodes, struct_rates, feat_eps, out_dir: Path):
    records, stats = [], []
    t_start = time.time()
    for seed in seeds:
        set_seed(seed)
        data = generate_iot_graph_dataset(num_nodes=nodes, seed=seed)
        stats.append(graph_stats(data))
        print(f"\n=== seed {seed} | {stats[-1]['nodes']} nodes, {stats[-1]['links']} links ===")

        # attacked graphs are victim-independent: build once per seed
        surrogate = train_surrogate(data, seed=seed)
        attacked = {("clean", 0.0): data.edge_index}
        for r in struct_rates:
            if r == 0:
                continue
            for a in STRUCTURE_ATTACKS:
                attacked[(a, r)] = apply_structure_attack(a, data, r, seed=seed)
            attacked[("gradient", r)] = gradient_attack(data, r, surrogate=surrogate, seed=seed)

        for m in MODELS:
            for d in DEFENSES_FOR[m]:
                t0 = time.time()
                model, prep = train_defended(d, m, data, seed=seed)
                base = dict(seed=seed, model=m, defense=d)

                x0, ei0 = prep(data.x, data.edge_index)
                clean = compute_metrics(model, x0, ei0, data.y, data.test_mask)
                lat = latency_ms(model, x0, ei0)
                records.append({**base, "attack": "clean", "budget": 0.0, **clean, "latency_ms": lat})

                if m != "mlp":  # structure attacks cannot affect a feature-only model
                    for (a, r), ei in attacked.items():
                        if a == "clean":
                            continue
                        x1, ei1 = prep(data.x, ei)
                        met = compute_metrics(model, x1, ei1, data.y, data.test_mask)
                        records.append({**base, "attack": a, "budget": r, **met})

                for eps in feat_eps:
                    if eps == 0:
                        continue
                    x_adv = pgd_evasion(model, x0, ei0, data.y == 1, eps)
                    x2, ei2 = prep(x_adv, data.edge_index)
                    met = compute_metrics(model, x2, ei2, data.y, data.test_mask)
                    records.append({**base, "attack": "feature_pgd", "budget": eps, **met})

                print(f"  {m:4s} + {d:12s} clean F1 {clean['f1']:.3f}  ({time.time()-t0:4.1f}s)")

    summary = summarize(records)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_sec": round(time.time() - t_start, 1),
        "seeds": list(seeds),
        "graph": {k: float(np.mean([s[k] for s in stats])) for k in stats[0]},
        "feature_names": FEATURE_NAMES,
        "hyperparameters": DEFAULT_HPARAMS,
        "structural_budgets": struct_rates,
        "feature_eps": feat_eps,
        "budget_definition": "structural: fraction of undirected links flipped; "
                             "feature: L-inf radius in feature standard deviations, "
                             "applied only to compromised devices",
        "setting": "evasion (test-time), metrics on test split, mean +/- std over seeds",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(terse=True)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "records": records}, f, indent=1)
    print(f"\n[*] Saved {len(records)} records -> {out_dir/'results.json'} "
          f"({meta['runtime_sec']}s)")
    return summary


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["defense"], r["attack"], r["budget"])].append(r)
    out = []
    for (m, d, a, b), rs in sorted(groups.items()):
        row = {"model": m, "defense": d, "attack": a, "budget": b, "n": len(rs)}
        for k in ("accuracy", "f1", "precision", "recall", "roc_auc", "latency_ms"):
            vals = [r[k] for r in rs if k in r and not np.isnan(r[k])]
            if vals:
                row[k] = round(float(np.mean(vals)), 4)
                row[k + "_std"] = round(float(np.std(vals)), 4)
        out.append(row)
    return out


def plot(summary, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(model, defense, attack):
        clean = [s for s in summary if s["model"] == model and s["defense"] == defense and s["attack"] == "clean"]
        pts = sorted([s for s in summary if s["model"] == model and s["defense"] == defense and s["attack"] == attack],
                     key=lambda s: s["budget"])
        pts = clean + pts
        return [p["budget"] for p in pts], [p["f1"] for p in pts], [p.get("f1_std", 0) for p in pts]

    colors = {"mlp": "#64748b", "gcn": "#2563eb", "gat": "#d97706", "sage": "#059669"}
    for attack in ["random_remove", "random_add", "dice", "gradient", "feature_pgd"]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for m in MODELS:
            if attack != "feature_pgd" and m == "mlp":
                continue
            x, y, e = series(m, "none", attack)
            if len(x) < 2:
                continue
            x = [v * (1 if attack == "feature_pgd" else 100) for v in x]
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3, lw=2, color=colors[m], label=m.upper())
        ax.set_xlabel("PGD eps (feature std)" if attack == "feature_pgd" else "Edges flipped (%)")
        ax.set_ylabel("Attack-class F1 (test)")
        ax.set_title(f"Robustness: {attack}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"robustness_{attack}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for d, c in zip(["none", "cosine_prune", "dropedge", "adv_train"],
                    ["#64748b", "#2563eb", "#d97706", "#059669"]):
        x, y, e = series("gcn", d, "gradient")
        if len(x) >= 2:
            ax.errorbar([v * 100 for v in x], y, yerr=e, marker="o", capsize=3, lw=2, color=c, label=d)
    ax.set_xlabel("Edges flipped (%)")
    ax.set_ylabel("Attack-class F1 (test)")
    ax.set_title("GCN defences under gradient attack")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "defenses.png", dpi=150)
    plt.close(fig)
    print(f"[*] Plots saved in {out_dir}/")


def main():
    p = argparse.ArgumentParser(description="GNN-NIDS robustness benchmark")
    p.add_argument("--quick", action="store_true", help="2 seeds, fewer budgets")
    p.add_argument("--nodes", type=int, default=1500)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    if a.quick:
        seeds, rates, eps = [0, 1], [0.0, 0.05, 0.10, 0.20], [0.0, 0.5, 1.0]
    else:
        seeds, rates, eps = list(range(a.seeds)), [0.0, 0.02, 0.05, 0.10, 0.15, 0.20], [0.0, 0.25, 0.5, 0.75, 1.0]
    out = Path(a.out)
    summary = run(seeds, a.nodes, rates, eps, out)
    plot(summary, out)
    table = markdown_table(summary, struct_budget=0.10, feat_eps=0.5)
    (out / "summary_table.md").write_text(table, encoding="utf-8")
    print("\n" + table + f"\n[*] Table saved -> {out/'summary_table.md'} (paste it into README.md)")


def markdown_table(summary, struct_budget: float, feat_eps: float) -> str:
    def cell(m, d, a, b):
        s = next((s for s in summary if s["model"] == m and s["defense"] == d
                  and s["attack"] == a and s["budget"] == b), None)
        return f"{s['f1']:.3f} ± {s['f1_std']:.3f}" if s else "—"

    cols = [("clean", 0.0, "Clean"), ("random_remove", struct_budget, f"Random −{struct_budget:.0%}"),
            ("dice", struct_budget, f"DICE {struct_budget:.0%}"),
            ("gradient", struct_budget, f"Gradient {struct_budget:.0%}"),
            ("feature_pgd", feat_eps, f"PGD ε={feat_eps}")]
    lines = ["| Model | Defence | " + " | ".join(c[2] for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for m in MODELS:
        for d in DEFENSES_FOR[m]:
            lines.append(f"| {m.upper()} | {d} | " + " | ".join(cell(m, d, a, b) for a, b, _ in cols) + " |")
    return "Attack-class F1 on test devices (mean ± std over seeds)\n\n" + "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
