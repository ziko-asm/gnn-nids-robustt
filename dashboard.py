"""
Interactive research dashboard. Every number on screen is computed, none is typed in.

    streamlit run dashboard.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from src.attack import apply_structure_attack
from src.attack_feature import pgd_evasion
from src.attack_smart import gradient_attack
from src.dataset import generate_iot_graph_dataset
from src.defend import train_defended
from src.utils import compute_metrics

st.set_page_config(page_title="GraphGuard IoT - GNN Robustness Lab", page_icon="🛡️", layout="wide")
st.title("GraphGuard IoT: Adversarial Robustness of GNN Intrusion Detection")
st.caption("Synthetic IoT network · evasion attacks · metrics on held-out test devices")

MODELS = {"GraphSAGE": "sage", "GCN": "gcn", "GAT": "gat", "MLP (no graph)": "mlp"}
ATTACKS = {"Gradient (greedy adjacency)": "gradient", "DICE": "dice",
           "Random link removal": "random_remove", "Random link injection": "random_add",
           "Feature PGD evasion": "feature_pgd"}
DEFENSES = {"None": "none", "Cosine pruning": "cosine_prune",
            "DropEdge": "dropedge", "PGD adversarial training": "adv_train"}


@st.cache_resource(show_spinner="Generating graph...")
def get_data(nodes, seed):
    return generate_iot_graph_dataset(num_nodes=nodes, seed=seed)


@st.cache_resource(show_spinner="Training model (cached after first run)...")
def get_model(model, defense, nodes, seed):
    return train_defended(defense, model, get_data(nodes, seed), seed=seed)


@st.cache_data(show_spinner="Running gradient attack...")
def get_grad_attack(nodes, seed, rate):
    return gradient_attack(get_data(nodes, seed), rate, seed=seed)


def attacked_inputs(attack, data, net, prep, budget, nodes, seed):
    if attack == "feature_pgd":
        x0, ei0 = prep(data.x, data.edge_index)
        return pgd_evasion(net, x0, ei0, data.y == 1, budget), data.edge_index
    if attack == "gradient":
        return data.x, get_grad_attack(nodes, seed, budget)
    return data.x, apply_structure_attack(attack, data, budget, seed=seed)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Experiment")
    model_label = st.selectbox("Model", list(MODELS))
    attack_label = st.selectbox("Attack", list(ATTACKS))
    model, attack = MODELS[model_label], ATTACKS[attack_label]
    allowed_def = ["None", "PGD adversarial training"] if model == "mlp" else list(DEFENSES)
    defense = DEFENSES[st.selectbox("Defence", allowed_def)]
    if attack == "feature_pgd":
        budget = st.slider("PGD eps (feature std)", 0.0, 1.5, 0.5, 0.05)
    else:
        budget = st.slider("Links flipped (%)", 0, 30, 5) / 100
    nodes = st.select_slider("Devices", [300, 600, 1000], 600)
    seed = st.number_input("Seed", 0, 99, 0)

tab_live, tab_bench, tab_xai = st.tabs(["Live experiment", "Benchmark (all seeds)", "Explain a detection"])

# ------------------------------------------------------------------ live
with tab_live:
    if model == "mlp" and attack != "feature_pgd":
        st.warning("The MLP baseline ignores the graph, so structural attacks have no effect on it.")
    data = get_data(nodes, seed)
    net, prep = get_model(model, defense, nodes, seed)
    xc, eic = prep(data.x, data.edge_index)
    clean = compute_metrics(net, xc, eic, data.y, data.test_mask)
    xa, eia = prep(*attacked_inputs(attack, data, net, prep, budget, nodes, seed))
    att = compute_metrics(net, xa, eia, data.y, data.test_mask)

    c = st.columns(4)
    for col, k, name in zip(c, ["f1", "recall", "roc_auc", "accuracy"],
                            ["Attack-class F1", "Detection rate (recall)", "ROC-AUC", "Accuracy"]):
        col.metric(name, f"{att[k]:.3f}", f"{att[k] - clean[k]:+.3f} vs clean")
    st.caption(f"Clean: F1 {clean['f1']:.3f} · recall {clean['recall']:.3f} · "
               f"{int(data.test_mask.sum())} test devices, {int((data.y[data.test_mask]).sum())} compromised")

    st.subheader("F1 vs attack budget (this seed)")
    grid = [0, 0.25, 0.5, 0.75, 1.0] if attack == "feature_pgd" else [0, 0.02, 0.05, 0.1, 0.15, 0.2]
    ys = []
    for b in grid:
        if b == 0:
            ys.append(clean["f1"])
            continue
        xb, eib = prep(*attacked_inputs(attack, data, net, prep, b, nodes, seed))
        ys.append(compute_metrics(net, xb, eib, data.y, data.test_mask)["f1"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot([g * (1 if attack == "feature_pgd" else 100) for g in grid], ys, marker="o", lw=2)
    ax.set_xlabel("eps" if attack == "feature_pgd" else "Links flipped (%)")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ------------------------------------------------------------------ benchmark
with tab_bench:
    path = Path("results/results.json")
    if not path.exists():
        st.info("No benchmark yet. Run `python -m src.evaluate --quick` and reload.")
    else:
        res = json.loads(path.read_text(encoding="utf-8"))
        st.caption(f"Generated {res['meta']['generated_at']} · seeds {res['meta']['seeds']} · "
                   f"{res['meta']['setting']}")
        df = pd.DataFrame(res["summary"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        for png in sorted(Path("results").glob("*.png")):
            st.image(str(png), caption=png.stem)

# ------------------------------------------------------------------ XAI
with tab_xai:
    if model == "mlp":
        st.info("GNNExplainer needs a graph model. Pick GCN, GAT or GraphSAGE.")
    else:
        from src.explain import explain_node
        test_bots = torch.where(data.test_mask & (data.y == 1))[0].tolist()
        node = st.selectbox("Compromised test device", test_bots)
        if st.button("Explain"):
            with st.spinner("Running GNNExplainer..."):
                r = explain_node(net, data, int(node))
            st.write(f"P(compromised) = **{r['p_compromised']:.3f}**")
            st.bar_chart(pd.DataFrame(r["top_features"]).set_index("feature"))
            st.dataframe(pd.DataFrame(r["top_links"]), hide_index=True)
            a, b = st.columns(2)
            a.metric("Fidelity+ (higher better)", f"{r['fidelity_plus']:.3f}")
            b.metric("Fidelity- (lower better)", f"{r['fidelity_minus']:.3f}")
