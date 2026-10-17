"""Run with:  pytest -q"""

import pytest
import torch

from src.attack import STRUCTURE_ATTACKS, apply_structure_attack, budget, undirected_pairs
from src.attack_feature import pgd_evasion
from src.attack_smart import gradient_attack
from src.dataset import FEATURE_NAMES, generate_graph_arrays, generate_iot_graph_dataset
from src.defend import cosine_prune
from src.model import MODEL_TYPES, IoTGNN
from src.utils import build_model, compute_metrics, set_seed, train_model

N = 200


@pytest.fixture(scope="module")
def data():
    return generate_iot_graph_dataset(num_nodes=N, seed=0)


@pytest.fixture(scope="module")
def trained_gcn(data):
    set_seed(0)
    return train_model(build_model("gcn", data.num_features), data, epochs=150)


# ---------------------------------------------------------------- dataset
def test_dataset_shapes(data):
    assert data.num_nodes == N
    assert data.x.shape == (N, len(FEATURE_NAMES))
    assert data.edge_index.shape[0] == 2
    assert data.y.shape == (N,)


def test_graph_is_undirected_without_self_loops(data):
    ei = data.edge_index
    assert not (ei[0] == ei[1]).any()
    fwd = set(map(tuple, ei.t().tolist()))
    assert all((v, u) in fwd for u, v in fwd)


def test_masks_are_disjoint_and_stratified(data):
    m = data.train_mask.int() + data.val_mask.int() + data.test_mask.int()
    assert (m == 1).all()
    for mask in (data.train_mask, data.val_mask, data.test_mask):
        assert data.y[mask].sum() > 0, "every split must contain attack nodes"


def test_dataset_is_deterministic():
    a = generate_graph_arrays(150, seed=7)
    b = generate_graph_arrays(150, seed=7)
    assert (a.x == b.x).all() and (a.edge_index == b.edge_index).all() and (a.y == b.y).all()


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_forward_pass(data, model_type):
    model = IoTGNN(data.num_features, 32, 2, model_type).eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
    assert out.shape == (N, 2)
    assert torch.allclose(out.exp().sum(1), torch.ones(N), atol=1e-4)


def test_invalid_model_raises():
    with pytest.raises(ValueError):
        IoTGNN(16, 32, 2, "transformer")


def test_model_learns_signal(data, trained_gcn):
    m = compute_metrics(trained_gcn, data.x, data.edge_index, data.y, data.test_mask)
    majority_acc = 1 - data.y[data.test_mask].float().mean().item()
    assert m["roc_auc"] > 0.75, "model should clearly beat chance"
    assert m["recall"] > 0.3, "model must not collapse to 'everything benign'"
    assert m["accuracy"] >= majority_acc - 0.05


# ---------------------------------------------------------------- attacks
@pytest.mark.parametrize("name", list(STRUCTURE_ATTACKS))
def test_structure_attack_respects_budget(data, name):
    rate = 0.1
    ei = apply_structure_attack(name, data, rate, seed=0)
    before = set(map(tuple, undirected_pairs(data.edge_index).tolist()))
    after = set(map(tuple, undirected_pairs(ei).tolist()))
    flips = len(before ^ after)
    assert 0 < flips <= budget(data.edge_index, rate)


def test_gradient_attack_budget_and_effect(data, trained_gcn):
    rate = 0.1
    ei = gradient_attack(data, rate, steps=5, seed=0)
    before = set(map(tuple, undirected_pairs(data.edge_index).tolist()))
    after = set(map(tuple, undirected_pairs(ei).tolist()))
    assert len(before ^ after) == budget(data.edge_index, rate)
    clean = compute_metrics(trained_gcn, data.x, data.edge_index, data.y, data.test_mask)
    hit = compute_metrics(trained_gcn, data.x, ei, data.y, data.test_mask)
    # transfer attack: allow small noise, but it must not help the defender
    assert hit["f1"] <= clean["f1"] + 0.05


def test_pgd_only_touches_controlled_rows_within_eps(data, trained_gcn):
    ctrl = data.y == 1
    eps = 0.3
    x_adv = pgd_evasion(trained_gcn, data.x, data.edge_index, ctrl, eps, steps=10)
    diff = (x_adv - data.x).abs()
    assert diff[~ctrl].max() == 0
    assert diff[ctrl].max() <= eps + 1e-5


def test_cosine_prune_only_removes(data):
    ei = cosine_prune(data.x, data.edge_index, 0.0)
    assert ei.size(1) <= data.edge_index.size(1)