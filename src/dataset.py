"""
Synthetic IoT network graph generator with a *learnable* attack signal.

Nodes are IoT devices, edges are observed communication links and node features
are per-device traffic statistics aggregated over a time window. A fraction of
devices are compromised (botnet members). Unlike a purely random graph, the
generator plants realistic, *overlapping* evidence of compromise in two places:

1. Features  - compromised devices shift a random subset of scan/flood related
               statistics (SYN ratio, failed connections, unique destination
               ports, ...). The shift is partial and noisy, so features alone are
               not enough to separate the classes perfectly.
2. Structure - benign devices talk inside their own site/VLAN (community) and to
               their gateway; compromised devices also talk to each other (C2 /
               P2P coordination) and probe devices in *other* communities.

This makes the task non-trivial and gives graph models something real to
exploit, which is exactly what the adversarial attacks then try to break.

The generator is fully deterministic given `seed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

FEATURE_NAMES = [
    "pkt_rate", "byte_rate", "mean_pkt_size", "flow_duration",
    "unique_dst_ips", "unique_dst_ports", "syn_ratio", "failed_conn_ratio",
    "dns_query_rate", "mqtt_publish_rate", "tcp_ratio", "udp_ratio",
    "iat_mean", "iat_std", "payload_entropy", "night_activity",
]

DEVICE_TYPES = ["camera", "sensor", "thermostat", "smart_plug", "plc", "gateway"]

# Statistics that typically move when a device scans / floods / beacons.
ATTACK_FEATURES = [
    FEATURE_NAMES.index(f)
    for f in ("pkt_rate", "unique_dst_ips", "unique_dst_ports", "syn_ratio",
              "failed_conn_ratio", "dns_query_rate", "iat_std", "night_activity")
]


@dataclass
class GraphArrays:
    x: np.ndarray            # [N, F] float32, standardised
    edge_index: np.ndarray   # [2, E] int64, undirected (both directions), no self loops
    y: np.ndarray            # [N] int64, 0 = benign, 1 = compromised
    device_type: np.ndarray  # [N] int64 index into DEVICE_TYPES
    community: np.ndarray    # [N] int64 site / VLAN id
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


def _stratified_split(y: np.ndarray, rng: np.random.Generator,
                      train: float = 0.6, val: float = 0.2):
    n = len(y)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n_tr = int(round(len(idx) * train))
        n_va = int(round(len(idx) * val))
        train_mask[idx[:n_tr]] = True
        val_mask[idx[n_tr:n_tr + n_va]] = True
        test_mask[idx[n_tr + n_va:]] = True
    return train_mask, val_mask, test_mask


def generate_graph_arrays(
    num_nodes: int = 300,
    num_features: int = 16,
    attack_ratio: float = 0.15,
    signal_strength: float = 1.5,
    seed: int = 0,
) -> GraphArrays:
    """Pure-numpy generator (no torch dependency, easy to test and reuse)."""
    if num_nodes < 20:
        raise ValueError("num_nodes must be >= 20")
    if num_features < len(FEATURE_NAMES):
        raise ValueError(f"num_features must be >= {len(FEATURE_NAMES)}")

    rng = np.random.default_rng(seed)
    n = num_nodes

    # ---- communities (sites / VLANs) and device roles ---------------------
    n_comm = max(3, n // 50)
    community = rng.integers(0, n_comm, size=n)
    device_type = rng.integers(0, len(DEVICE_TYPES) - 1, size=n)  # non-gateway types
    gateways = []
    for c in range(n_comm):
        members = np.where(community == c)[0]
        if len(members) == 0:
            members = np.array([rng.integers(0, n)])
            community[members[0]] = c
        g = int(members[0])
        device_type[g] = DEVICE_TYPES.index("gateway")
        gateways.append(g)
    gateways = np.array(gateways)

    # ---- labels: compromised devices (never a gateway) ---------------------
    candidates = np.setdiff1d(np.arange(n), gateways)
    n_attack = max(2, int(round(n * attack_ratio)))
    attackers = rng.choice(candidates, size=n_attack, replace=False)
    y = np.zeros(n, dtype=np.int64)
    y[attackers] = 1

    # ---- features -----------------------------------------------------------
    profile_rng = np.random.default_rng(12345)  # device profiles fixed across seeds
    profiles = profile_rng.normal(0.0, 1.0, size=(len(DEVICE_TYPES), num_features))
    x = profiles[device_type] + rng.normal(0.0, 1.0, size=(n, num_features))
    for a in attackers:
        k = rng.integers(2, len(ATTACK_FEATURES) + 1)          # stealth: partial footprint
        feats = rng.choice(ATTACK_FEATURES, size=k, replace=False)
        x[a, feats] += signal_strength * rng.uniform(0.6, 1.6, size=k)

    # ---- edges --------------------------------------------------------------
    edges = set()

    def add(u: int, v: int) -> None:
        if u != v:
            edges.add((min(u, v), max(u, v)))

    for c in range(n_comm):
        members = np.where(community == c)[0]
        g = gateways[c]
        for m in members:
            add(int(m), int(g))                                  # device <-> gateway
        # sparse intra-site peer traffic (avg ~3 peers per device)
        m_cnt = len(members)
        if m_cnt > 2:
            n_pairs = int(1.5 * m_cnt)
            u = rng.choice(members, size=n_pairs)
            v = rng.choice(members, size=n_pairs)
            for a, b in zip(u, v):
                add(int(a), int(b))
    # gateways form a backbone
    for i in range(len(gateways)):
        add(int(gateways[i]), int(gateways[(i + 1) % len(gateways)]))
    # benign cross-site noise (cloud sync, roaming devices, ...)
    for _ in range(int(0.1 * n)):
        add(int(rng.integers(0, n)), int(rng.integers(0, n)))
    # botnet: C2 / P2P coordination + cross-site probing
    for a in attackers:
        peers = rng.choice(attackers, size=min(2, n_attack - 1), replace=False)
        for p in peers:
            add(int(a), int(p))
        foreign = np.where(community != community[a])[0]
        if len(foreign):
            for t in rng.choice(foreign, size=min(3, len(foreign)), replace=False):
                add(int(a), int(t))

    e = np.array(sorted(edges), dtype=np.int64).T
    edge_index = np.concatenate([e, e[::-1]], axis=1)

    # ---- standardise features (statistics from training nodes only) --------
    train_mask, val_mask, test_mask = _stratified_split(y, rng)
    mu = x[train_mask].mean(0, keepdims=True)
    sd = x[train_mask].std(0, keepdims=True) + 1e-6
    x = ((x - mu) / sd).astype(np.float32)

    return GraphArrays(x, edge_index, y, device_type, community,
                       train_mask, val_mask, test_mask)


def generate_iot_graph_dataset(
    num_nodes: int = 300,
    num_features: int = 16,
    attack_ratio: float = 0.15,
    signal_strength: float = 1.5,
    seed: int = 0,
) -> Data:
    """Return a PyG `Data` object with train / val / test masks."""
    g = generate_graph_arrays(num_nodes, num_features, attack_ratio, signal_strength, seed)
    data = Data(
        x=torch.from_numpy(g.x),
        edge_index=torch.from_numpy(g.edge_index),
        y=torch.from_numpy(g.y),
        train_mask=torch.from_numpy(g.train_mask),
        val_mask=torch.from_numpy(g.val_mask),
        test_mask=torch.from_numpy(g.test_mask),
    )
    data.device_type = torch.from_numpy(g.device_type)
    data.community = torch.from_numpy(g.community)
    data.feature_names = FEATURE_NAMES[:num_features] + [
        f"extra_{i}" for i in range(num_features - len(FEATURE_NAMES))
    ]
    return data
