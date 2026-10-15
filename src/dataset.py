import torch
from torch_geometric.data import Data
import numpy as np

def generate_iot_graph_dataset(num_nodes: int = 300, num_features: int = 16) -> Data:
    """
    Generates a synthetic graph dataset simulating IoT network traffic.
    Nodes represent IoT devices, and edges represent network connections between them.
    """
    # 1. Node Features: Statistical characteristics of network traffic
    x = torch.randn((num_nodes, num_features), dtype=torch.float)

    # 2. Graph Connectivity (Edge Index): Simulated network connections
    num_edges = num_nodes * 3
    source_nodes = np.random.randint(0, num_nodes, size=(num_edges,))
    target_nodes = np.random.randint(0, num_nodes, size=(num_edges,))

    edges_np = np.array([source_nodes, target_nodes], dtype=np.int64)
    edge_index = torch.tensor(edges_np, dtype=torch.long)

    # 3. Labels: 0 -> Normal Traffic, 1 -> Attack / Zero-day Anomaly
    y = torch.zeros(num_nodes, dtype=torch.long)
    attack_nodes = np.random.choice(num_nodes, size=int(num_nodes * 0.15), replace=False)
    y[attack_nodes] = 1

    # 4. Train and Test Masks
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * 0.7)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[indices[train_size:]] = True

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
    return data