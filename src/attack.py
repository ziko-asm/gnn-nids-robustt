import argparse
import torch
from torch_geometric.utils import dropout_edge
from src.dataset import generate_iot_graph_dataset
from src.model import IoTGNN

def random_edge_attack(edge_index, drop_prob=0.2):
    """
    Simulates a structural perturbation attack on the graph
    by randomly removing a percentage (drop_prob) of edges.
    """
    edge_index_attacked, _ = dropout_edge(edge_index, p=drop_prob)
    return edge_index_attacked

def evaluate_robustness(model_type='sage', attack_rate=0.3):
    # 1. Generate Dataset and set train/test masks
    data = generate_iot_graph_dataset(num_nodes=300)
    
    if not hasattr(data, 'train_mask'):
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        train_size = int(num_nodes * 0.7)
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[indices[:train_size]] = True
        
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[indices[train_size:]] = True

    # 2. Initialize Model and Optimizer
    model = IoTGNN(
        in_channels=data.num_features,
        hidden_channels=64,
        out_channels=2,
        model_type=model_type
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # 3. Model Training
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # 4. Evaluate Clean Accuracy (Before Attack)
    model.eval()
    with torch.no_grad():
        out_clean = model(data.x, data.edge_index)
        pred_clean = out_clean.argmax(dim=-1)
        correct_clean = (pred_clean[data.test_mask] == data.y[data.test_mask]).sum()
        acc_clean = int(correct_clean) / int(data.test_mask.sum())

    # 5. Apply Structural Attack (Edge Perturbation)
    edge_index_attacked = random_edge_attack(data.edge_index, drop_prob=attack_rate)

    # 6. Evaluate Attacked Accuracy (After Attack)
    with torch.no_grad():
        out_attacked = model(data.x, edge_index_attacked)
        pred_attacked = out_attacked.argmax(dim=-1)
        correct_attacked = (pred_attacked[data.test_mask] == data.y[data.test_mask]).sum()
        acc_attacked = int(correct_attacked) / int(data.test_mask.sum())

    print(f"\n[+] Model Architecture: {model_type.upper()}")
    print(f"    - Clean Accuracy    : {acc_clean * 100:.2f}%")
    print(f"    - Attacked Accuracy : {acc_attacked * 100:.2f}% (Edge Drop Rate: {int(attack_rate*100)}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Robustness Evaluation for GNN-NIDS")
    parser.add_argument("--rate", type=float, default=0.3, help="Edge perturbation attack rate (default: 0.3)")
    args = parser.parse_args()

    print("==================================================")
    print("   GNN-NIDS Adversarial Robustness Evaluation     ")
    print("==================================================")
    for m in ['gcn', 'gat', 'sage']:
        evaluate_robustness(model_type=m, attack_rate=args.rate)