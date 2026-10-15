import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import dropout_edge
from src.dataset import generate_iot_graph_dataset
from src.model import IoTGNN

def evaluate_curve():
    print("[*] Starting adversarial robustness curve evaluation...")
    data = generate_iot_graph_dataset(num_nodes=300)
    
    if not hasattr(data, 'train_mask'):
        indices = torch.randperm(data.num_nodes)
        train_size = int(data.num_nodes * 0.7)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[indices[:train_size]] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[indices[train_size:]] = True

    models = ['gcn', 'gat', 'sage']
    attack_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = {m: [] for m in models}

    for model_type in models:
        # Train base model
        model = IoTGNN(data.num_features, 64, 2, model_type)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        for rate in attack_rates:
            if rate == 0.0:
                edge_idx = data.edge_index
            else:
                edge_idx, _ = dropout_edge(data.edge_index, p=rate)

            with torch.no_grad():
                out = model(data.x, edge_idx)
                pred = out.argmax(dim=-1)
                acc = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())
                results[model_type].append(acc * 100)

    # Plotting Results
    plt.figure(figsize=(10, 6))
    for model_type in models:
        plt.plot([r * 100 for r in attack_rates], results[model_type], marker='o', linewidth=2, label=f'{model_type.upper()}')

    plt.title('GNN NIDS Robustness under Structural Edge Perturbations', fontsize=14)
    plt.xlabel('Edge Drop Rate (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.savefig('robustness_curve.png')
    print("[*] Robustness curve successfully generated and saved as 'robustness_curve.png'!")

if __name__ == '__main__':
    evaluate_curve()