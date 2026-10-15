import torch
from src.dataset import generate_iot_graph_dataset
from src.model import IoTGNN

def evaluate_feature_perturbation(noise_level: float = 0.15):
    print(f"[*] Starting Feature-Level Perturbation Attack (Noise Level: {noise_level * 100}%)...")
    
    data = generate_iot_graph_dataset(num_nodes=300)
    
    if not hasattr(data, 'test_mask'):
        indices = torch.randperm(data.num_nodes)
        train_size = int(data.num_nodes * 0.7)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[indices[:train_size]] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[indices[train_size:]] = True

    # Train base GraphSAGE model
    model = IoTGNN(data.num_features, 64, 2, 'sage')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    
    # 1. Clean Evaluation
    with torch.no_grad():
        out_clean = model(data.x, data.edge_index)
        pred_clean = out_clean.argmax(dim=-1)
        acc_clean = int((pred_clean[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())

    # 2. Perturbed Feature Evaluation (Adding Gaussian Noise)
    noise = torch.randn_like(data.x) * noise_level
    x_perturbed = data.x + noise
    
    with torch.no_grad():
        out_pert = model(x_perturbed, data.edge_index)
        pred_pert = out_pert.argmax(dim=-1)
        acc_pert = int((pred_pert[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())

    print("-" * 50)
    print(f"[*] Feature Perturbation Results (GraphSAGE):")
    print(f"    - Clean Accuracy    : {acc_clean * 100:.2f}%")
    print(f"    - Perturbed Accuracy: {acc_pert * 100:.2f}% (Under {noise_level*100}% Feature Noise)")
    print("-" * 50)

if __name__ == '__main__':
    evaluate_feature_perturbation(noise_level=0.15)