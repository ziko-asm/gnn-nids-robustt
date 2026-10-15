import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from src.dataset import generate_iot_graph_dataset
from src.model import IoTGNN

def train_adversarial(model_type='sage', epochs=200, p=0.3):
    """
    Implements Adversarial Training for GNN-based NIDS to enhance 
    robustness against structural edge perturbation attacks.
    """
    print(f"[*] Starting Adversarial Training for {model_type.upper()} (Defense Perturbation Rate: {p * 100}%) ...")
    data = generate_iot_graph_dataset(num_nodes=300)

    # Ensure train/test masks exist
    if not hasattr(data, 'train_mask'):
        indices = torch.randperm(data.num_nodes)
        train_size = int(data.num_nodes * 0.7)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[indices[:train_size]] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[indices[train_size:]] = True

    # Initialize model
    model = IoTGNN(
        in_channels=data.num_features,
        hidden_channels=64,
        out_channels=2,
        model_type=model_type
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Adversarial Training Loop (Defensive Training)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Perturb edges dynamically during training to build resilience
        perturbed_edge_index, _ = dropout_edge(data.edge_index, p=p)
        
        out = model(data.x, perturbed_edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{epochs}], Adversarial Loss: {loss.item():.4f}")

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        # 1. Clean Evaluation
        out_clean = model(data.x, data.edge_index)
        pred_clean = out_clean.argmax(dim=-1)
        acc_clean = int((pred_clean[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())

        # 2. Attacked Evaluation (Under 30% Edge Drop)
        test_edge_index, _ = dropout_edge(data.edge_index, p=0.3)
        out_attack = model(data.x, test_edge_index)
        pred_attack = out_attack.argmax(dim=-1)
        acc_attack = int((pred_attack[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())

    print("-" * 50)
    print(f"[*] Robust Defense Results ({model_type.upper()}):")
    print(f"    - Clean Accuracy    : {acc_clean * 100:.2f}%")
    print(f"    - Attacked Accuracy : {acc_attack * 100:.2f}% (Defended against 30% Edge Drop)")
    print("-" * 50)

if __name__ == '__main__':
    train_adversarial(model_type='sage', epochs=200, p=0.3)