import argparse
import torch
import torch.nn.functional as F
from src.dataset import generate_iot_graph_dataset
from src.model import IoTGNN

def train():
    parser = argparse.ArgumentParser(description="Train GNN Model for IoT Intrusion Detection")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "sage"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    # Generate graph dataset
    data = generate_iot_graph_dataset(num_nodes=300)

    # Ensure train/test masks exist
    if not hasattr(data, 'train_mask'):
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        train_size = int(num_nodes * 0.7)
        
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[indices[:train_size]] = True
        
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[indices[train_size:]] = True

    # Initialize model
    model = IoTGNN(
        in_channels=data.num_features,
        hidden_channels=64,
        out_channels=2,
        model_type=args.model
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[*] Selected Model Architecture: {args.model.upper()}")
    print(f"[*] Starting model training ({args.epochs} epochs)...")

    # Training loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{args.epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=-1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

    print("-" * 50)
    print(f"[*] Result [{args.model.upper()}]: Test Accuracy = {acc * 100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    train()