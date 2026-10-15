import torch
from torch_geometric.explain import Explainer, GNNExplainer
from src.dataset import generate_iot_graph_dataset
from src.model import IoTGNN

def explain_predictions(node_idx: int = 0):
    print(f"[*] Initializing GNNExplainer for Node Index: {node_idx}...")
    
    # Generate dataset and initialize model
    data = generate_iot_graph_dataset(num_nodes=300)
    model = IoTGNN(in_channels=data.num_features, hidden_channels=64, out_channels=2, model_type='sage')
    
    # Train the model briefly
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
    model.eval()

    # Set up PyTorch Geometric Explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    # Generate explanation for the target node
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    
    print("-" * 50)
    print(f"[*] Explanation Generated for Node {node_idx}:")
    print(f"    - Important Feature Mask Shape : {explanation.node_mask.shape}")
    print(f"    - Important Edge Mask Shape    : {explanation.edge_mask.shape}")
    print(f"    - Top Feature Importance Weight: {explanation.node_mask.max().item():.4f}")
    print("-" * 50)
    
    return explanation

if __name__ == '__main__':
    explain_predictions(node_idx=5)