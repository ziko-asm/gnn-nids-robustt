import torch
from src.dataset import generate_iot_graph_dataset
from src.model import RobustGNN_NIDS

def test_dataset_generation():
    data = generate_iot_graph_dataset(num_nodes=50, num_features=16)
    assert data.num_nodes == 50
    assert data.x.shape == (50, 16)
    assert data.edge_index.shape[0] == 2
    assert data.y.shape == (50,)

def test_model_forward_pass():
    num_nodes = 50
    num_features = 16
    hidden_dim = 32
    num_classes = 2

    data = generate_iot_graph_dataset(num_nodes=num_nodes, num_features=num_features)
    model = RobustGNN_NIDS(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes)
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)

    # Output shape should be [num_nodes, num_classes]
    assert out.shape == (num_nodes, num_classes)