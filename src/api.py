from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from .model import RobustGNN_NIDS
from .dataset import generate_iot_graph_dataset

# FastAPI tətbiqini başladırıq
app = FastAPI(
    title="Robust GNN-NIDS API",
    description="Graph Neural Network based Intrusion Detection System for IoT Security",
    version="1.0.0"
)

# Dataset və Model instansiyasını hazırlayırıq
data = generate_iot_graph_dataset(num_nodes=50, num_features=16)
model = RobustGNN_NIDS(num_features=16, hidden_dim=32, num_classes=2)
model.eval()

# Sorğu üçün Pydantic modeli
class PredictionRequest(BaseModel):
    node_index: int

@app.get("/")
def read_root():
    return {
        "status": "Active",
        "message": "Robust GNN-NIDS API is running successfully.",
        "domain": "IoT Cybersecurity Research"
    }

@app.post("/predict")
def predict_anomaly(request: PredictionRequest):
    # Nöqtə indeksinin sərhədlər daxilində olmasını yoxlayırıq
    if request.node_index < 0 or request.node_index >= data.num_nodes:
        raise HTTPException(
            status_code=400, 
            detail=f"Node index out of bounds. Must be between 0 and {data.num_nodes - 1}."
        )
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probabilities = torch.exp(out[request.node_index])
        predicted_class = out[request.node_index].argmax().item()
        confidence = probabilities[predicted_class].item()

    label_mapping = {0: "Normal Traffic", 1: "Attack / Zero-day Anomaly"}
    
    return {
        "node_index": request.node_index,
        "prediction_class": predicted_class,
        "prediction_label": label_mapping[predicted_class],
        "confidence_score": round(confidence, 4)
    }