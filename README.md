# Robust GNN-Based Intrusion Detection System (NIDS) for IoT Networks

An end-to-end Graph Neural Network (GNN) framework designed for Network Intrusion Detection Systems (NIDS) in IoT environments, evaluating adversarial robustness against structural perturbations.

## 🌟 Key Features
- **Multi-Architecture Support:** Implementation and benchmark of GCN, GAT, and GraphSAGE models.
- **Dynamic Dataset Generation:** Synthetic IoT graph dataset generation with features and topology.
- **Adversarial Robustness Evaluation:** Structural attack simulation using random edge perturbation (`dropout_edge`).

---

## 📊 Experimental Results

### 1. Baseline Model Performance
Evaluation of GNN architectures on the synthetic IoT network dataset:

| Architecture | Clean Accuracy (%) |
|--------------|-------------------|
| **GraphSAGE** | **85.56%** |
| **GCN**       | **78.89%** |
| **GAT**       | **77.78%** |

### 2. Adversarial Robustness Benchmark (30% Edge Perturbation)
Evaluation under structural attack (30% random edge drop rate):

| Model Architecture | Clean Accuracy (%) | Attacked Accuracy (%) | Robustness Delta |
|--------------------|--------------------|-----------------------|------------------|
| **GraphSAGE**      | 81.11%             | **82.22%**            | +1.11%           |
| **GCN**            | 76.67%             | **78.89%**            | +2.22%           |
| **GAT**            | 80.00%             | **77.78%**            | -2.22%           |

---

## 🛠️ Installation & Usage

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt