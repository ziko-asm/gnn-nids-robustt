import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class IoTGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='gcn', heads=8):
        """
        IoT Şəbəkələri üçün Çox-Arxitekturalı GNN Modeli (GCN, GAT, GraphSAGE)
        """
        super(IoTGNN, self).__init__()
        self.model_type = model_type.lower()
        
        if self.model_type == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            
        elif self.model_type == 'gat':
            # GAT çoxbaşlıqlı (multi-head attention) diqqət mexanizmindən istifadə edir
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
            
        elif self.model_type == 'sage':
            self.conv2 = SAGEConv(in_channels, hidden_channels)
            self.conv1 = SAGEConv(hidden_channels, out_channels) # GraphSAGE konvolyusiyası
            # Qeyd: SAGEConv ardıcıllığı
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
            
        else:
            raise ValueError(f"Desteklenmeyen model türü: {model_type}. 'gcn', 'gat' və ya 'sage' seçin.")

    def forward(self, x, edge_index):
        """
        İrəi ötürmə (Forward pass) funksiyası
        """
        # İlk qraf konvolyusiya qatı + ReLU + Dropout (overfitting-in qarşısını almaq üçün)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # İkinci qatı tətbiq edirik
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)