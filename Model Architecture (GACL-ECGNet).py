import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GACL_ECGNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.mlp_margin = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        embeddings = x
        logits = self.classifier(embeddings)
        return logits, embeddings
    
    def adaptive_contrastive_loss(self, embeddings, y, margin_base=1.0):
        # Stress-aware margin adjustment (simplified)
        margins = self.mlp_margin(embeddings).squeeze()
        margins = margin_base + F.sigmoid(margins)
        
        # Pairwise contrastive loss
        pairwise_dist = F.pairwise_distance(embeddings.unsqueeze(1), embeddings.unsqueeze(0), p=2)
        mask = (y.unsqueeze(1) == y.unsqueeze(0)).float()
        loss_contrastive = torch.mean(mask * pairwise_dist.pow(2) + 
                           (1 - mask) * (margins - pairwise_dist).clamp(min=0).pow(2))
        return loss_contrastive
