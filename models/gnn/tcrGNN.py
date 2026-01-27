import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool



class TCRgnn(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        
        self.dropout = dropout
        
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
        )
        
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        
        out = self.classifier(x)
        
        return out