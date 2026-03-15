import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool

class TCRgnn(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.4):
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
            torch.nn.Linear(hidden_dim, 1),
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
    

class TCRgnnEdge(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dim, num_classes, edge_dim=1, dropout=0.4):
        super().__init__()
        
        self.dropout = dropout
        
        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=edge_dim
        )
        
        self.conv2 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=edge_dim
        )
        
        
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, 
            data.edge_index, 
            data.edge_attr, 
            data.batch
        )
        
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        
        out = self.classifier(x)
        
        return out

class TCRgnnEdgeLayers(torch.nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes=1,
        num_layers=3,
        edge_dim=1,
        dropout=0.4,
    ):
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"

        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()

        # First layer (input -> hidden)
        self.convs.append(
            GINEConv(
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                ),
                edge_dim=edge_dim,
            )
        )

        # Hidden layers (hidden -> hidden)
        for _ in range(num_layers - 1):
            self.convs.append(
                GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim),
                    ),
                    edge_dim=edge_dim,
                )
            )

        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Apply all GNN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        # Classification
        out = self.classifier(x)

        return out