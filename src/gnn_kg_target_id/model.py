import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from gnn_kg_target_id.config import GnnType

_GAT_HEADS = 4


class TargetGNN(nn.Module):
    """Two message-passing layers followed by a two-layer MLP head.

    Each node is a gene. The output is a per-gene score in (0, 1) for how likely the gene
    is a target for the disease the known-positive labels describe.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 8, dropout: float = 0.3, gnn_type: GnnType = "SAGE"):
        super().__init__()
        if gnn_type == "GCN":
            self.conv1, self.conv2 = GCNConv(in_dim, hidden_dim), GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == "SAGE":
            self.conv1, self.conv2 = SAGEConv(in_dim, hidden_dim), SAGEConv(hidden_dim, hidden_dim)
        elif gnn_type == "GAT":
            # Concatenated heads give hidden_dim output channels.
            self.conv1 = GATConv(in_dim, hidden_dim // _GAT_HEADS, heads=_GAT_HEADS)
            self.conv2 = GATConv(hidden_dim, hidden_dim // _GAT_HEADS, heads=_GAT_HEADS)
        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}")

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.dropout(F.relu(self.conv1(data.x, data.edge_index)))
        x = F.relu(self.conv2(x, data.edge_index))
        x = self.dropout(F.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)
