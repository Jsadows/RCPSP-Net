import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class RCPSPPolicy(nn.Module):
    def __init__(
        self,
        node_input_dim=6,
        global_input_dim=2,
        hidden_dim=256,
        hidden_dim_global=64,
        heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_input_dim, hidden_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim_global),
            nn.ReLU()
        )
        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.gat4 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim_global, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, global_state):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.node_encoder(x) # Do weryfikacji czy może owrapować w F.elu
        # Sprawdzić czy struktóra podobna do transformera nie zadziałą jepiej czyli po między 
        x = x + F.elu(self.gat1(x, edge_index))
        x = x + F.elu(self.gat2(x, edge_index))
        x = x + F.elu(self.gat3(x, edge_index))
        x = x + F.elu(self.gat4(x, edge_index))
        g_emb = self.global_encoder(global_state) # do weryfikacji czy potrzebne
        g_emb_expanded = g_emb[batch]
        combined = torch.cat([x, g_emb_expanded], dim=-1)
        logits = self.policy(combined).squeeze(-1)
        return logits