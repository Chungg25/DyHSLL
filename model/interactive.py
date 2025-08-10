import torch
import torch.nn as nn

class InteractiveGCN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj_1 = nn.Linear(hidden_dim, hidden_dim)
        self.proj_2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        # x: (B, N, D), adj: (N, N)
        large_graph_feat_1 = adj @ self.proj_1(x)
        large_graph_feat_2 = adj @ self.proj_2(x)
        feat_interactive = self.activation(large_graph_feat_1 * large_graph_feat_2)
        feat_full = feat_interactive + large_graph_feat_1
        y_final = self.norm(feat_full + x)
        return y_final