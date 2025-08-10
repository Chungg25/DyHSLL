import torch
import torch.nn as nn
import math

class HypergraphLearning(nn.Module):
    def __init__(self, hidden_dim, num_edges):
        super().__init__()
        self.num_edges = num_edges
        self.edge_clf = nn.Parameter(torch.randn(hidden_dim, num_edges) / math.sqrt(num_edges))
        self.edge_map = nn.Parameter(torch.randn(num_edges, num_edges) / math.sqrt(num_edges))
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):  # B x N x D
        # x: (B, N, D)
        feat = x
        hyper_assignment = torch.softmax(feat @ self.edge_clf, dim=-1)  # (B, N, num_edges)
        hyper_feat = hyper_assignment.transpose(1, 2) @ feat  # (B, num_edges, D)
        hyper_feat_mapped = self.activation(self.edge_map @ hyper_feat)  # (B, num_edges, D)
        hyper_out = hyper_feat_mapped + hyper_feat
        y = self.activation(hyper_assignment @ hyper_out)  # (B, N, D)
        y_final = self.norm(y + x)
        return y_final