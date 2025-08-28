import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import norm_adj
from backbone import GNNLayer

class TemporalEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.tod_emb = nn.Embedding(288, hidden_dim)
        self.dow_emb = nn.Embedding(7, hidden_dim)
    def forward(self, tod_idx, dow_idx):
        # tod_idx: B x T, dow_idx: B x T
        tod = self.tod_emb(tod_idx)  # B x T x D
        dow = self.dow_emb(dow_idx)  # B x T x D
        return tod, dow

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.WQ = nn.Linear(hidden_dim, hidden_dim)
        self.WK = nn.Linear(hidden_dim, hidden_dim)
        self.WV = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, X):  # X: B x T x N x D
        B, T, N, D = X.shape
        X_ = X.reshape(B*T, N, D)
        Q = self.WQ(X_)
        K = self.WK(X_)
        V = self.WV(X_)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        Z = torch.matmul(attn, V)
        Z = Z.reshape(B, T, N, D)
        return Z

class HyperedgeWeightOptimizer(nn.Module):
    def __init__(self, num_edges):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_edges) / num_edges, requires_grad=True)
    def forward(self):
        w = F.softmax(self.weights, dim=0)
        return w

class VertexWeightOptimizer(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_nodes) / num_nodes, requires_grad=True)
    def forward(self):
        u = F.softmax(self.weights, dim=0)
        return u

class DynamicHypergraphStructure(nn.Module):
    def __init__(self, num_nodes, num_edges, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.edge_proj = nn.Linear(emb_dim, num_edges)
        self.node_proj = nn.Linear(emb_dim, num_nodes)
    def forward(self, node_emb):
        if node_emb.dim() == 2:
            node_feat = torch.relu(self.fc1(node_emb))
            edge_feat = torch.relu(self.fc2(node_emb))
        else:
            node_feat = torch.relu(self.fc1(node_emb))
            edge_feat = torch.relu(self.fc2(node_emb))
        H_node = torch.sigmoid(self.node_proj(node_feat))
        H_edge = torch.sigmoid(self.edge_proj(edge_feat))
        H_proj = (H_node + H_edge) / 2
        H_proj = torch.clamp(H_proj, 0, 1)
        return H_proj

class HypergraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, H, W=None):
        super().__init__()
        self.H = H
        self.W = W if W is not None else torch.ones(H.shape[1], device=H.device)
        self.theta = nn.Parameter(torch.randn(in_dim, out_dim) / math.sqrt(in_dim))
    def forward(self, X):
        H = self.H.to(X.device)
        W = self.W.to(X.device)
        N = H.shape[0]
        E = H.shape[1]
        eps = 1e-6
        Dv_inv_sqrt = torch.diag(torch.pow(torch.sum(H, dim=1) + eps, -0.5)).to(X.device)
        De_inv = torch.diag(torch.pow(torch.sum(H, dim=0) + eps, -1)).to(X.device)
        W_diag = torch.diag(W).to(X.device)
        X = torch.matmul(X, self.theta.to(X.device))
        X = torch.einsum('ij,bjd->bid', Dv_inv_sqrt, X)
        X = torch.einsum('en,bnd->bed', H.t(), X)
        X = torch.einsum('ee,bed->bed', W_diag, X)
        X = torch.einsum('ee,bed->bed', De_inv, X)
        X = torch.einsum('ne,bed->bnd', H, X)
        X = torch.einsum('ij,bjd->bid', Dv_inv_sqrt, X)
        return X

class SelfAdaptiveFusion(nn.Module):
    def __init__(self, in_dim, num_features):
        super().__init__()
        self.fc1 = nn.Linear(in_dim * num_features, in_dim)
        self.fc2 = nn.Linear(in_dim, num_features)
    def forward(self, features):  # features: list of tensors [B x N x D]
        concat = torch.cat(features, dim=-1)
        weights = F.softmax(self.fc2(F.relu(self.fc1(concat))), dim=-1)
        fused = sum(f * weights[..., i:i+1] for i, f in enumerate(features))
        return fused

class FullModel(nn.Module):
    def __init__(self, args, k_nearest=8):
        super().__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.seq_out_len = args.seq_out_len
        self.num_nodes = args.adj_mx.shape[0]
        self.num_hyper_edge = args.num_hyper_edge

        self.temporal_emb = TemporalEmbedding(self.hidden_dim)
        self.node_embedding = nn.Parameter(torch.empty(self.num_nodes, self.hidden_dim))
        nn.init.xavier_uniform_(self.node_embedding)
        self.input_embedding = nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim), nn.ReLU())
        self.temporal_attention = TemporalSelfAttention(self.hidden_dim)

        # GAH construction
        self.H_adj = self.build_gah_incidence(args.adj_mx, k=k_nearest)
        self.edge_weight_optimizer = HyperedgeWeightOptimizer(self.H_adj.shape[1])
        self.vertex_weight_optimizer = VertexWeightOptimizer(self.num_nodes)
        self.dynamic_H = DynamicHypergraphStructure(self.num_nodes, self.num_hyper_edge, self.in_dim)
        self.gahcn = HypergraphConvolution(self.hidden_dim, self.hidden_dim, self.H_adj)
        self.fshcn = HypergraphConvolution(self.hidden_dim, self.hidden_dim, torch.rand(self.num_nodes, self.num_hyper_edge))
        self.leaky_relu = nn.LeakyReLU()
        self.fusion = SelfAdaptiveFusion(self.hidden_dim, 2)

        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 5 * 12, self.hidden_dim),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim * self.seq_out_len)
        )

    def build_gah_incidence(self, adj_mx, k=8):
        N = adj_mx.shape[0]
        E = N
        H = torch.zeros(N, E, device=adj_mx.device)
        for v in range(N):
            neighbors = torch.topk(adj_mx[v], k=k, largest=True).indices
            H[v, v] = 1
            H[neighbors, v] = 1
        return H

    def forward(self, data):
        feat = data['feat']  # B x T x N x Din
        tod_idx = data['tod_idx']  # B x T
        dow_idx = data['dow_onehot']  # B x T
        node_idx = torch.arange(0, self.num_nodes).to(feat.device)  # N

        input_emb = self.input_embedding(feat)  # B x T x N x D
        tod_emb, dow_emb = self.temporal_emb(tod_idx, dow_idx)  # B x T x D, B x T x D
        node_emb = self.node_embedding[node_idx].unsqueeze(0).unsqueeze(0)  # 1 x 1 x N x D

        feature = input_emb + tod_emb.unsqueeze(2) + dow_emb.unsqueeze(2) + node_emb  # B x T x N x D
        feature = self.temporal_attention(feature)

        feature_last = feature[:, -1, :, :self.in_dim]  # B x N x D
        node_emb_dyn = feature_last

        # Dynamic hypergraph
        H_proj = self.dynamic_H(node_emb_dyn)  # [N, E]
        edge_weights = self.edge_weight_optimizer()  # [E]
        vertex_weights = self.vertex_weight_optimizer()  # [N]

        # GAHCN
        gah_out = self.leaky_relu(
            HypergraphConvolution(
                self.hidden_dim,
                self.hidden_dim,
                self.H_adj,
                edge_weights
            )(feature_last * vertex_weights.unsqueeze(-1))
        )

        # FSHCN (dynamic)
        fsh_out = self.leaky_relu(
            HypergraphConvolution(
                self.hidden_dim,
                self.hidden_dim,
                H_proj,
                edge_weights
            )(feature_last * vertex_weights.unsqueeze(-1))
        )

        fused_feature = self.fusion([gah_out, fsh_out])

        future_feature = data['target'][:, :, :, -5:].transpose(1, 2).reshape(self.args.batch_size, self.num_nodes, -1)
        if self.args.feat_off == 1:
            future_feature = self.args.scaler.transform(future_feature)
        else:
            future_feature = self.args.scaler[0].transform(future_feature)

        pred = self.pred_head(torch.cat([fused_feature, future_feature], dim=-1))
        pred = pred.view(pred.size(0), self.num_nodes, self.out_dim, self.seq_out_len)
        pred = pred.permute(0, 3, 1, 2)  # B x seq_out_len x N x out_dim
        return pred