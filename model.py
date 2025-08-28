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
        self.dow_emb = nn.Linear(7, hidden_dim)
    def forward(self, tod_idx, dow_onehot):
        tod = self.tod_emb(tod_idx)  # B x T x D
        dow = self.dow_emb(dow_onehot)  # B x T x D
        return tod, dow

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.WQ = nn.Linear(hidden_dim, hidden_dim)
        self.WK = nn.Linear(hidden_dim, hidden_dim)
        self.WV = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, X):  # B x T x N x D
        B, T, N, D = X.shape
        X_ = X.permute(0,2,1,3).reshape(B*N, T, D)  # (B*N) x T x D
        Q = self.WQ(X_)
        K = self.WK(X_)
        V = self.WV(X_)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        Z = torch.matmul(attn, V)
        Z = Z.reshape(B, N, T, D).permute(0,2,1,3)  # B x T x N x D
        return Z

def build_gah(adj, k):
    N = adj.shape[0]
    H = torch.zeros(N, N, device=adj.device)
    for v in range(N):
        neighbors = torch.topk(adj[v], k=k, largest=True).indices
        H[v, v] = 1
        H[neighbors, v] = 1
    return H

def build_fsh(feature, num_edges):
    # feature: B x N x D
    B, N, D = feature.shape
    H = torch.zeros(B, N, num_edges, device=feature.device)
    for b in range(B):
        sim = torch.matmul(feature[b], feature[b].T)  # N x N
        for v in range(N):
            neighbors = torch.topk(sim[v], k=num_edges, largest=True).indices
            H[b, v, :] = 0
            H[b, v, :] = (neighbors == torch.arange(num_edges, device=feature.device)).float()
    return H  # B x N x num_edges

class HypergraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(in_dim, out_dim) / math.sqrt(in_dim))
    def forward(self, X, H, W=None):
        # X: B x N x D, H: B x N x E or N x E, W: E
        if H.dim() == 2:
            H = H.unsqueeze(0).expand(X.size(0), -1, -1)
        B, N, E = H.shape
        D = X.size(-1)
        eps = 1e-6
        Dv_inv_sqrt = torch.diag_embed(torch.pow(torch.sum(H, dim=2) + eps, -0.5))  # B x N x N
        De_inv = torch.diag_embed(torch.pow(torch.sum(H, dim=1) + eps, -1))         # B x E x E
        if W is None:
            W = torch.ones(E, device=X.device)
        W_diag = torch.diag(W).unsqueeze(0).expand(B, -1, -1)                       # B x E x E
        X = torch.matmul(X, self.theta)
        X = torch.matmul(Dv_inv_sqrt, X)
        X = torch.matmul(H.transpose(1,2), X)
        X = torch.matmul(W_diag, X)
        X = torch.matmul(De_inv, X)
        X = torch.matmul(H, X)
        X = torch.matmul(Dv_inv_sqrt, X)
        return X

class SelfAdaptiveFusion(nn.Module):
    def __init__(self, in_dim, num_features):
        super().__init__()
        self.fc1 = nn.Linear(in_dim * num_features, in_dim)
        self.fc2 = nn.Linear(in_dim, num_features)
    def forward(self, features):  # list of [B x N x D]
        concat = torch.cat(features, dim=-1)
        weights = F.softmax(self.fc2(F.relu(self.fc1(concat))), dim=-1)
        fused = sum(f * weights[..., i:i+1] for i, f in enumerate(features))
        return fused

class FullModel(nn.Module):
    def __init__(self, args, k_nearest=8, num_fsh_edges=8):
        super().__init__()
        self.args = args
        self.time_embedding = TemporalEmbedding(args.hidden_dim)
        self.node_embedding = nn.Embedding(args.num_nodes, args.hidden_dim)
        self.input_embedding = nn.Sequential(nn.Linear(args.in_dim, args.hidden_dim), nn.ReLU())
        self.temporal_attention = TemporalSelfAttention(args.hidden_dim)
        self.gah_conv = HypergraphConvolution(args.hidden_dim, args.hidden_dim)
        self.fsh_conv = HypergraphConvolution(args.hidden_dim, args.hidden_dim)
        self.fusion = SelfAdaptiveFusion(args.hidden_dim, 2)
        self.pred_head = nn.Sequential(
            nn.Linear(args.hidden_dim + 5 * 12, args.hidden_dim),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.out_dim * args.seq_out_len)
        )
        self.k_nearest = k_nearest
        self.num_fsh_edges = num_fsh_edges

    def forward(self, data):
        feat = data['feat']  # B x T x N x Din
        tod_idx = data['tod_idx']  # B x T
        dow_onehot = data['dow_onehot']  # B x T x 7
        node_idx = torch.arange(0, self.args.num_nodes).to(feat.device)  # N

        input_emb = self.input_embedding(feat)  # B x T x N x D
        tod_emb, dow_emb = self.time_embedding(tod_idx, dow_onehot)  # B x T x D, B x T x D
        node_emb = self.node_embedding(node_idx).unsqueeze(0).unsqueeze(0)  # 1 x 1 x N x D

        feature = input_emb + tod_emb.unsqueeze(2) + dow_emb.unsqueeze(2) + node_emb  # B x T x N x D
        feature = self.temporal_attention(feature)  # B x T x N x D

        # Dynamic: mỗi time step
        outputs = []
        gah_H = build_gah(self.args.predefined_adj, self.k_nearest)  # N x N
        print(feature.size(1))
        for t in range(feature.size(1)):
            print(t)
            x_t = feature[:, t, :, :]  # B x N x D
            fsh_H = build_fsh(x_t, self.num_fsh_edges)  # B x N x E
            gah_out = self.gah_conv(x_t, gah_H)
            fsh_out = self.fsh_conv(x_t, fsh_H)
            fused = self.fusion([gah_out, fsh_out])  # B x N x D
            outputs.append(fused.unsqueeze(1))
        fused_feature = torch.cat(outputs, dim=1)  # B x T x N x D

        # Lấy đặc trưng cuối cùng cho từng node
        final_feature = fused_feature[:, -1, :, :]  # B x N x D

        future_feature = data['target'][:, :, :, -5:].transpose(1, 2).reshape(self.args.batch_size, self.args.num_nodes, -1)
        if self.args.feat_off == 1:
            future_feature = self.args.scaler.transform(future_feature)
        else:
            future_feature = self.args.scaler[0].transform(future_feature)

        pred = self.pred_head(torch.cat([final_feature, future_feature], dim=-1))
        pred = pred.view(pred.size(0), self.args.num_nodes, self.args.out_dim, self.args.seq_out_len)
        pred = pred.permute(0, 3, 1, 2)  # B x seq_out_len x N x out_dim
        return pred