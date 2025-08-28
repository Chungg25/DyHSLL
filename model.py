import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import norm_adj
from backbone import GNNLayer

# --- Hyperedge Weight Optimization ---
class HyperedgeWeightOptimizer(nn.Module):
    def __init__(self, num_edges):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_edges) / num_edges, requires_grad=True)
    def forward(self):
        w = F.softmax(self.weights, dim=0)
        return w

# --- Vertex Weight Optimization ---
class VertexWeightOptimizer(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_nodes) / num_nodes, requires_grad=True)
    def forward(self):
        u = F.softmax(self.weights, dim=0)
        return u

# --- Dynamic Hypergraph Structure Optimization ---
class DynamicHypergraphStructure(nn.Module):
    def __init__(self, num_nodes, num_edges, emb_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.emb_dim = emb_dim
        # Các tham số học để sinh đặc trưng node cho hypergraph
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.edge_proj = nn.Linear(emb_dim, num_edges)
        self.node_proj = nn.Linear(emb_dim, num_nodes)

    def forward(self, node_emb):
        # node_emb: [N, D] hoặc [B, N, D]
        if node_emb.dim() == 2:
            node_feat = torch.relu(self.fc1(node_emb))  # [N, D]
            edge_feat = torch.relu(self.fc2(node_emb))  # [N, D]
        else:
            node_feat = torch.relu(self.fc1(node_emb))  # [B, N, D]
            edge_feat = torch.relu(self.fc2(node_emb))  # [B, N, D]

        # Sinh ma trận hypergraph incidence động
        H_node = torch.sigmoid(self.node_proj(node_feat))  # [N, N] hoặc [B, N, N]
        H_edge = torch.sigmoid(self.edge_proj(edge_feat))  # [N, E] hoặc [B, N, E]

        # Kết hợp hai đặc trưng để tạo H động
        H_proj = (H_node + H_edge) / 2  # [N, E] hoặc [B, N, E]
        # Chuẩn hóa để đảm bảo giá trị hợp lệ
        H_proj = torch.clamp(H_proj, 0, 1)
        return H_proj

# --- Temporal Self-attention (Transformer) ---
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

# --- DTW clustering (pseudo, replace with actual DTW) ---
def dtw_clustering(time_series, num_clusters):
    N = time_series.shape[0]
    H_sim = torch.zeros(N, num_clusters, device=time_series.device)
    cluster_idx = torch.randint(0, num_clusters, (N,), device=time_series.device)
    for i in range(N):
        H_sim[i, cluster_idx[i]] = 1
    return H_sim

class HypergraphLearning(nn.Module):
    def __init__(self, args, num_edges):
        super(HypergraphLearning, self).__init__()
        self.args = args
        self.num_edges = num_edges
        self.edge_clf = torch.randn(args.hidden_dim, self.num_edges) / math.sqrt(self.num_edges)
        self.edge_clf = nn.Parameter(self.edge_clf, requires_grad=True)
        self.edge_map = torch.randn(self.num_edges, self.num_edges) / math.sqrt(self.num_edges)
        self.edge_map = nn.Parameter(self.edge_map, requires_grad=True)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(args.hidden_dim)

    def forward(self, x):  # B x T x N x D
        feat = x.reshape(x.size(0), -1, x.size(3))
        hyper_assignment = torch.softmax(feat @ self.edge_clf, dim=-1)
        hyper_feat = hyper_assignment.transpose(1, 2) @ feat
        hyper_feat_mapped = self.activation(self.edge_map @ hyper_feat)
        hyper_out = hyper_feat_mapped + hyper_feat
        y = self.activation(hyper_assignment @ hyper_out)
        y = y.reshape(x.size(0), x.size(1), x.size(2), x.size(3))
        y_final = self.norm(y + x)
        return y_final

# --- Hypergraph Convolution Layer ---
class HypergraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, H, W=None):
        super().__init__()
        self.H = H  # N x E
        self.W = W if W is not None else torch.ones(H.shape[1], device=H.device)
        self.theta = nn.Parameter(torch.randn(in_dim, out_dim) / math.sqrt(in_dim))
    def forward(self, X):
        # X: B x N x F
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

# --- Self-adaptive Fusion ---
class SelfAdaptiveFusion(nn.Module):
    def __init__(self, in_dim, num_features):
        super().__init__()
        self.fc1 = nn.Linear(in_dim * num_features, in_dim)
        self.fc2 = nn.Linear(in_dim, num_features)
    def forward(self, features):  # features: list of tensors [B x N x D]
        concat = torch.cat(features, dim=-1)  # B x N x D*num_features
        weights = F.softmax(self.fc2(F.relu(self.fc1(concat))), dim=-1)  # B x N x num_features
        fused = sum(f * weights[..., i:i+1] for i, f in enumerate(features))  # weighted sum
        return fused

class ScaleAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_scales):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_scales = num_scales

    def forward(self, scale_features):  # list of [B x N x D], len=num_scales
        # Stack: [B, num_scales, N, D] -> [B*N, num_scales, D]
        B, N, D = scale_features[0].shape
        x = torch.stack(scale_features, dim=1)  # [B, num_scales, N, D]
        x = x.permute(0, 2, 1, 3).reshape(B*N, self.num_scales, D)  # [B*N, num_scales, D]
        attn_out, _ = self.attn(x, x, x)  # [B*N, num_scales, D]
        attn_out = self.norm(attn_out)
        # Lấy đặc trưng tổng hợp cho mỗi node: mean hoặc lấy scale cuối
        fused = attn_out.mean(dim=1).reshape(B, N, D)  # [B, N, D]
        return fused

class TemporalPooling(nn.Module):
    def __init__(self, mode='mean', ratio=2):
        super().__init__()
        self.mode = mode
        self.ratio = ratio

    def forward(self, x):
        B, T, N, D = x.shape
        new_len = T // self.ratio
        x = x[:, :new_len * self.ratio].reshape(B, new_len, self.ratio, N, D)
        if self.mode == 'max':
            y = x.max(dim=2)[0]
        else:
            y = x.mean(dim=2)
        return y

# --- Main Model ---
class FullModel(nn.Module):
    def __init__(self, args, num_dtw_clusters=32):
        super().__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.adj_mx = args.adj_mx  # N x N
        self.num_nodes = args.adj_mx.shape[0]
        self.hidden_dim = args.hidden_dim

        self.time_of_day_emb = nn.Parameter(torch.empty(48, self.hidden_dim))
        self.day_of_week_emb = nn.Parameter(torch.empty(7, self.hidden_dim))
        nn.init.xavier_uniform_(self.time_of_day_emb)
        nn.init.xavier_uniform_(self.day_of_week_emb)
        
        self.node_embedding = nn.Parameter(torch.empty(self.num_nodes, self.hidden_dim))
        nn.init.xavier_uniform_(self.node_embedding)

        self.input_embedding = nn.Sequential(nn.Linear(args.in_dim, args.hidden_dim), nn.ReLU())
        self.temporal_attention = TemporalSelfAttention(self.hidden_dim)
        # self.H_adj = self.build_gah_incidence(args.adj_mx, k=args.k_nearest)
        # self.edge_weight_optimizer = HyperedgeWeightOptimizer(self.H_adj.shape[1])
        # self.vertex_weight_optimizer = VertexWeightOptimizer(args.num_nodes)
        # self.dynamic_H = DynamicHypergraphStructure(args.num_nodes, self.H_adj.shape[1])
        # self.gahcn = HypergraphConvolution(args.hidden_dim, args.hidden_dim, self.H_adj)
        # self.fshcn = HypergraphLearning(args, args.num_hyper_edge)
        # self.leaky_relu = nn.LeakyReLU()
        # self.fusion = SelfAdaptiveFusion(args.hidden_dim, 2)

        # Dynamic hypergraph modules
        self.dynamic_H = DynamicHypergraphStructure(self.num_nodes, args.num_hyper_edge, self.hidden_dim)
        self.edge_weight_optimizer = HyperedgeWeightOptimizer(args.num_hyper_edge)
        self.vertex_weight_optimizer = VertexWeightOptimizer(self.num_nodes)
        self.gahcn = HypergraphConvolution(self.hidden_dim, self.hidden_dim, torch.rand(self.num_nodes, args.num_hyper_edge))
        self.fshcn = HypergraphLearning(args, args.num_hyper_edge)
        self.leaky_relu = nn.LeakyReLU()
        self.fusion = SelfAdaptiveFusion(self.hidden_dim, 2)

        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 5 * 12, self.hidden_dim),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, args.out_dim * args.seq_out_len)
        )

    def build_gah_incidence(self, adj_mx, k=32):
        N = adj_mx.shape[0]
        E = N
        H = torch.zeros(N, E, device=adj_mx.device)
        for v in range(N):
            neighbors = torch.topk(-adj_mx[v], k=k, largest=True).indices
            H[v, v] = 1
            H[neighbors, v] = 1
        return H

    def forward(self, data):
        feat = data['feat']  # B x T x N x Din
        tod_idx = data['tod_idx']  # B x T
        dow_onehot = data['dow_onehot']  # B x T x 7
        node_idx = torch.arange(0, self.num_nodes).to(feat.device)  # N

        input_emb = self.input_embedding(feat)  # B x T x N x D
        time_emb = self.time_of_day_emb[tod_idx].unsqueeze(2)  # B x T x 1 x D
        date_emb = torch.matmul(dow_onehot, self.day_of_week_emb).unsqueeze(2)  # B x T x 1 x D
        node_emb = self.node_embedding[node_idx].unsqueeze(0).unsqueeze(0)  # 1 x 1 x N x D

        feature = input_emb + time_emb + date_emb + node_emb  # B x T x N x D
        feature = self.temporal_attention(feature)

        # Lấy frame cuối cùng (hoặc bạn có thể dùng toàn bộ chuỗi)
        feature_last = feature[:, -1, :, :self.in_dim]  # B x N x D
        node_emb_dyn = feature_last

        # Dynamic hypergraph xây dựng lại mỗi forward
        H_proj = self.dynamic_H(node_emb_dyn)  # [N, E]
        edge_weights = self.edge_weight_optimizer()  # [E]
        vertex_weights = self.vertex_weight_optimizer()  # [N]

        if H_proj.dim() == 3:
            gah_out_list = []
            for b in range(H_proj.shape[0]):
                gah_out_b = self.leaky_relu(
                    HypergraphConvolution(
                        self.hidden_dim,
                        self.hidden_dim,
                        H_proj[b],
                        edge_weights
                    )(feature_last[b] * vertex_weights.unsqueeze(-1))
                )
                gah_out_list.append(gah_out_b)
            gah_out = torch.stack(gah_out_list, dim=0)  # [B, N, D]
        else:
            gah_out = self.leaky_relu(
                HypergraphConvolution(
                    self.hidden_dim,
                    self.hidden_dim,
                    H_proj,
                    edge_weights
                )(feature_last * vertex_weights.unsqueeze(-1))
            )

        fsh_out = self.fshcn(feature)
        fsh_out_last = fsh_out[:, -1, :, :]
        fused_feature = self.fusion([gah_out, fsh_out_last])

        future_feature = data['target'][:, :, :, -5:].transpose(1, 2).reshape(self.args.batch_size, self.num_nodes, -1)
        if self.args.feat_off == 1:
            future_feature = self.args.scaler.transform(future_feature)
        else:
            future_feature = self.args.scaler[0].transform(future_feature)

        pred = self.pred_head(torch.cat([fused_feature, future_feature], dim=-1))
        pred = pred.view(pred.size(0), self.num_nodes, self.args.out_dim, self.args.seq_out_len)
        pred = pred.permute(0, 3, 1, 2)
        return pred