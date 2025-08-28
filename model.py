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
    def __init__(self, num_nodes, num_edges):
        super().__init__()
        self.H = nn.Parameter(torch.rand(num_nodes, num_edges), requires_grad=True)
    def forward(self):
        H_proj = torch.clamp(self.H, 0, 1)
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
        Dv_inv_sqrt = torch.diag(torch.pow(torch.sum(H, dim=1), -0.5)).to(X.device)  # [N, N]
        De_inv = torch.diag(torch.pow(torch.sum(H, dim=0), -1)).to(X.device)         # [E, E]
        W_diag = torch.diag(W).to(X.device)                                          # [E, E]
        X = torch.matmul(X, self.theta.to(X.device))                                 # [B, N, out_dim]

        # Step 1: Dv^{-1/2} X
        X = torch.einsum('ij,bjd->bid', Dv_inv_sqrt, X)                              # [B, N, out_dim]
        # Step 2: H^T X
        X = torch.einsum('en,bnd->bed', H.t(), X)                                   # [B, E, out_dim]
        # Step 3: W X
        X = torch.einsum('ee,bed->bed', W_diag, X)                                  # [B, E, out_dim]
        # Step 4: De^{-1} X
        X = torch.einsum('ee,bed->bed', De_inv, X)                                  # [B, E, out_dim]
        # Step 5: H X
        X = torch.einsum('ne,bed->bnd', H, X)                                       # [B, N, out_dim]
        # Step 6: Dv^{-1/2} X
        X = torch.einsum('ij,bjd->bid', Dv_inv_sqrt, X)                             # [B, N, out_dim]
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

        self.poolings = nn.ModuleList([
            TemporalPooling('mean', ratio) for ratio in [12, 8, 6, 4, 3, 2, 1]
        ])

        self.num_scales = len(self.poolings)

        self.H_adj_list = []
        self.edge_weight_optimizer_list = nn.ModuleList()
        self.vertex_weight_optimizer_list = nn.ModuleList()
        self.dynamic_H_list = nn.ModuleList()
        self.gahcn_list = nn.ModuleList()
        self.fshcn_list = nn.ModuleList()


        for _ in range(self.num_scales):
            H_adj = self.build_gah_incidence(args.adj_mx, k=args.k_nearest)
            self.H_adj_list.append(H_adj)
            self.edge_weight_optimizer_list.append(HyperedgeWeightOptimizer(H_adj.shape[1]))
            self.vertex_weight_optimizer_list.append(VertexWeightOptimizer(args.num_nodes))
            self.dynamic_H_list.append(DynamicHypergraphStructure(args.num_nodes, H_adj.shape[1]))
            self.gahcn_list.append(HypergraphConvolution(args.hidden_dim, args.hidden_dim, H_adj))
            
            self.dynamic_H_list.append(DynamicHypergraphStructure(self.num_nodes, H_adj.shape[1]))
            self.fshcn_list.append(HypergraphLearning(args, args.num_hyper_edge))

        self.leaky_relu = nn.LeakyReLU()
        self.fusion = SelfAdaptiveFusion(self.hidden_dim, 2)
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales, requires_grad=True)
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
        node_emb = self.node_embedding[node_idx].unsqueeze(0).unsqueeze(0)

        feature = input_emb + time_emb + date_emb + node_emb  # B x T x N x D
        feature = self.temporal_attention(feature)

        multi_scale_features = []
        for i, pooling in enumerate(self.poolings):
            pooled = pooling(feature)  # B x T' x N x D
            feature_last = pooled[:, -1, :, :]  # B x N x D

            H_gah = self.H_adj_list[i]
            edge_weights_gah = self.edge_weight_optimizer_list[i]()
            vertex_weights_gah = self.vertex_weight_optimizer_list[i]()
            gah_out = self.leaky_relu(
                HypergraphConvolution(
                    self.hidden_dim,
                    self.hidden_dim,
                    H_gah,
                    edge_weights_gah
                )(feature_last * vertex_weights_gah.unsqueeze(-1))
            )

            H_fsh = self.dynamic_H_list[i]()
            edge_weights_fsh = self.edge_weight_optimizer_list[i]()
            vertex_weights_fsh = self.vertex_weight_optimizer_list[i]()
            fsh_out = self.leaky_relu(
                HypergraphConvolution(
                    self.hidden_dim,
                    self.hidden_dim,
                    H_fsh,
                    edge_weights_fsh
                )(feature_last * vertex_weights_fsh.unsqueeze(-1))
            )
            fused = self.fusion([gah_out, fsh_out])  # B x N x D
            multi_scale_features.append(fused)
        # Fusion all scales
        # fused_feature = self.global_fusion_layer(torch.cat(multi_scale_features, dim=-1))  # B x N x D

        # fused_feature = self.scale_fusion(multi_scale_features)

        # scale_weights = F.softmax(self.scale_weights, dim=0)  # [num_scales]
        scale_weights = F.softmax(self.scale_weights, dim=0)  # [num_scales]
        fused_feature = sum(w * f for w, f in zip(scale_weights, multi_scale_features))

        future_feature = data['target'][:, :, :, -5:].transpose(1, 2).reshape(self.args.batch_size, self.num_nodes, -1)
        if self.args.feat_off == 1:
            future_feature = self.args.scaler.transform(future_feature)
        else:
            future_feature = self.args.scaler[0].transform(future_feature)

        pred = self.pred_head(torch.cat([fused_feature, future_feature], dim=-1))  # B x N x (out_dim * T)
        pred = pred.view(pred.size(0), self.num_nodes, self.args.out_dim, self.args.seq_out_len)  # B x N x out_dim x T
        pred = pred.permute(0, 3, 1, 2)  # B x T x N x out_dim
        return pred

        # Temporal Self-attention
        # feature = self.temporal_attention(feature)  # B x T x N x D

        # # Prepare for hypergraph convolution: B x N x D
        # feature_last = feature[:, -1, :, :]  # lấy frame cuối cùng

        # # GAHCN
        # gah_out = self.leaky_relu(self.gahcn(feature_last))  # B x N x D

        # # FSHCN (HypergraphLearning)
        # fsh_out = self.fshcn(feature)  # B x T x N x D
        # fsh_out_last = fsh_out[:, -1, :, :]  # lấy frame cuối cùng, B x N x D

        # # Fusion
        # fused = self.fusion([gah_out, fsh_out_last])  # B x N x D

        # future_feature = data['target'][:, :, :, -5:].transpose(1, 2).reshape(self.args.batch_size, self.num_nodes, -1)
        # if self.args.feat_off == 1:
        #     future_feature = self.args.scaler.transform(future_feature)
        # else:
        #     future_feature = self.args.scaler[0].transform(future_feature)

        # pred = self.pred_head(torch.cat([fused, future_feature], dim=-1))  # B x N x (out_dim * T)
        # pred = pred.view(pred.size(0), self.num_nodes, self.args.out_dim, self.args.seq_out_len)  # B x N x out_dim x T
        # pred = pred.permute(0, 3, 1, 2)  # B x T x N x out_dim
        # return pred