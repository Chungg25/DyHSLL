import math
import torch
import torch.nn as nn

class TemporalEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.tod_fc = nn.Linear(48, hidden_dim)
        self.dow_fc = nn.Linear(7, hidden_dim)

    def forward(self, tod_onehot, dow_onehot):
        # tod_onehot: B x T x 288, dow_onehot: B x T x 7
        tod_onehot = tod_onehot.float()  # ép kiểu về float
        dow_onehot = dow_onehot.float()  # ép kiểu về float
        tod_emb = self.tod_fc(tod_onehot)
        dow_emb = self.dow_fc(dow_onehot)
        return tod_emb, dow_emb

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.WQ = nn.Linear(hidden_dim, hidden_dim)
        self.WK = nn.Linear(hidden_dim, hidden_dim)
        self.WV = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: B x T x N x D
        B, T, N, D = x.shape
        x_reshape = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        Q = self.WQ(x_reshape)
        K = self.WK(x_reshape)
        V = self.WV(x_reshape)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(D), dim=-1)
        out = attn @ V
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return out

class HypergraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, num_edges):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(in_dim, out_dim) / math.sqrt(out_dim))
        self.num_edges = num_edges

    def forward(self, X, H, W):
        B, T, N, D = X.shape
        X = X.reshape(B * T, N, D)
        Dv_inv_sqrt = torch.diag(1.0 / torch.sqrt(H.sum(dim=1) + 1e-6))
        De_inv = torch.diag(1.0 / (H.sum(dim=0) + 1e-6))
        HW = H * W.unsqueeze(0)
        out = Dv_inv_sqrt @ HW @ De_inv @ H.t() @ Dv_inv_sqrt @ X @ self.theta
        out = out.reshape(B, T, N, -1)
        return out

class GeographicalAdjacencyHypergraphLearning(nn.Module):
    def __init__(self, args, num_edges, adj):
        super().__init__()
        self.gah_conv = HypergraphConvolution(args.hidden_dim, args.hidden_dim, num_edges)
        self.norm = nn.LayerNorm(args.hidden_dim)
        self.activation = nn.LeakyReLU()
        self.H, self.W = self.construct_GAH(adj, num_edges)

    def construct_GAH(self, adj, k):
        N = adj.shape[0]
        H = torch.zeros(N, N, dtype=torch.float32)
        W = torch.ones(N, dtype=torch.float32)
        for v in range(N):
            distances = adj[v].clone()
            distances[v] = float('inf')
            nearest_idx = torch.topk(-distances, k, largest=True).indices.tolist()
            H[v, v] = 1.0
            for idx in nearest_idx:
                H[idx, v] = 1.0
        return H, W

    def forward(self, x):
        y = self.gah_conv(x, self.H.to(x.device), self.W.to(x.device))
        y = self.activation(y)
        y_final = self.norm(y + x)
        return y_final

class HypergraphLearning(nn.Module):
    def __init__(self, args, num_edges):
        super().__init__()
        self.edge_clf = nn.Parameter(torch.randn(args.hidden_dim, num_edges) / math.sqrt(num_edges))
        self.edge_map = nn.Parameter(torch.randn(num_edges, num_edges) / math.sqrt(num_edges))
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(args.hidden_dim)

    def forward(self, x):
        feat = x.reshape(x.size(0), -1, x.size(3))
        hyper_assignment = torch.softmax(feat @ self.edge_clf, dim=-1)
        hyper_feat = hyper_assignment.transpose(1, 2) @ feat
        hyper_feat_mapped = self.activation(self.edge_map @ hyper_feat)
        hyper_out = hyper_feat_mapped + hyper_feat
        y = self.activation(hyper_assignment @ hyper_out)
        y = y.reshape(x.size(0), x.size(1), x.size(2), x.size(3))
        y_final = self.norm(y + x)
        return y_final

class TemporalPooling(nn.Module):
    def __init__(self, mode='mean', ratio=2):
        super().__init__()
        self.mode = mode
        self.ratio = ratio

    def forward(self, x):
        x = x.reshape(x.size(0), -1, self.ratio, x.size(2), x.size(3))
        if self.mode == 'max':
            y = x.max(dim=2)[0]
        else:
            y = x.mean(dim=2)
        return y

class MultiScaleHypergraphModule(nn.Module):
    def __init__(self, args, adj):
        super().__init__()
        scales = [12, 6, 4, 3, 2, 1]
        self.scales = nn.ModuleList([
            nn.Sequential(
                TemporalPooling(mode='mean', ratio=r),
                TemporalSelfAttention(args.hidden_dim),
                HypergraphLearning(args, args.num_hyper_edge),
                GeographicalAdjacencyHypergraphLearning(args, args.num_hyper_edge, adj)
            ) for r in scales
        ])
        self.fusion_fc1 = nn.Linear(args.hidden_dim * 2 * len(scales), args.hidden_dim)
        self.fusion_fc2 = nn.Linear(args.hidden_dim, len(scales))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        scale_features = []
        for scale in self.scales:
            # FSH
            fsh = scale[2](scale[1](scale[0](x)))
            # GAH
            gah = scale[3](scale[1](scale[0](x)))
            # Fusion FSH & GAH at each scale
            fusion_cat = torch.cat([fsh, gah], dim=-1)
            fusion_score = nn.Linear(fusion_cat.shape[-1], 2).to(x.device)(fusion_cat)
            fusion_weight = nn.Softmax(dim=-1)(fusion_score)
            fused = fsh * fusion_weight[..., 0:1] + gah * fusion_weight[..., 1:2]
            scale_features.append(fused)
        # Tổng hợp các scale
        concat_feature = torch.cat([f[:, -1, :, :] for f in scale_features], dim=-1)  # B x N x (2D * num_scales)
        fusion_score = self.fusion_fc2(self.relu(self.fusion_fc1(concat_feature)))
        fusion_weight = self.softmax(fusion_score)  # B x N x num_scales
        fusion_weight = fusion_weight.unsqueeze(-1)  # B x N x num_scales x 1
        stacked = torch.stack([f[:, -1, :, :] for f in scale_features], dim=2)  # B x N x num_scales x 2D
        fused_final = (stacked * fusion_weight).sum(dim=2)  # B x N x 2D
        return fused_final

class FullModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.out_dim = args.out_dim
        self.temporal_embedding = TemporalEmbedding(args.hidden_dim)
        self.input_embedding = nn.Sequential(nn.Linear(args.in_dim, args.hidden_dim), nn.ReLU())
        self.multi_scale_hyper = MultiScaleHypergraphModule(args, args.predefined_adj)
        self.pred_head = nn.Sequential(
            nn.Linear(args.hidden_dim * 2 + 5 * 12, args.hidden_dim),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, self.out_dim)
        )

    def forward(self, data):
        feat = data['feat']  # B x T x N x Din
        tod_onehot = data['tod_idx']  # B x T x 288
        dow_onehot = data['dow_onehot']  # B x T x 7
        input_emb = self.input_embedding(feat)
        tod_emb, dow_emb = self.temporal_embedding(tod_onehot, dow_onehot)
        feature = input_emb + tod_emb + dow_emb  # B x T x N x D

        fused_feature = self.multi_scale_hyper(feature)  # B x N x 2D

        future_feature = data['target'][:, :, :, -5:].transpose(1, 2).reshape(self.args.batch_size, self.args.num_nodes, -1)
        if self.args.feat_off == 1:
            future_feature = self.args.scaler.transform(future_feature)
        else:
            future_feature = self.args.scaler[0].transform(future_feature)
        seq_out_len = self.args.seq_out_len
        B, N, D = fused_feature.shape
        pred_list = []
        for t in range(seq_out_len):
            pred_input = torch.cat([fused_feature, future_feature], dim=-1)  # B x N x (2D + 5*12)
            pred_t = self.pred_head(pred_input)  # B x N x out_dim
            pred_list.append(pred_t.unsqueeze(1))  # B x 1 x N x out_dim

        prediction = torch.cat(pred_list, dim=1)  # B x seq_out_len x N x out_dim
        return prediction