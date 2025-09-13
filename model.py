import math
import torch
import torch.nn as nn
from util import norm_adj
from backbone import GNNLayer

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, in_dim, df):
        super().__init__()
        self.fc_tod = nn.Linear(1, df)
        self.fc_dow = nn.Linear(7, df)  # nhận one-hot
        self.fc_feat = nn.Linear(in_dim, in_dim)

    def forward(self, x, tod, dow_onehot):
        # x: B x T x N x F
        B, T, N, F = x.shape
        tod = tod.float().unsqueeze(-1)  # B x T x 1
        dow = dow_onehot.float()         # B x T x 7
        tod_emb = self.fc_tod(tod)       # B x T x df
        dow_emb = self.fc_dow(dow)       # B x T x df
        x_emb = self.fc_feat(x)          # B x T x N x F
        tod_emb = tod_emb.unsqueeze(2).expand(-1, -1, N, -1)  # B x T x N x df
        dow_emb = dow_emb.unsqueeze(2).expand(-1, -1, N, -1)  # B x T x N x df
        out = torch.cat([x_emb, tod_emb, dow_emb], dim=-1)    # B x T x N x (F + 2*df)
        return out

class TemporalSelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.WQ = nn.Linear(in_dim, in_dim)
        self.WK = nn.Linear(in_dim, in_dim)
        self.WV = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        # x: B x T x N x D
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B*N, T, D)
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attn_score = torch.softmax(attn_score, dim=-1)
        out = torch.matmul(attn_score, V)
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return out

class LocalSelfAttention(nn.Module):
    def __init__(self, in_dim, window_size=4):
        super().__init__()
        self.in_dim = in_dim
        self.window_size = window_size
        self.WQ = nn.Linear(in_dim, in_dim)
        self.WK = nn.Linear(in_dim, in_dim)
        self.WV = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        # x: B x T x N x D
        B, T, N, D = x.shape
        pad = self.window_size // 2
        x_padded = torch.cat([x[:, :pad], x, x[:, -pad:]], dim=1)
        out = []
        for t in range(T):
            win = x_padded[:, t:t+self.window_size]  # B x window x N x D
            win = win.permute(0, 2, 1, 3).reshape(B*N, self.window_size, D)
            Q = self.WQ(win[:, pad:pad+1])  # B*N x 1 x D
            K = self.WK(win)
            V = self.WV(win)
            attn = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            val = torch.matmul(attn, V)  # B*N x 1 x D
            val = val.reshape(B, N, D)
            out.append(val)
        out = torch.stack(out, dim=1)  # B x T x N x D
        return out

class FullModel(nn.Module):
    def __init__(self, args):
        super(FullModel, self).__init__()
        self.args = args
        self.time_series_embedding = TimeSeriesEmbedding(args.in_dim, args.embed_dim)
        self.temporal_proj = nn.Linear(args.in_dim + 2 * args.embed_dim, args.hidden_dim)
        self.temporal_attention = TemporalSelfAttention(args.hidden_dim)
        self.input_embedding = nn.Sequential(nn.Linear(args.in_dim, args.hidden_dim), nn.ReLU())
        self.time_embedding = nn.Embedding(288, args.hidden_dim)
        self.date_embedding = nn.Linear(7, args.hidden_dim)
        self.node_embedding = nn.Embedding(args.num_nodes, args.hidden_dim)
        self.main_model = MainModel(args, adj=args.predefined_adj)
        self.pred_head = nn.Sequential(
            nn.Linear(args.main_output_dim + 5 * 12, args.hidden_dim),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.seq_out_len)
        )

    def forward(self, data):
        feat = data['feat']  # B x T x N x Din
        tod_idx = data['tod_idx']  # B x T
        dow_onehot = data['dow_onehot']  # B x T x 7
        feature = self.time_series_embedding(feat, tod_idx, dow_onehot)  # B x T x N x (F + 2*df)
        feature = self.temporal_proj(feature)  # B x T x N x hidden_dim
        feature = self.temporal_attention(feature)  # B x T x N x hidden_dim

        node_idx = torch.arange(0, self.args.num_nodes).to(feat.device)  # N
        input_emb = self.input_embedding(feat)
        time_emb = self.time_embedding(tod_idx).unsqueeze(2)
        date_emb = self.date_embedding(dow_onehot).unsqueeze(2)
        node_emb = self.node_embedding(node_idx).unsqueeze(0).unsqueeze(0)
        feature = input_emb + time_emb + date_emb + node_emb  # B x T x N x D

        out_feat = self.main_model(feature)  # B x N x nD

        future_feature = data['target'][:, :, :, -5:].transpose(1, 2).reshape(self.args.batch_size, self.args.num_nodes, -1)
        if self.args.feat_off == 1:
            future_feature = self.args.scaler.transform(future_feature)
        else:
            future_feature = self.args.scaler[0].transform(future_feature)
        prediction = self.pred_head(torch.cat([out_feat, future_feature], dim=-1))  # B x N x T
        prediction = prediction.transpose(1, 2).unsqueeze(-1)  # B x T x N x 1
        return prediction

class MainModel(nn.Module):
    def __init__(self, args, adj=None):
        super(MainModel, self).__init__()
        self.args = args
        self.adj = args.predefined_adj if adj is None else adj
        args.main_output_dim = args.hidden_dim * 2
        self.backbone = STBackbone(args, args.num_backbone_layers)
        self.hyper = HypergraphLearning(args, self.args.num_hyper_edge)
        if self.args.use_multi_scale:
            self.multi_scale_STGCN = nn.ModuleList([
                nn.Sequential(STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 12))),
                nn.Sequential(LocalSelfAttention(args.hidden_dim, window_size=6),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 6))),
                nn.Sequential(LocalSelfAttention(args.hidden_dim, window_size=4),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 4))),
                nn.Sequential(LocalSelfAttention(args.hidden_dim, window_size=3),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 3))),
                nn.Sequential(LocalSelfAttention(args.hidden_dim, window_size=2),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 2))),
                nn.Sequential(LocalSelfAttention(args.hidden_dim, window_size=1),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 1)))
            ])
        elif args.biscale:
            self.multi_scale_STGCN = nn.ModuleList([
                nn.Sequential(STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 12))),
                nn.Sequential(TemporalPooling(mode='mean', ratio=3),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 4)))
            ])
        else:
            self.multi_scale_STGCN = nn.ModuleList([
                nn.Sequential(STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers, hyper=self.hyper)),
            ])
        self.global_fusion_layer = nn.Sequential(
            nn.Linear(args.hidden_dim * len(self.multi_scale_STGCN), args.main_output_dim // 2),
            nn.ReLU()
        )
        self.local_fusion_layer = nn.Sequential(
            nn.Linear(args.hidden_dim * len(self.multi_scale_STGCN), args.main_output_dim // 2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x)
        global_features = []
        local_features = []
        for i, path in enumerate(self.multi_scale_STGCN):
            y = path(x)
            local_feature = y[:, -1, :, :]
            local_features.append(local_feature)
            global_feature = y.mean(dim=1)
            global_features.append(global_feature)
        local_feature = self.local_fusion_layer(torch.cat(local_features, dim=-1))
        global_feature = self.global_fusion_layer(torch.cat(global_features, dim=-1))
        feature = torch.cat([local_feature, global_feature], dim=-1)
        return feature

class STBackbone(nn.Module):
    def __init__(self, args, num_layers):
        super(STBackbone, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(*[GNNLayer(args, args.predefined_adjs[i],
                                     use_learned_adj=False, padding=2) for j in range(num_layers)]) for i in range(2)])

    def forward(self, feature):
        feature_list = []
        for layer in self.layers:
            x = layer(feature)
            feature_list.append(x)
        feature = torch.stack(feature_list, dim=3).max(dim=3)[0]  # B x T x N x D
        return feature

class STGCNWithHypergraphLearning(nn.Module):
    def __init__(self, args, adj=None, depth=3, num_edges=32, hyper=None):
        super(STGCNWithHypergraphLearning, self).__init__()
        self.args = args
        self.depth = depth
        self.adj = args.predefined_adj if adj is None else adj
        self.stgcns = nn.ModuleList([SpatialTemporalInteractiveGCN(args, self.adj, window_size=args.winsize)
                                     for _ in range(depth)])
        self.hypers = HypergraphLearning(args, num_edges) if hyper is None else hyper
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        if not self.args.use_hyper_graph and not self.args.use_interactive:
            return x
        for i in range(self.depth):
            if self.args.use_hyper_graph and self.args.use_interactive:
                out_1 = self.stgcns[i](x)
                out_2 = self.hypers(x)
                x = (out_1 + out_2) / 2
            elif self.args.use_hyper_graph:
                x = self.hypers(x)
            else:
                x = self.stgcns[i](x)
            if i != self.depth - 1:
                x = self.dropout(x)
        return x

class SpatialTemporalInteractiveGCN(nn.Module):
    def __init__(self, args, adj=None, window_size=2):
        super(SpatialTemporalInteractiveGCN, self).__init__()
        self.args = args
        self.adj = args.predefined_adj if adj is None else adj
        self.padding = window_size - 1
        self.window_size = window_size
        self.proj_1 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.proj_2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(args.hidden_dim)
        self.expanded_adj = torch.zeros(size=(args.num_nodes, args.num_nodes * window_size)).to(args.device)
        self.expanded_adj[:args.num_nodes, -args.num_nodes:] = self.adj
        self.expanded_adj[:args.num_nodes, :] = torch.eye(args.num_nodes).repeat(1, window_size)
        self.expanded_adj = norm_adj(self.expanded_adj)

    def forward(self, x):
        temporal_length = x.size(1)
        next_features = []
        pad = torch.zeros(x.size(0), self.padding, x.size(2), x.size(3)).to(self.args.device)
        feat = torch.cat([pad, x], dim=1)
        for i in range(temporal_length):
            win_feat = feat[:, i: self.window_size + i, :, :]
            large_graph_feat = win_feat.reshape(x.size(0), -1, x.size(3))
            large_graph_feat_1 = self.expanded_adj @ self.proj_1(large_graph_feat)
            large_graph_feat_2 = self.expanded_adj @ self.proj_2(large_graph_feat)
            feat_interactive = self.activation(large_graph_feat_1 * large_graph_feat_2)
            feat_full = feat_interactive + large_graph_feat_1
            next_features.append(feat_full)
        next_feat = torch.stack(next_features, dim=1)
        y_final = self.norm(next_feat + x)
        return y_final

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

class GSL(nn.Module):
    def __init__(self, args, temporal_length):
        super(GSL, self).__init__()
        self.args = args
        self.adj_learned = nn.Linear(temporal_length * args.num_nodes, temporal_length * args.num_nodes, bias=False)
        self.norm = nn.LayerNorm(args.hidden_dim)

    def forward(self, x):
        feat = x.reshape(x.size(0), -1, x.size(3))
        feat = feat.transpose(1, 2)
        feat = self.adj_learned(feat).transpose(1, 2)
        y = feat.reshape(x.size(0), x.size(1), x.size(2), x.size(3))
        y_final = self.norm(y + x)
        return y_final

class TemporalPooling(nn.Module):
    def __init__(self, mode='mean', ratio=2):
        super(TemporalPooling, self).__init__()
        self.mode = mode
        self.ratio = ratio

    def forward(self, x):
        x = x.reshape(x.size(0), -1, self.ratio, x.size(2), x.size(3))
        if self.mode == 'max':
            y = x.max(dim=2)[0]
        else:
            y = x.mean(dim=2)
        return y