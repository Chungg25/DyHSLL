import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, softmax
from util import norm_adj
from backbone import GNNLayer


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.zero_()

class HypergraphConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_attention=True,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.1,
                 bias=False):
        super(HypergraphConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def __forward__(self, x, hyperedge_index, alpha=None):
        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B[B == float("inf")] = 0

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out
        return out

    def forward(self, x, hyperedge_index):
        # x: [N, in_channels]
        x = torch.matmul(x, self.weight)
        x_i = torch.index_select(x, dim=0, index=hyperedge_index[0])
        edge_sums = {}
        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            edge_id = edge_id.item()
            node_id = node_id.item()
            if edge_id not in edge_sums:
                edge_sums[edge_id] = x[node_id, :]
            else:
                edge_sums[edge_id] += x[node_id, :]
        result_list = torch.stack([value for value in edge_sums.values()], dim=0)
        x_j = torch.index_select(result_list, dim=0, index=hyperedge_index[1])
        loss_hyper = 0
        for k in range(len(edge_sums)):
            for m in range(len(edge_sums)):
                inner_product = torch.sum(edge_sums[k] * edge_sums[m], dim=0, keepdim=True)
                norm_q_i = torch.norm(edge_sums[k], dim=0, keepdim=True)
                norm_q_j = torch.norm(edge_sums[m], dim=0, keepdim=True)
                alpha = inner_product / (norm_q_i * norm_q_j)
                distan = torch.norm(edge_sums[k] - edge_sums[m], dim=0, keepdim=True)
                loss_item = alpha * distan + (1 - alpha) * (torch.clamp(torch.tensor(4.2) - distan, min=0.0))
                loss_hyper += torch.abs(torch.mean(loss_item))
        loss_hyper = loss_hyper / ((len(edge_sums) + 1) ** 2)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha.squeeze(), hyperedge_index[0], num_nodes=x.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges/2), x.dtype)
        B[B == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        constrain_loss = x_i - x_j
        constrain_lossfin1 = torch.mean(constrain_loss)
        constrain_losstotal = abs(constrain_lossfin1) + loss_hyper
        return out, constrain_losstotal

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)

class FullModel(nn.Module):
    def __init__(self, args):
        super(FullModel, self).__init__()
        self.args = args
        self.time_embedding = nn.Embedding(48, args.hidden_dim)
        self.date_embedding = nn.Linear(7, args.hidden_dim)
        self.node_embedding = nn.Embedding(args.num_nodes, args.hidden_dim)
        self.input_embedding = nn.Sequential(nn.Linear(args.in_dim, args.hidden_dim), nn.ReLU())
        self.main_model = MainModel(args, adj=args.predefined_adj)
        self.pred_head = nn.Sequential(
            nn.Linear(args.main_output_dim + 5 * 12, args.hidden_dim),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.seq_out_len)
        )

    def forward(self, data):
        feat = data['feat']  # B x T x N x 2 (in, out)
        feat_in = feat[..., 0].unsqueeze(-1)   # B x T x N x 1
        feat_out = feat[..., 1].unsqueeze(-1)  # B x T x N x 1

        input_emb_in = self.input_embedding(feat_in)   # B x T x N x hidden_dim
        input_emb_out = self.input_embedding(feat_out) # B x T x N x hidden_dim

        tod_idx = data['tod_idx']  # B x T
        dow_onehot = data['dow_onehot']  # B x T x 7
        node_idx = torch.arange(0, self.args.num_nodes).to(feat.device)  # N
        time_emb = self.time_embedding(tod_idx).unsqueeze(2)
        date_emb = self.date_embedding(dow_onehot).unsqueeze(2)
        node_emb = self.node_embedding(node_idx).unsqueeze(0).unsqueeze(0)

        feature_in = input_emb_in + time_emb + date_emb + node_emb  # B x T x N x hidden_dim
        feature_out = input_emb_out + time_emb + date_emb + node_emb  # B x T x N x hidden_dim

        out_feat_in = self.main_model(feature_in)    # B x N x nD
        out_feat_out = self.main_model(feature_out)  # B x N x nD

        out_feat = torch.cat([out_feat_in, out_feat_out], dim=-1)  # B x N x (2*nD)

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
        self.adj = args.predefined_adj if adj is None else adj  # N x N
        args.main_output_dim = args.hidden_dim * 2
        self.backbone = STBackbone(args, args.num_backbone_layers)
        self.hyper = HypergraphLearning(args, self.args.num_hyper_edge)
        if self.args.use_multi_scale:
            self.multi_scale_STGCN = nn.ModuleList([
                nn.Sequential(STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 12))),
                nn.Sequential(TemporalPooling(mode='mean', ratio=2),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 6))),
                nn.Sequential(TemporalPooling(mode='mean', ratio=3),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 4))),
                nn.Sequential(TemporalPooling(mode='mean', ratio=4),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 3))),
                nn.Sequential(TemporalPooling(mode='mean', ratio=6),
                              STGCNWithHypergraphLearning(args, adj=self.adj, depth=args.num_head_layers,
                                                          hyper=self.hyper if not args.GSL else GSL(args, 2))),
                nn.Sequential(TemporalPooling(mode='mean', ratio=12),
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
        self.adj = args.predefined_adj if adj is None else adj  # N x N
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
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(args.hidden_dim)
        self.hypergraph_conv = HypergraphConv(
            in_channels=args.hidden_dim,
            out_channels=args.hidden_dim,
            use_attention=True,
            heads=1,
            concat=True,
            negative_slope=0.2,
            dropout=0.1,
            bias=False
        )

    def forward(self, x):  # x: [B, T, N, D]
        B, T, N, D = x.shape
        out_list = []
        for b in range(B):
            batch_out = []
            for t in range(T):
                # x[b, t]: [N, D]
                # Bạn cần truyền hyperedge_index phù hợp, ví dụ lấy từ self.args.hyperedge_index
                # mask = self.args.hyperedge_index.to(x.device)  # [2, num_edges]
                mask = self.args.hyperedge_index.to(x.device)
                out, _ = self.hypergraph_conv(x[b, t], mask)
                batch_out.append(out.unsqueeze(0))  # [1, N, D]
            batch_out = torch.cat(batch_out, dim=0)  # [T, N, D]
            out_list.append(batch_out.unsqueeze(0))  # [1, T, N, D]
        out_tensor = torch.cat(out_list, dim=0)  # [B, T, N, D]
        y_final = self.norm(out_tensor + x)
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