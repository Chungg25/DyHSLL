import torch
import torch.nn as nn
from model.PDG2SeqCell import PDG2SeqCell
import numpy as np
import torch.nn.functional as F

# Temporal Embedding Module
class TemporalEmbedding(nn.Module):
    def __init__(self, time_dim, dow_dim, emb_dim):
        super(TemporalEmbedding, self).__init__()
        self.fc_tod = nn.Linear(time_dim, emb_dim)
        self.fc_dow = nn.Linear(dow_dim, emb_dim)

    def forward(self, tod_onehot, dow_onehot):
        # tod_onehot: (B, T, N, time_dim), dow_onehot: (B, T, N, dow_dim)
        tod_emb = self.fc_tod(tod_onehot)
        dow_emb = self.fc_dow(dow_onehot)
        return tod_emb, dow_emb

# Temporal Self-Attention Module
class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalSelfAttention, self).__init__()
        self.WQ = nn.Linear(hidden_dim, hidden_dim)
        self.WK = nn.Linear(hidden_dim, hidden_dim)
        self.WV = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (B, T, N, dh)
        B, T, N, dh = x.shape
        x_reshape = x.permute(0,2,1,3).contiguous() # (B, N, T, dh)
        Q = self.WQ(x_reshape)  # (B, N, T, dh)
        K = self.WK(x_reshape)  # (B, N, T, dh)
        V = self.WV(x_reshape)  # (B, N, T, dh)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (dh ** 0.5)  # (B, N, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        Z = torch.matmul(attn_weights, V)  # (B, N, T, dh)
        Z = Z.permute(0,2,1,3).contiguous()  # (B, T, N, dh)
        return Z

class PDG2Seq_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1, use_hypergraph=True, use_interactive=True, num_hyper_edges=32):
        super(PDG2Seq_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.embed_dim = embed_dim
        self.input_dim_with_time = dim_in + 2 * embed_dim
        self.num_layers = num_layers

        self.use_hypergraph = use_hypergraph
        self.use_interactive = use_interactive
        self.num_hyper_edges = num_hyper_edges

        self.PDG2Seq_cells = nn.ModuleList()
        self.PDG2Seq_cells.append(PDG2SeqCell(node_num, self.input_dim_with_time, dim_out, cheb_k, embed_dim, time_dim,
                                              use_hypergraph=self.use_hypergraph,
                                              use_interactive=self.use_interactive,
                                              num_hyper_edges=self.num_hyper_edges))
        for _ in range(1, num_layers):
            self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim,
                                                  use_hypergraph=self.use_hypergraph,
                                                  use_interactive=self.use_interactive,
                                                  num_hyper_edges=self.num_hyper_edges))

        # Add temporal embedding and self-attention
        self.temporal_emb = TemporalEmbedding(time_dim=48, dow_dim=7, emb_dim=embed_dim)
        self.temporal_attn = TemporalSelfAttention(hidden_dim=dim_out)

    def forward(self, x, init_state, node_embeddings):
        # x: (B, T, N, D) where D = input_dim + 2 (tod, dow)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num, f"Expected node dim {self.node_num}, got {x.shape[2]}"
        assert x.shape[3] >= self.input_dim + 2, f"Expected at least {self.input_dim + 2} features, got {x.shape[3]}"
        seq_length = x.shape[1]
        # Split input features and time features
        x_input = x[..., :self.input_dim]  # (B, T, N, input_dim)
        tod = x[..., -2].long()           # (B, T, N)
        dow = x[..., -1].long()           # (B, T, N)
        tod_onehot = F.one_hot(tod, num_classes=48).float()
        dow_onehot = F.one_hot(dow, num_classes=7).float()
        tod_emb, dow_emb = self.temporal_emb(tod_onehot, dow_onehot)  # (B, T, N, emb_dim)
        # Concatenate temporal embeddings to input
        current_inputs = torch.cat([x_input, tod_emb, dow_emb], dim=-1)
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.PDG2Seq_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :], node_embeddings[1][:, t, :], node_embeddings[2]])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            current_inputs = self.temporal_attn(current_inputs)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.PDG2Seq_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)



class PDG2Seq_Dncoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1,  use_hypergraph=True, use_interactive=True, num_hyper_edges=32):
        super(PDG2Seq_Dncoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.embed_dim = embed_dim
        self.input_dim_with_time = dim_in + 2 * embed_dim
        self.num_layers = num_layers
        
        self.use_hypergraph = use_hypergraph
        self.use_interactive = use_interactive
        self.num_hyper_edges = num_hyper_edges

        self.PDG2Seq_cells = nn.ModuleList()
        self.PDG2Seq_cells.append(PDG2SeqCell(node_num, self.input_dim_with_time, dim_out, cheb_k, embed_dim, time_dim,
                                              use_hypergraph=self.use_hypergraph,
                                              use_interactive=self.use_interactive,
                                              num_hyper_edges=self.num_hyper_edges))
        for _ in range(1, num_layers):
            self.PDG2Seq_cells.append(PDG2SeqCell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim,
                                                  use_hypergraph=self.use_hypergraph,
                                                  use_interactive=self.use_interactive,
                                                  num_hyper_edges=self.num_hyper_edges))

        self.temporal_emb = TemporalEmbedding(time_dim=48, dow_dim=7, emb_dim=embed_dim)
        self.temporal_attn = TemporalSelfAttention(hidden_dim=dim_out)

    def forward(self, xt, init_state, node_embeddings):
        # xt: (B, N, D) where D = input_dim + 2 (tod, dow)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num, f"Expected node dim {self.node_num}, got {xt.shape[1]}"
        assert xt.shape[2] >= self.input_dim + 2, f"Expected at least {self.input_dim + 2} features, got {xt.shape[2]}"
        # Split input features and time features
        x_input = xt[..., :self.input_dim]  # (B, N, input_dim)
        tod = xt[..., -2].long()           # (B, N)
        dow = xt[..., -1].long()           # (B, N)
        tod_onehot = F.one_hot(tod, num_classes=48).float()
        dow_onehot = F.one_hot(dow, num_classes=7).float()
        tod_emb, dow_emb = self.temporal_emb(tod_onehot.unsqueeze(1), dow_onehot.unsqueeze(1))  # (B, 1, N, emb_dim)
        tod_emb = tod_emb.squeeze(1)
        dow_emb = dow_emb.squeeze(1)
        current_inputs = torch.cat([x_input, tod_emb, dow_emb], dim=-1)
        output_hidden = []
        for i in range(self.num_layers):
            state = self.PDG2Seq_cells[i](current_inputs, init_state[i], [node_embeddings[0], node_embeddings[1], node_embeddings[2]])
            output_hidden.append(state)
            current_inputs = state
        current_inputs = current_inputs.unsqueeze(1)
        current_inputs = self.temporal_attn(current_inputs)
        current_inputs = current_inputs.squeeze(1)
        return current_inputs, output_hidden


class PDG2Seq(nn.Module):
    def __init__(self, args):
        super(PDG2Seq, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.cl_decay_steps = args.lr_decay_step
        self.node_embeddings1 = nn.Parameter(torch.empty(self.num_node, args.embed_dim))
        self.T_i_D_emb1 = nn.Parameter(torch.empty(48, args.time_dim))
        self.D_i_W_emb1 = nn.Parameter(torch.empty(7, args.time_dim))
        self.T_i_D_emb2 = nn.Parameter(torch.empty(48, args.time_dim))
        self.D_i_W_emb2 = nn.Parameter(torch.empty(7, args.time_dim))

        self.encoder = PDG2Seq_Encoder(
            args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
            args.embed_dim, args.time_dim, args.num_layers,
            use_hypergraph=args.use_hypergraph,
            use_interactive=args.use_interactive,
            num_hyper_edges=args.num_hyper_edges
        )

        self.decoder = PDG2Seq_Dncoder(
            args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
            args.embed_dim, args.time_dim, args.num_layers,
            use_hypergraph=args.use_hypergraph,
            use_interactive=args.use_interactive,
            num_hyper_edges=args.num_hyper_edges
        )

        #predictor
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        # Adjust input_dim for temporal embedding concat
        self.temporal_emb_dim = args.embed_dim * 2
        self.input_dim_with_time = self.input_dim + self.temporal_emb_dim

    def forward(self, source, traget=None, batches_seen=None):
        #source: B, T_1, N, D
        #target: B, T_2, N, D


        t_i_d_data1 = source[..., 0,-2]
        t_i_d_data2 = traget[..., 0,-2]
        # T_i_D_emb = self.T_i_D_emb[(t_i_d_data[:, -1, :] * 48).type(torch.LongTensor)]
        T_i_D_emb1_en = self.T_i_D_emb1[(t_i_d_data1 * 48).type(torch.LongTensor)]
        T_i_D_emb2_en = self.T_i_D_emb2[(t_i_d_data1 * 48).type(torch.LongTensor)]

        T_i_D_emb1_de = self.T_i_D_emb1[(t_i_d_data2 * 48).type(torch.LongTensor)]
        T_i_D_emb2_de = self.T_i_D_emb2[(t_i_d_data2 * 48).type(torch.LongTensor)]
        if self.use_W:
            d_i_w_data1 = source[..., 0,-1]
            d_i_w_data2 = traget[..., 0,-1]
            # D_i_W_emb = self.D_i_W_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
            D_i_W_emb1_en = self.D_i_W_emb1[(d_i_w_data1).type(torch.LongTensor)]
            D_i_W_emb2_en = self.D_i_W_emb2[(d_i_w_data1).type(torch.LongTensor)]

            D_i_W_emb1_de = self.D_i_W_emb1[(d_i_w_data2).type(torch.LongTensor)]
            D_i_W_emb2_de = self.D_i_W_emb2[(d_i_w_data2).type(torch.LongTensor)]

            node_embedding_en1 = torch.mul(T_i_D_emb1_en, D_i_W_emb1_en)
            node_embedding_en2 = torch.mul(T_i_D_emb2_en, D_i_W_emb2_en)

            node_embedding_de1 = torch.mul(T_i_D_emb1_de, D_i_W_emb1_de)
            node_embedding_de2 = torch.mul(T_i_D_emb2_de, D_i_W_emb2_de)
        else:
            node_embedding_en1 = T_i_D_emb1_en
            node_embedding_en2 = T_i_D_emb2_en

            node_embedding_de1 = T_i_D_emb1_de
            node_embedding_de2 = T_i_D_emb2_de


        en_node_embeddings=[node_embedding_en1, node_embedding_en2, self.node_embeddings1]

        # Prepare source with all features for temporal embedding
        # (B, T, N, D) where D includes speed + tod + dow
        # The encoder will handle temporal embedding and attention
        init_state = self.encoder.init_hidden(source.shape[0]).to(source.device)
        state, _ = self.encoder(source, init_state, en_node_embeddings)
        state = state[:, -1:, :, :].squeeze(1)

        ht_list = [state] * self.num_layers

        go = torch.zeros((source.shape[0], self.num_node, self.output_dim), device=source.device)
        out = []
        for t in range(self.horizon):
            # Prepare decoder input: concat go with time features for t-th step
            # Assume traget contains time features at [:, t, :, :]
            # dec_input = traget[:, t, :, :self.input_dim]
            # Add time features (tod, dow) from traget
            tod = traget[:, t, :, -2].unsqueeze(-1)
            dow = traget[:, t, :, -1].unsqueeze(-1)
            dec_input = torch.cat([go, tod, dow], dim=-1)
            state, ht_list = self.decoder(dec_input, ht_list, [node_embedding_de1[:, t, :], node_embedding_de2[:, t, :], self.node_embeddings1])
            go = self.proj(state)
            out.append(go)
            if self.training:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    go = traget[:, t, :, :self.input_dim]
        output = torch.stack(out, dim=1)

        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
        return x