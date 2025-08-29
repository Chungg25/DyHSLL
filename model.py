import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import SpectralClustering
import pickle
import os

# Import xPatch components
from layers.decomp import DECOMP

class HypergraphConstruction:
    """Construct GAH and FSH hypergraphs"""
    
    @staticmethod
    def construct_gah(adj_matrix, k_neighbors=8):
        """Construct Geographical Adjacency Hypergraph"""
        N = adj_matrix.shape[0]
        H_gah = torch.zeros(N, N)  # Each node forms one hyperedge
        
        for v in range(N):
            # Find k-nearest neighbors for vertex v
            distances = adj_matrix[v].clone()
            distances[v] = float('inf')  # Exclude self
            k_nearest = torch.topk(-distances, k=k_neighbors, largest=True).indices
            
            # Create hyperedge: {v} ∪ Nk(v)
            H_gah[v, v] = 1.0  # Include vertex itself
            H_gah[k_nearest, v] = 1.0  # Include k-nearest neighbors
            
        return H_gah
    
    @staticmethod
    def construct_fsh_from_dtw(dtw_distance_matrix, num_clusters=16):
        """Construct Feature Similarity Hypergraph using precomputed DTW distances"""
        N = dtw_distance_matrix.shape[0]
        
        # Convert numpy to torch if needed
        if isinstance(dtw_distance_matrix, np.ndarray):
            dtw_distance_matrix = torch.from_numpy(dtw_distance_matrix).float()
        
        # Convert distance to similarity matrix
        similarity_matrix = torch.exp(-dtw_distance_matrix / (dtw_distance_matrix.std() + 1e-8))
        
        # Use spectral clustering on DTW similarities
        clustering = SpectralClustering(
            n_clusters=num_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        cluster_labels = clustering.fit_predict(similarity_matrix.numpy())
        
        # Construct hypergraph incidence matrix
        H_fsh = torch.zeros(N, num_clusters)
        for node_idx, cluster_id in enumerate(cluster_labels):
            H_fsh[node_idx, cluster_id] = 1.0
            
        return H_fsh, cluster_labels

class HypergraphConvolution(nn.Module):
    """Hypergraph Convolution Layer"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) / math.sqrt(in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, H):
        """
        X: [B, N, F] - node features
        H: [N, E] - hypergraph incidence matrix
        """
        H = H.to(X.device)
        B, N, F = X.shape
        E = H.shape[1]
        
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        
        # Compute degree matrices
        Dv = torch.sum(H, dim=1) + eps  # Node degrees [N]
        De = torch.sum(H, dim=0) + eps  # Hyperedge degrees [E]
        
        # Compute normalization matrices
        Dv_inv_sqrt = torch.pow(Dv, -0.5)  # [N]
        De_inv = torch.pow(De, -1)  # [E]
        
        # Apply linear transformation
        X = torch.matmul(X, self.weight) + self.bias  # [B, N, out_dim]
        
        # Hypergraph convolution: Dv^{-1/2} H De^{-1} H^T Dv^{-1/2} X
        # Step 1: Dv^{-1/2} X
        X = X * Dv_inv_sqrt.unsqueeze(0).unsqueeze(-1)  # [B, N, out_dim]
        
        # Step 2: H^T X -> aggregate nodes to hyperedges
        X = torch.einsum('ne,bnd->bed', H, X)  # [B, E, out_dim]
        
        # Step 3: De^{-1} X -> normalize by hyperedge degrees
        X = X * De_inv.unsqueeze(0).unsqueeze(-1)  # [B, E, out_dim]
        
        # Step 4: H X -> aggregate hyperedges back to nodes
        X = torch.einsum('ne,bed->bnd', H, X)  # [B, N, out_dim]
        
        # Step 5: Dv^{-1/2} X
        X = X * Dv_inv_sqrt.unsqueeze(0).unsqueeze(-1)  # [B, N, out_dim]
        
        return self.dropout(X)

class SpatialHypergraphModule(nn.Module):
    """Spatial processing using hypergraphs (GAH + FSH)"""
    def __init__(self, hidden_dim, H_gah, H_fsh, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Register hypergraph incidence matrices as buffers
        self.register_buffer('H_gah', H_gah)
        self.register_buffer('H_fsh', H_fsh)
        
        # GAH and FSH convolution layers
        self.gah_conv = HypergraphConvolution(hidden_dim, hidden_dim, dropout)
        self.fsh_conv = HypergraphConvolution(hidden_dim, hidden_dim, dropout)
        
        # Fusion layers
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        
    def forward(self, x):
        """
        x: [B, T, N, D] or [B, N, D] - temporal node features
        """
        if x.dim() == 4:  # [B, T, N, D]
            B, T, N, D = x.shape
            x_reshaped = x.view(B * T, N, D)  # [B*T, N, D]
            need_reshape = True
        else:  # [B, N, D]
            x_reshaped = x
            need_reshape = False
        
        # Apply GAH and FSH convolutions
        gah_out = self.gah_conv(x_reshaped, self.H_gah)  # [B*T, N, D] or [B, N, D]
        fsh_out = self.fsh_conv(x_reshaped, self.H_fsh)  # [B*T, N, D] or [B, N, D]
        
        # Adaptive fusion with learnable weights
        weights = F.softmax(self.fusion_weight, dim=0)
        fused = weights[0] * gah_out + weights[1] * fsh_out
        
        # Additional fusion through concatenation and projection
        concat_features = torch.cat([gah_out, fsh_out], dim=-1)  # [B*T, N, 2D] or [B, N, 2D]
        projected = self.fusion_proj(concat_features)  # [B*T, N, D] or [B, N, D]
        
        # Combine adaptive and projected fusion
        final_out = self.norm(self.activation(fused + projected + x_reshaped))
        
        # Reshape back to original format if needed
        if need_reshape:
            final_out = final_out.view(B, T, N, D)  # [B, T, N, D]
        
        return final_out

class TemporalEmbedding(nn.Module):
    """Enhanced temporal embedding"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.time_embedding = nn.Embedding(288, hidden_dim)  # 24*12 = 288 (5-min intervals)
        self.day_embedding = nn.Linear(7, hidden_dim)  # Day of week one-hot
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, time_idx, day_onehot):
        """
        time_idx: [B, T] - time of day indices
        day_onehot: [B, T, 7] - day of week one-hot
        """
        time_emb = self.time_embedding(time_idx)  # [B, T, D]
        day_emb = self.day_embedding(day_onehot)  # [B, T, D]
        
        temporal_emb = self.norm(time_emb + day_emb)  # [B, T, D]
        return temporal_emb

class xPatchTemporalModule(nn.Module):
    """xPatch-based temporal processing with decomposition"""
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_in_len
        self.pred_len = args.seq_out_len
        self.hidden_dim = args.hidden_dim
        self.num_nodes = args.num_nodes
        
        # xPatch decomposition
        self.ma_type = getattr(args, 'ma_type', 'ema')
        self.alpha = getattr(args, 'alpha', 0.2)
        self.beta = getattr(args, 'beta', 0.1)
        
        if self.ma_type != 'reg':
            self.decomp = DECOMP(self.ma_type, self.alpha, self.beta)
        
        # xPatch networks for seasonal and trend components
        self.seasonal_net = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, self.pred_len)
        )
        
        self.trend_net = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, self.pred_len)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.pred_len * 2, self.pred_len),
            nn.GELU()
        )
        
    def forward(self, x):
        """
        x: [B, T, N, D] - input features
        Returns: [B, pred_len, N, D] - predicted features
        """
        B, T, N, D = x.shape
        
        # Reshape for processing: [B*N*D, T]
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * N * D, T)
        
        if self.ma_type == 'reg':
            # No decomposition
            seasonal_out = self.seasonal_net(x_reshaped)  # [B*N*D, pred_len]
            trend_out = self.trend_net(x_reshaped)  # [B*N*D, pred_len]
        else:
            # xPatch decomposition
            seasonal, trend = self.decomp(x_reshaped.unsqueeze(-1))  # [B*N*D, T, 1]
            seasonal = seasonal.squeeze(-1)  # [B*N*D, T]
            trend = trend.squeeze(-1)  # [B*N*D, T]
            
            # Process seasonal and trend separately
            seasonal_out = self.seasonal_net(seasonal)  # [B*N*D, pred_len]
            trend_out = self.trend_net(trend)  # [B*N*D, pred_len]
        
        # Fusion of seasonal and trend
        combined = torch.cat([seasonal_out, trend_out], dim=-1)  # [B*N*D, pred_len*2]
        output = self.fusion_layer(combined)  # [B*N*D, pred_len]
        
        # Reshape back to [B, pred_len, N, D]
        output = output.reshape(B, N, D, self.pred_len)
        output = output.permute(0, 3, 1, 2)  # [B, pred_len, N, D]
        
        return output

class FullModel(nn.Module):
    """Integrated xPatch + Hypergraph model for spatio-temporal forecasting"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Model parameters
        self.seq_len = args.seq_in_len
        self.pred_len = args.seq_out_len
        self.hidden_dim = args.hidden_dim
        self.num_nodes = args.num_nodes
        self.out_dim = args.out_dim
        
        # Input embedding for traffic data (pick + drop = 2 dimensions)
        self.input_embedding = nn.Linear(2, self.hidden_dim)  # Fixed to 2 for pick/drop
        
        # Temporal embedding
        self.temporal_embedding = TemporalEmbedding(self.hidden_dim)
        
        # Node embedding
        self.node_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)
        
        # Construct hypergraphs
        self.construct_hypergraphs()
        
        # Spatial hypergraph module
        self.spatial_hypergraph = SpatialHypergraphModule(
            self.hidden_dim, 
            self.H_gah, 
            self.H_fsh, 
            args.dropout
        )
        
        # xPatch temporal processing module
        self.xpatch_temporal = xPatchTemporalModule(args)
        
        # Final prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.hidden_dim // 2, self.out_dim)
        )
        
    def construct_hypergraphs(self):
        """Construct GAH and FSH hypergraphs"""
        # Load adjacency matrix
        adj_data_path = f'{self.args.adj_data}' if hasattr(self.args, 'adj_data') and self.args.adj_data else 'NYC/adj_mx_taxi_2.pkl'
        
        try:
            with open(adj_data_path, 'rb') as f:
                adj_matrix = pickle.load(f)
        except FileNotFoundError:
            # Fallback to args.adj_mx if file not found
            adj_matrix = self.args.adj_mx.cpu().numpy() if hasattr(self.args, 'adj_mx') else None
            
        if adj_matrix is None:
            raise ValueError("Cannot load adjacency matrix")
            
        if isinstance(adj_matrix, np.ndarray):
            adj_matrix = torch.from_numpy(adj_matrix).float()
        
        # Load DTW distance matrix
        dtw_data_path = 'NYC_dtw_distance.npy'
        try:
            dtw_distance = np.load(dtw_data_path)
        except FileNotFoundError:
            # Create dummy DTW matrix if file not found
            print("Warning: DTW distance file not found, creating random matrix")
            dtw_distance = np.random.rand(self.num_nodes, self.num_nodes)
            dtw_distance = (dtw_distance + dtw_distance.T) / 2  # Make symmetric
            np.fill_diagonal(dtw_distance, 0)  # Zero diagonal
        
        # Construct hypergraphs
        self.H_gah = HypergraphConstruction.construct_gah(adj_matrix, k_neighbors=8)
        self.H_fsh, self.fsh_clusters = HypergraphConstruction.construct_fsh_from_dtw(dtw_distance, num_clusters=16)
        
    def forward(self, data):
        """
        data: dict containing:
            - feat: [B, T, N, in_dim] (pick + drop + additional features)
            - tod_idx: [B, T] time of day indices
            - dow_onehot: [B, T, 7] day of week one-hot
        """
        feat = data['feat']  # [B, T, N, in_dim]
        tod_idx = data['tod_idx']  # [B, T]
        dow_onehot = data['dow_onehot']  # [B, T, 7]
        
        B, T, N, _ = feat.shape
        
        # Input embedding (use first 2 dimensions for pick/drop only)
        x = self.input_embedding(feat[:, :, :, :2])  # [B, T, N, D]
        
        # Add temporal embeddings
        temporal_emb = self.temporal_embedding(tod_idx, dow_onehot)  # [B, T, D]
        temporal_emb = temporal_emb.unsqueeze(2).expand(-1, -1, N, -1)  # [B, T, N, D]
        
        # Add node embeddings
        node_idx = torch.arange(N, device=x.device)
        node_emb = self.node_embedding(node_idx)  # [N, D]
        node_emb = node_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)  # [B, T, N, D]
        
        # Combine embeddings
        x = x + temporal_emb + node_emb  # [B, T, N, D]
        
        # Apply spatial hypergraph processing
        x_spatial = self.spatial_hypergraph(x)  # [B, T, N, D]
        
        # Apply xPatch temporal processing
        x_temporal = self.xpatch_temporal(x_spatial)  # [B, pred_len, N, D]
        
        # Final prediction
        output = self.pred_head(x_temporal)  # [B, pred_len, N, out_dim]
        
        return output

# Compatibility wrapper for existing training code
class MultiLayerHypergraphBlock(nn.Module):
    """Compatibility wrapper"""
    def __init__(self, args, num_layers, H_adj):
        super().__init__()
        self.block = SpatialHypergraphModule(args.hidden_dim, H_adj, H_adj, args.dropout)
        
    def forward(self, x):
        return self.block(x)

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