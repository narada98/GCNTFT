import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class GCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.2,
        n_layers: int = 3
    ):
        """
        Graph Convolutional Network for generating node embeddings.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate
            n_layers: Number of GCN layers
        """
        super(GCN, self).__init__()
        
        # Encoder - more layers and batch normalization
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer (input dim -> hidden dim)
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Middle layers (hidden dim -> hidden dim)
        for _ in range(n_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Last layer (hidden dim -> output dim)
        self.conv_layers.append(GCNConv(hidden_dim, output_dim))
        
        # Additional transformation to strengthen node identity
        self.node_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Decoder with more capacity
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.dropout = dropout
    
    def forward(self, data):
        """
        Forward pass through the GNN.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            node_embeddings: Embeddings for each node
            graph_embedding: Embedding for the entire graph
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr.squeeze()
        
        # Node-specific transformation to strengthen node identity
        node_features = self.node_transform(x)
        
        # Process through GCN layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.conv_layers[:-1], self.batch_norms)):
            # Apply convolution
            x_conv = conv(x, edge_index, edge_weight)
            x_conv = bn(x_conv)
            x_conv = F.relu(x_conv)
            
            # Add residual connection if dimensions match
            if x.size(1) == x_conv.size(1):
                x = x_conv + x
            else:
                x = x_conv
                
            # Add node features after first layer to preserve identity
            if i == 0:
                x = x + node_features
                
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer (no batch norm or residual)
        x = self.conv_layers[-1](x, edge_index, edge_weight)
        
        # Node embeddings
        node_embeddings = x
        
        # Graph embedding - mean of node embeddings
        graph_embedding = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        
        return node_embeddings, graph_embedding
    
    def decode(self, embeddings):
        """
        Decode embeddings back to input space.
        
        Args:
            embeddings: Node embeddings from the encoder
            
        Returns:
            Reconstructed node features
        """
        return self.decoder(embeddings)