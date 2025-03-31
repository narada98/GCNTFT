import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1
    ):
        """
        Graph Convolutional Network for generating node embeddings.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super(GCN, self).__init__()
        
        # Encoder
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Decoder
        self.decoder1 = nn.Linear(output_dim, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder3 = nn.Linear(hidden_dim, input_dim)
        
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
        
        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Third GCN layer
        x = self.conv3(x, edge_index, edge_weight)
        
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
        x = self.decoder1(embeddings)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.decoder2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.decoder3(x)
        
        return x