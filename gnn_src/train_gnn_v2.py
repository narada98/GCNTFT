import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from graph_dataset import AirQualityGraphDataset
from graph_models_w_decoder import GCN

def train_gnn(
    data_path: str,
    output_dir: str,
    window_size: int = 24,
    hidden_dim: int = 64,
    output_dim: int = 32,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 0.001
):
    """
    Train a GNN on air quality data and save node embeddings.
    
    Args:
        data_path: Path to the air quality data CSV
        output_dir: Directory to save embeddings
        window_size: Size of time window for node features
        hidden_dim: Hidden dimension for GNN
        output_dim: Output embedding dimension
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    dataset = AirQualityGraphDataset(
        data_path=data_path,
        window_size=window_size,
        normalize=True
    )
    
    # Split into train/val/test
    n_samples = len(dataset)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    test_size = n_samples - train_size - val_size
    
    indices = list(range(n_samples))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Get input dimension from the first graph
    input_dim = dataset[0].x.size(1)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    
    # Define loss function and optimizer
    # We'll use a simple MSE reconstruction loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            node_emb, graph_emb = model(data)

            x_reconstructed = model.decode(node_emb)
            loss = F.mse_loss(x_reconstructed, data.x)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                node_emb, graph_emb = model(data)
                x_reconstructed = model.decode(node_emb)
                loss = F.mse_loss(x_reconstructed, data.x)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Generate embeddings for all data
    model.eval()
    all_embeddings = []
    
    # Print dataset information
    print(f"Dataset size: {len(dataset)} graphs")
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # Check first graph to get structure information
    first_graph = dataset[0]
    print(f"Node features shape: {first_graph.x.shape}")
    print(f"Number of nodes in first graph: {first_graph.x.size(0)}")
    print(f"Number of edges in first graph: {first_graph.edge_index.size(1)}")
    
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=1):
            data = data.to(device)
            node_emb, _ = model(data)
            all_embeddings.append(node_emb.cpu().numpy())
    
    # Save embeddings
    embeddings_file = os.path.join(output_dir, "node_embeddings.npy")
    np.save(embeddings_file, np.concatenate(all_embeddings))
    
    print(f"Embeddings saved to {embeddings_file}")
    
    # Save model
    model_file = os.path.join(output_dir, "gnn_model.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    data_path = "/home/naradaw/code/GCNTFT/data/processed/data_w_geo_v3.csv"
    output_dir = "/home/naradaw/code/GCNTFT/data/embeddings_train_v2_test"
    
    train_gnn(
        data_path=data_path,
        output_dir=output_dir,
        window_size=24,  # 24-hour window
        hidden_dim=64,
        output_dim=32,
        batch_size=32,
        epochs=20,
        lr=0.001
    )