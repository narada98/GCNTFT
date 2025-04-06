import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from graph_dataset import AirQualityGraphDataset
from graph_model import GCN

def train_gnn(
    data_path: str,
    output_dir: str,
    window_size: int = 24,
    hidden_dim: int = 128,
    output_dim: int = 64,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    patience: int = 10,
    min_delta: float = 0.001,
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
        weight_decay: L2 regularization
        patience: Early stopping patience
        min_delta: Minimum change to qualify as improvement
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
    
    # Use random split instead of sequential for better generalization
    indices = torch.randperm(n_samples).tolist()
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
    
    # Initialize model with more capacity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(
        input_dim=input_dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim, 
        dropout=0.2, 
        n_layers=4
    ).to(device)
    
    # Define loss function and optimizer with higher weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    all_train_losses = []
    all_val_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
            
            # MSE Loss with L1 regularization for embeddings
            mse_loss = F.mse_loss(x_reconstructed, data.x)
            l1_reg = 0.001 * torch.norm(node_emb, p=1)
            loss = mse_loss + l1_reg
            
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += mse_loss.item()  # Track only the MSE part for monitoring
            
        train_loss /= len(train_loader)
        all_train_losses.append(train_loss)
        
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
        all_val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping with minimum improvement threshold
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(output_dir, "best_gnn_model.pt"))
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement, patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training and validation loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(all_train_losses, label='Train Loss')
    plt.plot(all_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    
    # Load best model for embedding generation
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_gnn_model.pt")))
    
    # Test set evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            node_emb, graph_emb = model(data)
            x_reconstructed = model.decode(node_emb)
            loss = F.mse_loss(x_reconstructed, data.x)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    print(f"Final test loss: {test_loss:.4f}")
    
    # Generate embeddings for all data
    model.eval()
    all_embeddings = []
    all_stations = []
    all_timestamps = []
    timestamp_counter = 0
    
    # Print dataset information
    print(f"Dataset size: {len(dataset)} graphs")
    
    # Dictionary to store embeddings by station and timestamp
    station_embeddings = {}
    
    with torch.no_grad():
        for time_idx, data in enumerate(DataLoader(dataset, batch_size=1)):
            data = data.to(device)
            node_emb, _ = model(data)
            
            # Process each node (station) separately to maintain identity
            for i, station_id in enumerate(data.station_ids[0]):
                if station_id not in station_embeddings:
                    station_embeddings[station_id] = {}
                    
                # Store embedding for this station at this timestamp
                station_embeddings[station_id][time_idx] = node_emb[i].cpu().numpy()
                
                # Also append to the flat lists for backward compatibility
                all_embeddings.append(node_emb[i].cpu().numpy())
                all_stations.append(station_id)
                all_timestamps.append(time_idx)
    
    # Create structured embeddings with proper mapping
    embeddings_data = {
        'embedding': all_embeddings,
        'station_id': all_stations,
        'timestamp_idx': all_timestamps,
        'station_embeddings': station_embeddings
    }
    np.save(os.path.join(output_dir, "structured_embeddings.npy"), embeddings_data)
    
    # Save model
    model_file = os.path.join(output_dir, "gnn_model.pt")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    # data_path = "/home/naradaw/code/GCNTFT/data/processed/data_w_geo_v3.csv"
    # output_dir = "/home/naradaw/code/GCNTFT/data/embeddings_train_v2_test"
    
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