import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graph_dataset import AirQualityGraphDataset
from train_gnn import train_gnn
import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def main():
    # Paths
    data_path = "/home/naradaw/code/GCNTFT/data/processed/gnn_ready.csv"
    output_dir = f"/home/naradaw/code/GCNTFT/data/embeddings_v2_final_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create dataset with improved parameters
    print("Creating graph dataset...")
    dataset = AirQualityGraphDataset(
        data_path=data_path,
        window_size=24,
        connection_radius=30.0,  # More restrictive radius for better locality
        normalize=True,
        use_robust_scaler=True  # Better handles outliers in PM2.5 values
    )
    
    # Plot graph
    print("Plotting graph structure...")
    dataset.plot_graph()
    
    # Train GNN and generate embeddings with optimized hyperparameters
    print("Training GNN and generating embeddings...")
    train_gnn(
        data_path=data_path,
        output_dir=output_dir,
        window_size=24,  
        hidden_dim=256,  # Increased capacity
        output_dim=32,   # Increased embedding dimension
        batch_size=32,   # Smaller batch size for better training
        epochs=200,      # More epochs to ensure convergence
        lr=0.0003,       # Lower learning rate for more stable training
        weight_decay=1e-4, # Higher regularization
        patience=15,     # More patience for early stopping
        min_delta=0.0001 # Smaller improvement threshold
    )
    
    # Load embeddings
    structured_embeddings_file = os.path.join(output_dir, "structured_embeddings.npy")
    if os.path.exists(structured_embeddings_file):
        embeddings_data = np.load(structured_embeddings_file, allow_pickle=True).item()
        
        # Create DataFrame with station and timestamp information
        embedding_arrays = [arr for arr in embeddings_data['embedding']]
        
        # Convert list to numpy array for visualization methods
        embedding_arrays_np = np.array(embedding_arrays)
        
        embedding_df = pd.DataFrame(
            embedding_arrays,
            columns=[f"emb_{i}" for i in range(len(embedding_arrays[0]))]
        )
        
        # Add station and timestamp info
        embedding_df['station_id'] = embeddings_data['station_id']
        embedding_df['timestamp_idx'] = embeddings_data['timestamp_idx']
        
        # Load raw data to map timestamp indices to actual timestamps
        raw_data = pd.read_csv(data_path)
        raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
        
        # Map timestamp indices to actual timestamps
        all_timestamps = sorted(raw_data['datetime'].unique())
        valid_timestamps = all_timestamps[23:]  # Skip first window_size-1
        
        embedding_df['datetime'] = embedding_df['timestamp_idx'].apply(
            lambda idx: valid_timestamps[idx] if idx < len(valid_timestamps) else None
        )
        
        # Set multi-index
        embedding_df = embedding_df.set_index(['datetime', 'station_id']).drop('timestamp_idx', axis=1)
        embedding_df.to_csv(os.path.join(output_dir, "tft_ready_embeddings.csv"))
        print("TFT-ready embeddings saved successfully.")
        
        # Visualize embeddings with both PCA and t-SNE, colored by station
        station_ids = embeddings_data['station_id']
        unique_stations = sorted(set(station_ids))
        station_to_color = {station: i for i, station in enumerate(unique_stations)}
        colors = [station_to_color[station] for station in station_ids]
        
        # Run PCA
        pca = PCA(n_components=2)
        reduced_embeddings_pca = pca.fit_transform(embedding_arrays_np)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(reduced_embeddings_pca[:, 0], reduced_embeddings_pca[:, 1], 
                  c=colors, alpha=0.6, cmap='tab20')
        
        # Create legend with station names
        legend1 = plt.legend(handles=scatter.legend_elements()[0], 
                            labels=unique_stations, 
                            title="Stations",
                            loc="upper right")
        plt.gca().add_artist(legend1)
        
        plt.title("PCA of Node Embeddings by Station")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig(os.path.join(output_dir, "station_embeddings_pca.png"))
        
        # Run t-SNE for better visualization of complex relationships
        # Set perplexity based on number of data points, not just unique stations
        perplexity = min(30, max(5, len(embedding_arrays_np) // 100))
        tsne = TSNE(n_components=2, perplexity=perplexity)
        reduced_embeddings_tsne = tsne.fit_transform(embedding_arrays_np)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(reduced_embeddings_tsne[:, 0], reduced_embeddings_tsne[:, 1], 
                  c=colors, alpha=0.6, cmap='tab20')
        
        # Create legend with station names
        legend1 = plt.legend(handles=scatter.legend_elements()[0], 
                            labels=unique_stations, 
                            title="Stations",
                            loc="upper right")
        plt.gca().add_artist(legend1)
        
        plt.title("t-SNE of Node Embeddings by Station")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.savefig(os.path.join(output_dir, "station_embeddings_tsne.png"))
        
        print(f"Visualizations saved to {output_dir}")
    else:
        print(f"Error: Embeddings file not found at {structured_embeddings_file}")

if __name__ == "__main__":
    main()