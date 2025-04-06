import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graph_dataset import AirQualityGraphDataset
from train_gnn import train_gnn
import datetime

def main():
    # Paths
    data_path = "data/processed/data_w_geo_v3.csv"
    output_dir = f"data/embeddings_v2_lap_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create dataset
    print("Creating graph dataset...")
    dataset = AirQualityGraphDataset(
        data_path=data_path,
        window_size=24,
        normalize=True
    )
    
    # Plot graph
    print("Plotting graph structure...")
    dataset.plot_graph()
    
    # Train GNN and generate embeddings
    print("Training GNN and generating embeddings...")
    train_gnn(
        data_path=data_path,
        output_dir=output_dir,
        window_size=24,  # Increase from 24 to capture longer patterns
        hidden_dim=128,  # Increase from 64 for more expressive power
        output_dim=64,   # Increase from 32 to capture more details
        batch_size=64,   # Larger batch size for more stable learning
        epochs=100,      # Train longer (from 50)
        lr=0.0005,       # Slightly lower learning rate
    )
    
    # Load embeddings
    embeddings_file = os.path.join(output_dir, "node_embeddings.npy")
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file)
        print(f"Loaded embeddings with shape: {embeddings.shape}")
        
        # Visualize embeddings using PCA
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("2D Projection of Graph Embeddings")
        plt.colorbar(label="Time Index")
        plt.savefig("embedding_visualization.png")
        plt.close()
        
        print("Embeddings visualization saved as 'embedding_visualization.png'")
        
        # Save embeddings with timestamps for future TFT model
        raw_data = pd.read_csv(data_path)
        raw_data['datetime'] = pd.to_datetime(raw_data['datetime'])
        nunique_station_locs = raw_data['station_loc'].nunique()
        
        n_samples = len(dataset)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        test_size = n_samples - train_size - val_size
        
        # Get timestamps for each embedding (skip the first window_size-1 timestamps)
        timestamps = raw_data['datetime'].nunique() - 24 +1
        
        print(f"timestamps: {timestamps}")
        print(f"embeddings: {embeddings.shape[0]}")
        
        if timestamps*nunique_station_locs == embeddings.shape[0]:
            # Save embeddings with timestamps
            embedding_df = pd.DataFrame(embeddings, index=timestamps)
            embedding_df.index.name = 'datetime'
            
            # Save for TFT model
            embedding_df.to_csv(os.path.join(output_dir, "tft_ready_embeddings.csv"))
            print("TFT-ready embeddings saved successfully.")
        else:
            print("Warning: Number of timestamps doesn't match number of embeddings.")
    else:
        print(f"Error: Embeddings file not found at {embeddings_file}")

if __name__ == "__main__":
    main()