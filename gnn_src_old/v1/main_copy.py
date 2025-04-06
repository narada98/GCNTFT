import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graph_dataset import AirQualityGraphDataset
from train_gnn_v2 import train_gnn
import datetime

def main():
    # Paths
    data_path = "/home/naradaw/code/GCNTFT/data/processed/gnn_ready.csv"
    output_dir = f"/home/naradaw/code/GCNTFT/data/embeddings_v2_final_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create dataset
    print("Creating graph dataset...")
    dataset = AirQualityGraphDataset(
        data_path=data_path,
        window_size=24,
        normalize=False
    )
    
    # Plot graph
    print("Plotting graph structure...")
    dataset.plot_graph()
    
    # Train GNN and generate embeddings
    print("Training GNN and generating embeddings...")
    train_gnn(
        data_path=data_path,
        output_dir=output_dir,
        window_size=24,  
        hidden_dim=128,  
        output_dim=32,   
        batch_size=64,   
        epochs=100,      
        lr=0.0005,       
        weight_decay=1e-5  # Add L2 regularization
    )
    
    # Load embeddings
    embeddings_file = os.path.join(output_dir, "node_embeddings.npy")
    
    structured_embeddings_file = os.path.join(output_dir, "structured_embeddings.npy")
    if os.path.exists(structured_embeddings_file):
        embeddings_data = np.load(structured_embeddings_file, allow_pickle=True).item()
        
        # Create DataFrame with station and timestamp information
        embedding_arrays = [arr for arr in embeddings_data['embedding']]
        embedding_df = pd.DataFrame(
            embedding_arrays,
            columns=[f"emb_{i}" for i in range(len(embedding_arrays[0]))]
        )
        
        # Add station and timestamp info
        embedding_df['station_id'] = embeddings_data['station_id']
        embedding_df['timestamp_idx'] = embeddings_data['timestamp_idx']
        
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
    
    elif os.path.exists(embeddings_file):
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
        
        # Get unique stations and timestamps
        unique_stations = raw_data['station_loc'].unique()
        nunique_station_locs = len(unique_stations)
        
        # Get unique sorted timestamps
        all_timestamps = sorted(raw_data['datetime'].unique())
        valid_timestamps = all_timestamps[23:]  # Skip first 23 due to window_size=24
        num_timestamps = len(valid_timestamps)
        
        print(f"Number of unique timestamps: {num_timestamps}")
        print(f"Number of stations: {nunique_station_locs}")
        print(f"Total embeddings: {embeddings.shape[0]}")
        
        # Check if the dimensions match
        if num_timestamps * nunique_station_locs == embeddings.shape[0]:
            print("Dimensions match! Creating properly indexed DataFrame...")
            
            # Create multi-index DataFrame
            multi_index = []
            for ts in valid_timestamps:
                for station in unique_stations:
                    multi_index.append((ts, station))
            
            # Create a MultiIndex from timestamp and station pairs
            idx = pd.MultiIndex.from_tuples(multi_index, names=['datetime', 'station_id'])
            
            # Create DataFrame with proper indexing
            embedding_df = pd.DataFrame(embeddings, index=idx)
            
            # Save for TFT model
            embedding_df.to_csv(os.path.join(output_dir, "tft_ready_embeddings.csv"))
            print("TFT-ready embeddings saved successfully.")
            
            # Create station-specific files if needed
            stations_dir = os.path.join(output_dir, "stations")
            os.makedirs(stations_dir, exist_ok=True)
            
            # Split by station for easier processing
            for station in unique_stations:
                station_data = embedding_df.xs(station, level='station_id')
                station_data.to_csv(os.path.join(stations_dir, f"{station}_embeddings.csv"))
            
            print(f"Station-specific embeddings saved to {stations_dir}/")
        else:
            print("Warning: Number of embeddings doesn't match the expected count!")
            print(f"Expected: {num_timestamps * nunique_station_locs}, Got: {embeddings.shape[0]}")
            
            # Try to save without proper indexing
            embedding_df = pd.DataFrame(embeddings)
            embedding_df.to_csv(os.path.join(output_dir, "raw_embeddings.csv"))
            print("Saved raw embeddings without timestamp/station indexing.")
    else:
        print(f"Error: Embeddings file not found at {embeddings_file}")

if __name__ == "__main__":
    main()