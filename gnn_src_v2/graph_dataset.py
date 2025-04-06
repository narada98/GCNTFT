import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler

class AirQualityGraphDataset:
    def __init__(
        self, 
        data_path: str,
        window_size: int = 24,  # 24 hours as default window size
        connection_radius: float = 50.0,  # default radius in kilometers (reduced from 1000km)
        normalize: bool = True,
        use_robust_scaler: bool = True,  # Use RobustScaler for PM2.5 values
        min_samples: int = 5  # Minimum number of samples per station
    ):
        """
        Create a graph dataset from air quality data.
        
        Args:
            data_path: Path to the CSV file containing air quality data
            window_size: Number of time steps to include in each node's features
            connection_radius: Maximum distance between nodes to create an edge (in km)
            normalize: Whether to normalize the PM2.5 values
            use_robust_scaler: Whether to use RobustScaler instead of StandardScaler
            min_samples: Minimum number of samples per station
        """
        self.window_size = window_size
        self.connection_radius = connection_radius
        self.normalize = normalize
        self.use_robust_scaler = use_robust_scaler
        self.min_samples = min_samples
        
        # Load and preprocess data
        self.raw_data = pd.read_csv(data_path)
        self.raw_data['datetime'] = pd.to_datetime(self.raw_data['datetime'])
        
        # Filter stations with too few samples
        station_counts = self.raw_data.groupby('station_loc').size()
        valid_stations = station_counts[station_counts >= self.min_samples].index
        self.raw_data = self.raw_data[self.raw_data['station_loc'].isin(valid_stations)]
        
        self.stations = self._extract_unique_stations()
        
        # Create graphs
        self.scalers = {}  # Store scalers for each station
        self.graphs = self._create_graphs()
    
    def _extract_unique_stations(self) -> Dict:
        """Extract unique monitoring stations from the dataset."""
        stations = {}
        for _, row in self.raw_data.drop_duplicates(['station_loc', 'latitude', 'longitude']).iterrows():
            station_id = row['station_loc']
            stations[station_id] = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'city': row['city']
            }
        
        print(f"Found {len(stations)} unique monitoring stations")
        return stations
    
    def _create_distance_matrix(self) -> np.ndarray:
        """Create a distance matrix between all stations."""
        n_stations = len(self.stations)
        distance_matrix = np.zeros((n_stations, n_stations))
        
        station_ids = list(self.stations.keys())
        
        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                station_i = self.stations[station_ids[i]]
                station_j = self.stations[station_ids[j]]
                
                coord_i = (station_i['latitude'], station_i['longitude'])
                coord_j = (station_j['latitude'], station_j['longitude'])
                distance = geodesic(coord_i, coord_j).kilometers

                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                
        return distance_matrix
    
    def _create_adjacency_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Create an adjacency matrix based on the distance matrix and connection radius."""
        adj_matrix = (distance_matrix <= self.connection_radius).astype(float)
        
        # Set diagonal to 1 (self-loops)
        np.fill_diagonal(adj_matrix, 1.0)
        
        # Create distance-weighted edges
        # Closer stations have higher weights
        edge_indices = np.where((adj_matrix > 0) & (distance_matrix > 0))
        for i, j in zip(edge_indices[0], edge_indices[1]):
            if i != j:  # Skip self-loops
                # Use inverse distance weighting (closer = higher weight)
                adj_matrix[i, j] = 1.0 / (1.0 + distance_matrix[i, j]/10)
        
        return adj_matrix
    
    def _create_edge_index_and_attr(self, adj_matrix: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create edge_index and edge_attr from adjacency matrix."""
        # Get indices of non-zero elements
        edges = np.where(adj_matrix > 0)
        edge_index = torch.tensor(np.vstack(edges), dtype=torch.long)
        
        # Get distance values for edges
        edge_attr = torch.tensor(adj_matrix[edges], dtype=torch.float).view(-1, 1)
        
        return edge_index, edge_attr
    
    def _get_station_time_series(self) -> Dict[str, np.ndarray]:
        """Extract time series for each station."""
        station_series = {}
        for station_id in self.stations.keys():
            station_data = self.raw_data[self.raw_data['station_loc'] == station_id]
            station_data = station_data.sort_values('datetime')
            pm25_series = station_data['PM25'].values
            
            # Handle missing values (NaN)
            if np.isnan(pm25_series).any():
                # Linear interpolation for NaN values
                nan_indices = np.where(np.isnan(pm25_series))[0]
                valid_indices = np.where(~np.isnan(pm25_series))[0]
                if len(valid_indices) > 0:  # If there are any valid values
                    for idx in nan_indices:
                        # Find nearest valid points
                        left_valid = valid_indices[valid_indices < idx]
                        right_valid = valid_indices[valid_indices > idx]
                        
                        if len(left_valid) > 0 and len(right_valid) > 0:
                            # Interpolate between valid points
                            left_idx = left_valid[-1]
                            right_idx = right_valid[0]
                            left_val = pm25_series[left_idx]
                            right_val = pm25_series[right_idx]
                            weight = (idx - left_idx) / (right_idx - left_idx)
                            pm25_series[idx] = left_val + weight * (right_val - left_val)
                        elif len(left_valid) > 0:
                            # Use last valid value
                            pm25_series[idx] = pm25_series[left_valid[-1]]
                        elif len(right_valid) > 0:
                            # Use next valid value
                            pm25_series[idx] = pm25_series[right_valid[0]]
            
            if self.normalize:
                # Create and store a scaler for each station
                if self.use_robust_scaler:
                    scaler = RobustScaler()  # Less sensitive to outliers
                else:
                    scaler = StandardScaler()
                
                # Reshape for scaler
                pm25_series = pm25_series.reshape(-1, 1)
                pm25_series = scaler.fit_transform(pm25_series).flatten()
                self.scalers[station_id] = scaler
            
            station_series[station_id] = pm25_series
        
        return station_series
    
    def _create_node_features(self, station_series: Dict[str, np.ndarray]) -> List[torch.Tensor]:
        """Create node features as sliding windows of time series."""
        all_features = []
        
        for station_id, series in station_series.items():
            features = []
            for i in range(len(series) - self.window_size + 1):
                window = series[i:i+self.window_size]
                features.append(torch.tensor(window, dtype=torch.float))
            
            if features:
                # Stack all windows for this station
                station_features = torch.stack(features)
                all_features.append(station_features)
        
        return all_features
    
    def _create_graphs(self) -> List[Data]:
        """Create PyTorch Geometric graph objects."""
        distance_matrix = self._create_distance_matrix()
        adj_matrix = self._create_adjacency_matrix(distance_matrix)
        edge_index, edge_attr = self._create_edge_index_and_attr(adj_matrix)
        
        station_series = self._get_station_time_series()
        node_features_by_station = self._create_node_features(station_series)
        
        # Ensure all stations have the same number of time windows
        min_windows = min(len(features) for features in node_features_by_station) if node_features_by_station else 0
        if min_windows == 0:
            raise ValueError("No valid time windows found for stations")
            
        # Truncate all stations to have the same number of windows
        node_features_by_station = [features[:min_windows] for features in node_features_by_station]
        
        # Create station ID to index mapping for consistent node ordering
        self.station_to_idx = {station_id: idx for idx, station_id in enumerate(self.stations.keys())}
        
        # Create a graph for each time window
        graphs = []
        for time_idx in range(min_windows):
            # Collect features for all stations at time window i
            try:
                x = torch.stack([station_features[time_idx] for station_features in node_features_by_station])
                station_ids = list(self.stations.keys())
                # Create graph
                graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                graph.station_ids = station_ids
                graphs.append(graph)
            except Exception as e:
                print(f"Error creating graph at time index {time_idx}: {e}")
                continue
        
        print(f"Created {len(graphs)} valid graphs")
        return graphs
    
    def plot_graph(self, index: int = 0):
        """Plot the graph structure with node positions based on lat/lon."""
        if len(self.stations) <= 1:
            print("Not enough stations to plot a meaningful graph.")
            return
        
        graph = self.graphs[index]
        
        plt.figure(figsize=(10, 8))
        
        # Extract station coordinates
        station_ids = list(self.stations.keys())
        positions = np.array([[self.stations[id]['latitude'], self.stations[id]['longitude']] for id in station_ids])
        
        # Plot nodes
        plt.scatter(positions[:, 1], positions[:, 0], s=100, c='blue', label='Stations')
        
        # Plot edges
        for i, j in graph.edge_index.t().tolist():
            if i != j:  # Skip self-loops
                plt.plot([positions[i, 1], positions[j, 1]], [positions[i, 0], positions[j, 0]], 'k-', alpha=0.3)
        
        # Add labels
        for i, station_id in enumerate(station_ids):
            plt.annotate(station_id, (positions[i, 1], positions[i, 0]), fontsize=8)
        
        plt.title(f"Air Quality Monitoring Stations Graph")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig("graph_structure.png")
        plt.close()
        
    def __len__(self) -> int:
        """Return the number of graphs."""
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        """Get a graph by index."""
        return self.graphs[idx]