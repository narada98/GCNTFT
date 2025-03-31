import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class AirQualityGraphDataset:
    def __init__(
        self, 
        data_path: str,
        window_size: int = 24,  # 24 hours as default window size
        connection_radius: float = 10.0,  # in kilometers
        normalize: bool = True
    ):
        """
        Create a graph dataset from air quality data.
        
        Args:
            data_path: Path to the CSV file containing air quality data
            window_size: Number of time steps to include in each node's features
            connection_radius: Maximum distance between nodes to create an edge (in km)
            normalize: Whether to normalize the PM2.5 values
        """
        self.window_size = window_size
        self.connection_radius = connection_radius
        self.normalize = normalize
        
        # Load and preprocess data
        self.raw_data = pd.read_csv(data_path)
        self.raw_data['datetime'] = pd.to_datetime(self.raw_data['datetime'])
        self.stations = self._extract_unique_stations()
        
        # Create graphs
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
            pm25_series = station_data['PM2.5 (ug/m3)'].values
            
            if self.normalize:
                # Normalize the series
                mean = np.mean(pm25_series)
                std = np.std(pm25_series)
                if std > 0:
                    pm25_series = (pm25_series - mean) / std
            
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
        # If we only have one station, create a simple graph
        if len(self.stations) == 1:
            print("Only one station found. Creating a single-node graph.")
            station_series = self._get_station_time_series()
            node_features = self._create_node_features(station_series)[0]
            
            # Create a single node graph for each time window
            graphs = []
            for i in range(len(node_features)):
                # Single node with self-loop
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.tensor([1.0], dtype=torch.float).view(1, 1)
                
                # Create graph
                x = node_features[i].view(1, -1)  # Add batch dimension
                graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                graphs.append(graph)
            
            return graphs
        
        # If we have multiple stations
        else:
            distance_matrix = self._create_distance_matrix()
            adj_matrix = self._create_adjacency_matrix(distance_matrix)
            edge_index, edge_attr = self._create_edge_index_and_attr(adj_matrix)
            
            station_series = self._get_station_time_series()
            node_features_list = self._create_node_features(station_series)
            
            # Ensure all stations have the same number of time windows
            min_windows = min(len(features) for features in node_features_list)
            
            # Create a graph for each time window
            graphs = []
            for i in range(min_windows):
                # Collect features for all stations at time window i
                x = torch.stack([features[i] for features in node_features_list])
                
                # Create graph
                graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                graphs.append(graph)
            
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