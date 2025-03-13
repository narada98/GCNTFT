from graph_dataset import AirQualityGraphDataset
import torch

def test_dataset():
    # Create dataset
    dataset = AirQualityGraphDataset(
        data_path='/home/naradalinux/dev/GCNTFT/outputs/tests/sample_air_quality.csv',
        window_size=24,  # 24-hour window
        connection_radius=15.0,  # 15km radius
        normalize=True
    )
    
    # Print basic information
    print(f"Dataset contains {len(dataset)} graphs")
    
    # Check the first graph
    graph = dataset[0]
    print("\nFirst graph structure:")
    print(f"- Number of nodes: {graph.x.shape[0]}")
    print(f"- Node feature dimension: {graph.x.shape[1]}")
    print(f"- Number of edges: {graph.edge_index.shape[1]}")
    
    # Check connection pattern
    print("\nConnection pattern:")
    for i in range(graph.x.shape[0]):
        connected_to = []
        for j in range(graph.edge_index.shape[1]):
            if graph.edge_index[0, j].item() == i:
                connected_to.append(graph.edge_index[1, j].item())
        print(f"Node {i} is connected to: {connected_to}")
    
    # Visualize the graph
    print("\nSaving graph visualization...")
    dataset.plot_graph(index=0)
    print("Graph visualization saved as 'graph_structure.png'")

if __name__ == "__main__":
    test_dataset()