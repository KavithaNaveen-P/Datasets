from sklearn.neighbors import NearestNeighbors
import torch_geometric.utils as utils

def build_graph(X, y, temporal_window=3, k=5):
    num_nodes = X.shape[0]
    features = X.reshape(num_nodes, -1)
    
    # Temporal edges (sliding window)
    temporal_edges = []
    for i in range(num_nodes):
        for j in range(max(0, i-temporal_window), min(num_nodes, i+temporal_window)):
            if i != j:
                temporal_edges.append((i, j))
    
    # Morphological edges (KNN)
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors(features)
    morphological_edges = [(i, j) for i in range(num_nodes) for j in indices[i]]
    
    # Combine edges
    edge_index = torch.tensor(list(set(temporal_edges + morphological_edges)), dtype=torch.long).t().contiguous()
    
    # Node features and labels
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)

# Build PyTorch Geometric dataset
graph_data = build_graph(X_balanced, y_balanced)
