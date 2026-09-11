"""Music graph construction utilities."""

import torch
from torch_geometric.data import Data

def create_graph(node_features, edge_index, labels):
    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long)
    )
