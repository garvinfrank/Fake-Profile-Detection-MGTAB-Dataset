import torch
from torch_geometric.data import Data

# Loading tensors 
features = torch.load("features.pt")
labels_bot = torch.load("labels_bot.pt")
edge_index = torch.load("edge_index.pt")
edge_weight = torch.load("edge_weight.pt")

# Creating the PyTorch Geometric Data object
data = Data(x=features, edge_index=edge_index, edge_attr=edge_weight, y=labels_bot)

# Print information about the graph
print("Graph has {} nodes and {} edges".format(data.num_nodes, data.num_edges))
print("Node features shape:", data.x.shape)
print("Labels shape:", data.y.shape)

# Saving the Data object 
torch.save(data, "mgtab_graph.pt")
print("Graph saved as mgtab_graph.pt")
