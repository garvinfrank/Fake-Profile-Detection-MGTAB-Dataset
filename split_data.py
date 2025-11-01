import torch

# Loading the graph data we previously saved
data = torch.load("mgtab_graph.pt")

num_nodes = data.num_nodes
# Creating a random mask for training (80% training, 20% testing)
mask = torch.rand(num_nodes) < 0.8
data.train_mask = mask
data.test_mask = ~mask

print("Training nodes:", data.train_mask.sum().item())
print("Testing nodes:", data.test_mask.sum().item())

# Saving the updated graph with masks
torch.save(data, "mgtab_graph_split.pt")
print("Updated graph saved as mgtab_graph_split.pt")
