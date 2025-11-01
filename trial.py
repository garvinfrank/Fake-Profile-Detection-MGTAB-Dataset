import torch

# Load the preprocessed tensors
features = torch.load("features.pt")           # Node feature matrix
labels_bot = torch.load("labels_bot.pt")       # Bot (fake profile) labels
labels_stance = torch.load("labels_stance.pt") # Stance labels
edge_index = torch.load("edge_index.pt")       # Graph connectivity
edge_type = torch.load("edge_type.pt")         # Edge types
edge_weight = torch.load("edge_weight.pt")     # Edge weights

# Print basic info
print("Features shape:", features.shape)
print("Bot labels shape:", labels_bot.shape)
print("Edge index shape:", edge_index.shape)
print("Edge type shape:", edge_type.shape)
print("Edge weight shape:", edge_weight.shape)
