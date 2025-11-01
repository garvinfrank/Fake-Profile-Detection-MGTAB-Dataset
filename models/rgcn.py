print("Loading RGCN module")

import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)
