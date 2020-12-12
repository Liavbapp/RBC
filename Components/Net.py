import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class Net(torch.nn.Module):
    def __init__(self, dataset, R_flat, T_flat):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, T_flat.size()[1])
        self.conv1.weight = torch.nn.Parameter(T_flat)
        self.conv2 = GCNConv(T_flat.size()[1], R_flat.size()[1])
        self.conv2.weight = torch.nn.Parameter(R_flat)
        self.conv3 = GCNConv(R_flat.size()[1], dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
