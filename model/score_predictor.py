import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, global_mean_pool
from torch_scatter import scatter_mean, scatter_add, scatter_std


def my_mean_pooling(data, group_sizes):
    """
    PyTorch 텐서를 사용하여 첫 번째 리스트(벡터)의 요소를 두 번째 리스트의 그룹 크기에 따라 평균 계산.

    Parameters:
    - data (torch.Tensor): 벡터로 구성된 텐서 (N, D)
    - group_sizes (list of int): 그룹 크기를 나타내는 리스트

    Returns:
    - torch.Tensor: 그룹별 평균 벡터를 하나로 묶은 텐서 (G, D), G는 그룹 수
    """
                    
    group_indices = []
    start = 0
    for size in group_sizes:
        group_indices.extend([len(group_indices)] * size)
        start += size

            
    group_indices = torch.tensor(group_indices, dtype=torch.long)

                          
    result = torch.stack([
        data[group_indices == i].mean(dim=0)
        for i in torch.unique(group_indices)
    ])

    return result


class GatherModel(nn.Module):
    """
    MPNN from `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`
    """
    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=3,
                 dropout=0.0):
        super().__init__()

        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_channels=node_hidden_dim,
                           out_channels=node_hidden_dim,
                           nn=edge_network,
                           aggr='add',
                           root_weight=True
                           )
        self.dropout = dropout

    def forward(self, g):
        init = g.x.clone()
        out = F.relu(self.lin0(g.x))
        for i in range(self.num_step_message_passing):
            if len(g.edge_attr) != 0:
                m = torch.relu(self.conv(out, g.edge_index, g.edge_attr))
            else:
                m = torch.relu(self.conv.bias + out)
            out = self.message_layer(torch.cat([m, out], dim=1))
        return out + init


class SP(nn.Module):
    def __init__(self,
                device,
                node_input_dim=44,
                edge_input_dim=10,
                node_hidden_dim=44):
        super().__init__()

        self.device = device

        self.gather = GatherModel(node_input_dim, edge_input_dim,
                                  node_hidden_dim, edge_input_dim)

                                          
                                                          
                                              
                        
                                            
                           
        
        self.embed = nn.Sequential(
            nn.Linear(node_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128))
                    
        self.reg_predictor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1))

                       
        self.tf_predictor = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid())
        
        self.mse_loss = torch.nn.MSELoss()

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def forward(self, graph, get_w=False):
        node_features = F.normalize(self.gather(graph), dim=1)
                                                                                
        graph_features = global_mean_pool(node_features, graph.batch)
        embedding = self.embed(graph_features)
        reg_pred = self.reg_predictor(embedding)
        tf_pred = self.tf_predictor(embedding)
                      
                                    
                                    
                                     
                     
        return reg_pred, tf_pred, embedding
                                    
        