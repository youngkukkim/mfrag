import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import json
import time
import random
import os
import argparse
import datetime
                                 

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch import optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool
import torch.nn.functional as F

import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import BRICS, Draw
from rdkit.Chem import RWMol
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv, global_mean_pool
from torch_scatter import scatter_mean, scatter_add, scatter_std




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


class GraphEncoder(nn.Module):
    def __init__(self,
                node_input_dim=44,
                edge_input_dim=10,
                node_hidden_dim=44):
        super().__init__()

        self.gather = GatherModel(node_input_dim, edge_input_dim,
                                  node_hidden_dim, edge_input_dim)

        self.embed = nn.Sequential(
            nn.Linear(node_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128))

    def forward(self, graph):
        node_features = F.normalize(self.gather(graph), dim=1)
        graph_features = global_mean_pool(node_features, graph.batch)
        return self.embed(graph_features)


class MFRAG(nn.Module):
    def __init__(self,
                device,
                node_input_dim=44,
                edge_input_dim=10,
                node_hidden_dim=44,
                model_arch='dual'):
        super().__init__()

        self.device = device
        self.model_arch = model_arch

        if model_arch == 'shared':
            self.gather = GatherModel(node_input_dim, edge_input_dim,
                                      node_hidden_dim, edge_input_dim)
            self.embed = nn.Sequential(
                nn.Linear(node_hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128))
            self.mol_encoder = None
            self.frag_encoder = None
        elif model_arch == 'dual':
            self.mol_encoder = GraphEncoder(node_input_dim, edge_input_dim, node_hidden_dim)
            self.frag_encoder = GraphEncoder(node_input_dim, edge_input_dim, node_hidden_dim)
        else:
            raise ValueError(f'Unsupported model_arch: {model_arch}')

        self.value_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1))

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def encode_mol(self, graph):
        if self.model_arch == 'shared':
            node_features = F.normalize(self.gather(graph), dim=1)
            graph_features = global_mean_pool(node_features, graph.batch)
            return self.embed(graph_features)
        return self.mol_encoder(graph)

    def encode_frag(self, graph):
        if self.model_arch == 'shared':
            return self.encode_mol(graph)
        return self.frag_encoder(graph)

    def forward(self, graph, get_w=False):
        embedding = self.encode_mol(graph)
        value_pred = self.value_predictor(embedding)
        return value_pred, embedding
        
