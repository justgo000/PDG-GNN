from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T
from data_utils import even_quantile_labels, to_sparse_tensor

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Twitch, PPI, Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data, Batch
from torch_geometric.utils import stochastic_blockmodel_graph, subgraph, homophily, to_dense_adj, dense_to_sparse

from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv


import pickle as pkl
import os

class GCN_gen(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN_gen, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels))

        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
        x = self.convs[-1](x, edge_index)
        return x

def load_twitch_dataset(data_dir, method, train_num=3, train_ratio=0.5, valid_ratio=0.25):
    transform = T.NormalizeFeatures()
    sub_graphs = ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
    x_list, edge_index_list, y_list, env_list = [], [], [], []
    node_idx_list = []
    idx_shift = 0
    for i, g in enumerate(sub_graphs):
        torch_dataset = Twitch(root=f'{data_dir}Twitch',
                              name=g, transform=transform)
        data = torch_dataset[0]
        x, edge_index, y = data.x, data.edge_index, data.y
        x_list.append(x)
        y_list.append(y)
        edge_index_list.append(edge_index + idx_shift)
        env_list.append(torch.ones(x.size(0)) * i)
        node_idx_list.append(torch.arange(data.num_nodes) + idx_shift)

        idx_shift += data.num_nodes

    x = torch.cat(x_list, dim=0)
    y = torch.cat(y_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    env = torch.cat(env_list, dim=0)
    dataset = Data(x=x, edge_index=edge_index, y=y)
    dataset.env = env
    dataset.env_num = len(sub_graphs)
    dataset.train_env_num = train_num

    assert (train_num <= 5)

    ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    idx = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx[:int(idx.size(0) * train_ratio)]
    valid_idx_ind = idx[int(idx.size(0) * train_ratio) : int(idx.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx[int(idx.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]
    dataset.test_ood_idx = [node_idx_list[-1]] if train_num>=4 else node_idx_list[train_num:]

    if method == 'eerm':
        A = to_dense_adj(dataset.edge_index)[0].to(torch.int)[dataset.train_idx][:,dataset.train_idx]
        train_edge_reindex = dense_to_sparse(A)[0]
        #train_edge_index, _ = subgraph(dataset.train_idx, dataset.edge_index)
        dataset.train_edge_reindex = train_edge_reindex

    return dataset

def load_synthetic_dataset(data_dir, name, method, env_num=6, train_num=3, train_ratio=0.5, valid_ratio=0.25):
    transform = T.NormalizeFeatures()
    if name in ['cora', 'citeseer', 'pubmed']:
        torch_dataset = Planetoid(root=f'{data_dir}Planetoid',
                              name=name, transform=transform)
        preprocess_dir = os.path.join(data_dir, 'Planetoid', name)
    elif name == 'photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Photo', transform=transform)
        preprocess_dir = os.path.join(data_dir, 'Amazon', 'Photo')
    elif name == 'computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                               name='Computers', transform=transform)
        preprocess_dir = os.path.join(data_dir, 'Amazon', 'Computers')

    data = torch_dataset[0]

    edge_index = data.edge_index
    x = data.x
    d = x.shape[1]

    preprocess_dir = os.path.join(preprocess_dir, 'gen')
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)
    spu_feat_num = 10
    class_num = data.y.max().item() + 1

    node_idx_list = [torch.arange(data.num_nodes) + i*data.num_nodes for i in range(env_num)]

    file_path = preprocess_dir + f'/{class_num}-{spu_feat_num}-{env_num}.pkl'
    if not os.path.exists(file_path):

        print("creating new synthetic data...")
        x_list, edge_index_list, y_list, env_list = [], [], [], []
        idx_shift = 0

        # Generator_y = GCN_gen(in_channels=d, hidden_channels=10, out_channels=class_num, num_layers=2)
        Generator_x = GCN_gen(in_channels=class_num, hidden_channels=10, out_channels=spu_feat_num, num_layers=2)
        Generator_noise = nn.Linear(env_num, spu_feat_num)

        with torch.no_grad():
            for i in range(env_num):
                # x_new = x
                # y_new = Generator_y(x, edge_index)
                # y_new = torch.argmax(y_new, dim=-1)
                # mask = (torch.ones_like(y_new).float().uniform_(0, 1) >= 0.9).float()
                # y_new = mask * y_new + (1 - mask) * torch.randint(0, class_num, size=y_new.size())
                # y_new = y_new.long()
                label_new = F.one_hot(data.y, class_num).squeeze(1).float()
                context_ = torch.zeros(x.size(0), env_num)
                context_[:, i] = 1
                x2 = Generator_x(label_new, edge_index) + Generator_noise(context_)
                x2 += torch.ones_like(x2).normal_(0, 0.1)
                x_new = torch.cat([x, x2], dim=1)

                x_list.append(x_new)
                y_list.append(data.y)
                edge_index_list.append(edge_index + idx_shift)
                env_list.append(torch.ones(x.size(0)) * i)

                idx_shift += data.num_nodes

        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        env = torch.cat(env_list, dim=0)
        dataset = Data(x=x, edge_index=edge_index, y=y)
        dataset.env = env

        with open(file_path, 'wb') as f:
            pkl.dump((dataset), f, pkl.HIGHEST_PROTOCOL)
    else:
        print("using existing synthetic data...")
        with open(file_path, 'rb') as f:
            dataset = pkl.load(f)

    assert (train_num <= env_num-1)

    ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    idx = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx[:int(idx.size(0) * train_ratio)]
    valid_idx_ind = idx[int(idx.size(0) * train_ratio): int(idx.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx[int(idx.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]
    dataset.test_ood_idx = [node_idx_list[-1]] if train_num==env_num-1 else node_idx_list[train_num:]
    dataset.env_num = env_num
    dataset.train_env_num = train_num

    if method == 'eerm':
        A = to_dense_adj(dataset.edge_index)[0].to(torch.int)[dataset.train_idx][:,dataset.train_idx]
        train_edge_reindex = dense_to_sparse(A)[0]
        #train_edge_index, _ = subgraph(dataset.train_idx, dataset.edge_index)
        dataset.train_edge_reindex = train_edge_reindex

    return dataset

def load_proteins_dataset(data_dir,method, training_species=3, train_ratio=0.5, valid_ratio=0.25, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-proteins', root=f'{data_dir}/ogb')

    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    edge_feat = torch.as_tensor(ogb_dataset.graph['edge_feat'])
    label = torch.as_tensor(ogb_dataset.labels)

    edge_index_ = to_sparse_tensor(edge_index, edge_feat, ogb_dataset.graph['num_nodes'])
    node_feat = edge_index_.mean(dim=1)

    node_species = torch.as_tensor(ogb_dataset.graph['node_species'])

    dataset = Data(x=node_feat, edge_index=edge_index, y=label)

    species = node_species.unique()
    m = {}
    for i in range(species.shape[0]):
        m[int(species[i])] = i
    env = torch.zeros(dataset.num_nodes)
    for i in range(dataset.num_nodes):
        env[i] = m[int(node_species[i])]
    dataset.env = torch.as_tensor(env, dtype=torch.long)
    dataset.env_num = node_species.unique().size(0)
    dataset.train_env_num = training_species

    species_t = node_species.unique()[training_species]
    ind_mask = (node_species < species_t).squeeze(1)
    idx = torch.arange(dataset.num_nodes)
    ind_idx = idx[ind_mask]
    idx_ = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx_[:int(idx_.size(0) * train_ratio)]
    valid_idx_ind = idx_[int(idx_.size(0) * train_ratio): int(idx_.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx_[int(idx_.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]

    dataset.test_ood_idx = []
    for i in range(training_species, node_species.unique().size(0)):
        species_t = node_species.unique()[i]
        ood_mask_i = (node_species == species_t).squeeze(1)
        dataset.test_ood_idx.append(idx[ood_mask_i])

    if method == 'eerm':
        A = to_dense_adj(dataset.edge_index)[0].to(torch.int)[dataset.train_idx][:,dataset.train_idx]
        train_edge_reindex = dense_to_sparse(A)[0]
        dataset.train_edge_reindex = train_edge_reindex

    # if inductive:
    #     train_edge_index, _ = subgraph(train_mask, edge_index)
    #     valid_edge_index, _ = subgraph(train_mask+valid_mask, edge_index)
    #     dataset.train_edge_index = train_edge_index
    #     dataset.valid_edge_index = valid_edge_index

    return dataset


def load_arxiv_dataset(data_dir, method, train_num=3, train_ratio=0.5, valid_ratio=0.25, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'{data_dir}/ogb')

    node_years = ogb_dataset.graph['node_year']

    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels)

    year_bound = [2005, 2010, 2012, 2014, 2016, 2018, 2021]
    env = torch.zeros(label.shape[0])
    for n in range(node_years.shape[0]):
        year = int(node_years[n])
        for i in range(len(year_bound)-1):
            if year >= year_bound[i+1]:
                continue
            else:
                env[n] = i
                break

    dataset = Data(x=node_feat, edge_index=edge_index, y=label)
    dataset.env = env
    dataset.env_num = len(year_bound)
    dataset.train_env_num = train_num

    ind_mask = (node_years < year_bound[train_num]).squeeze(1)
    idx = torch.arange(dataset.num_nodes)
    ind_idx = idx[ind_mask]
    idx_ = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx_[:int(idx_.size(0) * train_ratio)]
    valid_idx_ind = idx_[int(idx_.size(0) * train_ratio): int(idx_.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx_[int(idx_.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    print('dataset',max(env[dataset.train_idx].long()))
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]

    dataset.test_ood_idx = []

    for i in range(train_num, len(year_bound)-1):
        ood_mask_i = ((node_years >= year_bound[i]) * (node_years < year_bound[i+1])).squeeze(1)
        dataset.test_ood_idx.append(idx[ood_mask_i])

    if method == 'eerm':
        A = to_dense_adj(dataset.edge_index)[0].to(torch.int)[dataset.train_idx][:,dataset.train_idx]
        train_edge_reindex = dense_to_sparse(A)[0]
        #train_edge_index, _ = subgraph(dataset.train_idx, dataset.edge_index)
        dataset.train_edge_reindex = train_edge_reindex

    # if inductive:
    #     train_edge_index, _ = subgraph(train_mask, edge_index)
    #     valid_edge_index, _ = subgraph(train_mask+valid_mask, edge_index)
    #     dataset.train_edge_index = train_edge_index
    #     dataset.valid_edge_index = valid_edge_index

    return dataset