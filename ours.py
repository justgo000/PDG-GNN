import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import erdos_renyi_graph, to_dense_adj, dense_to_sparse
from data_utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor

class GraphConvolutionBase(nn.Module):

    def __init__(self, in_features, out_features, variant=False, residual=False):
        super(GraphConvolutionBase, self).__init__()
        self.variant = variant
        self.residual = residual
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        if self.residual:
            self.weight_r = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0):
        hi = torch.spmm(adj, input)
        if self.variant:
            hi = torch.cat([hi, h0], 1)
        output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + torch.mm(input, self.weight_r)
        return output

class GraphMultiConvolution(nn.Module):

    def __init__(self, in_features, out_features, K, residual=True, variant=False, device=None):
        super(GraphMultiConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weights = Parameter(torch.FloatTensor(K, self.in_features,self.out_features))
        self.K = K
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weights.data.uniform_(-stdv, stdv)

    def reg_loss(self):
        weights_mean = torch.mean(self.weights, dim=0, keepdim=True).repeat(self.K, 1, 1)
        return torch.sum(torch.square(self.weights - weights_mean))

    def forward(self, input, adj, h0, z, weights=None):
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
        else:
            support = hi
        supports = support.unsqueeze(0).repeat(self.K, 1, 1)
        if weights is None:
            outputs = torch.matmul(supports, self.weights)
        else:
            outputs = torch.matmul(supports, weights)
        outputs = outputs.transpose(1, 0) # [N, K, D]
        zs = z.unsqueeze(2).repeat(1, 1, self.out_features)
        output = torch.sum(torch.mul(zs, outputs), dim=1)
        if self.residual:
            output = output+input
        return output

class PDGGNN(nn.Module):
    def __init__(self, d, c, args, device):
        super(PDGGNN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(args.num_layers):
            self.convs.append(GraphMultiConvolution(args.hidden_channels, args.hidden_channels, args.K, variant=args.variant,residual=True, device=device))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d, args.hidden_channels))
        self.fcs.append(nn.Linear(args.hidden_channels, c))
        self.context_enc = nn.ModuleList()
        for _ in range(args.num_layers):
            if args.our_method == 'node':
                self.context_enc.append(nn.Linear(args.hidden_channels, args.K))
            elif args.our_method == 'graph':
                self.context_enc.append(GraphConvolutionBase(args.hidden_channels, args.K, variant=False, residual=True))
            else:
                raise NotImplementedError
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.tau = args.tau
        self.method = args.our_method
        self.prior = args.prior
        self.device = device

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        for enc in self.context_enc:
            enc.reset_parameters()

    def forward(self, x, edge_index, training=True):
        n = x.shape[0]
        # adj = torch.zeros(n, n, dtype=torch.int).to(self.device)
        # adj[edge_index[0],edge_index[1]] = 1
        adj = to_dense_adj(edge_index)[0].to(self.device)
        self.training = training
        x = F.dropout(x, self.dropout, training=self.training)
        h = self.act_fn(self.fcs[0](x))
        h0 = h.clone()

        if self.prior == 'mixture' and self.training:
            n = x.shape[0]
            p = adj.coalesce().indices().shape[1] / n / (n - 1)
            edge_index_r = erdos_renyi_graph(num_nodes = n, edge_prob = p)
            values_r = torch.ones(edge_index_r.size(1)).to(self.device)
            g_r = (values_r.numpy(), (edge_index_r[0].numpy(), edge_index_r[1].numpy()))
            adj_r = sys_normalized_adjacency(g_r, size=(n, n))
            adj_r = sparse_mx_to_torch_sparse_tensor(adj_r)
            adj_r = adj_r.to(adj.device)

            logits = []
            h_r = h.clone()
            for i, con in enumerate(self.convs):
                h_r = F.dropout(h_r, self.dropout, training=self.training)
                if self.method == 'node':
                    logit_r = self.context_enc[i](h_r)
                else:
                    logit_r = self.context_enc[i](h_r, adj_r, h0)
                logits.append(logit_r)
                z_r = F.gumbel_softmax(logit_r, tau=self.tau, dim=-1)
                h_r = self.act_fn(con(h_r, adj_r, h0, z_r))
                logits.append(logit_r.detach())

        reg = 0
        for i,con in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            if self.training:
                if self.method == 'node':
                    logit = self.context_enc[i](h)
                else:
                    logit = self.context_enc[i](h, adj, h0)
                z = F.gumbel_softmax(logit, tau=self.tau, dim=-1)
                if self.prior == 'uniform':
                    reg += self.reg_loss(z, logit)
                elif self.prior == 'mixture':
                    reg += self.reg_loss(z, logit, logits[i])
            else:
                if self.method == 'node':
                    z = F.softmax(self.context_enc[i](h), dim=-1)
                else:
                    z = F.softmax(self.context_enc[i](h, adj, h0), dim=-1)
            h = self.act_fn(con(h,adj,h0, z))

        h = F.dropout(h, self.dropout, training=self.training)
        out = self.fcs[-1](h)
        if self.training:
            return out, reg / self.num_layers
        else:
            return out

    def reg_loss(self, z, logit, logit_0 = None):
        if self.prior == 'uniform':
            log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
            return torch.mean(torch.sum(
                torch.mul(z, log_pi), dim=1))
        elif self.prior == 'mixture':
            log_pi = logit - torch.logsumexp(logit, dim=-1, keepdim=True).repeat(1, logit.size(1))
            log_pi_0 = F.softmax(logit_0, dim=1).mean(dim=1, keepdim=True).log()
            return torch.mean(torch.sum(
                torch.mul(z, log_pi - log_pi_0), dim=1))

    def loss_compute(self, d, criterion, logits, reg_loss, lamda):
        y = d.y.squeeze(1).long()
        sup_loss = criterion(logits[d.train_idx], y[d.train_idx])
        loss = sup_loss + lamda * reg_loss
        return loss

