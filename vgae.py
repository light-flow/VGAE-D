import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VGAE(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, node_num):
        super(VGAE, self).__init__()
        self.hidden2_dim = hidden2_dim

        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, activation=torch.sigmoid).cuda()
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x: x).cuda()
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x: x).cuda()

        self.gcn_decoder1 = nn.Linear(hidden2_dim, hidden1_dim, bias=False).cuda()
        self.gcn_decoder2 = nn.Linear(hidden1_dim, input_dim, bias=False).cuda()
        self.leaky_relu = nn.LeakyReLU(0.5).cuda()

        self.linear = LinearEncoder(hidden2_dim
                                    , node_num)

    def encode(self, X, adj):
        hidden = self.base_gcn(X, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        # 高斯噪音
        gaussian_noise = torch.randn(X.size(1), self.hidden2_dim).cuda()
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z
        # return self.mean

    def decoder(self, X, adj):
        x = self.leaky_relu(self.gcn_decoder1(torch.matmul(adj, X)))
        x = self.gcn_decoder2(torch.matmul(adj, x))
        return x

    def forward(self, X, adj):
        # Z是节点级representation
        Z = self.encode(X, adj)
        A_pred = dot_product_decode(Z)
        X_pred = self.decoder(Z, adj)
        score = self.linear(Z)
        return A_pred, X_pred, score, self.logstd


class LinearEncoder(nn.Module):

    def __init__(self, hidden_dim, node_num):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * node_num, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).cuda()

    def forward(self, Z):
        score = self.linear(torch.reshape(Z, (len(Z), -1)))
        return score


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.transpose(1, 2)))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)