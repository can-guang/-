from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
import math
import config

arg = config.Config.config()
device = arg['device']
label_nums = arg['label_nums']
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        # output = F.relu(self.bn1(output))
        output = F.leaky_relu(self.bn1(output), 0.01)
        output = self.conv2(output)
        output = self.bn2(output)
        # return F.relu(x + output)
        return F.leaky_relu((x + output), 0.01)

class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0]),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        # out = F.relu(self.bn1(output))
        out = F.leaky_relu(self.bn1(output), 0.01)
        out = self.conv2(out)
        out = self.bn2(out)
        # return F.relu(extra_x + out)
        return F.leaky_relu((extra_x + out), 0.01)

class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1)
        # self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=8, stride=2)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(512, 1024, [2, 1]),
                                    RestNetBasicBlock(1024, 1024, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.mlp_1 = nn.Sequential(
            nn.Linear(label_nums, label_nums),
            nn.ReLU(),
            # nn.Dropout(p=0.8),
            #self.relu,
            nn.Linear(label_nums, label_nums)
        )

        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 147)
        '''以下是图卷积的内容'''
        # self.num_classes = 147
        t = 0.1
        in_channel = 300
        self.gc1 = GraphConvolution(in_channel, 512)
        self.gc2 = GraphConvolution(512, 1024)
        self.relu = nn.LeakyReLU(0.01)
        adj_file = 'medicine_adj.npy'
        _adj = gen_A(label_nums, t, adj_file)
        # print(_adj)
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 147)


    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        # print(out.shape)
        x_res = self.fc1(out)
        x_res = self.fc2(x_res)
        x_res = self.fc3(x_res)


        word2vec = np.load("word2vec_cor.npy")
        # word2vec = preprocessing.scale(word2vec)
        word2vec = torch.from_numpy(word2vec)
        adj = gen_adj(self.A)
        adj = adj.to(device)
        word2vec = word2vec.to(device)
        # print(word2vec.shape)
        # print(adj.shape)
        x_gcn = self.gc1(word2vec, adj)
        x_gcn = self.relu(x_gcn)
        x_gcn = self.gc2(x_gcn, adj)
        x_gcn = self.relu(x_gcn)
        # print('x_new', x_new)
        x_gcn = x_gcn.transpose(0, 1)
        x_gcn = torch.matmul(out, x_gcn)
        x_gcn = self.mlp_1(x_gcn)

        # print(x_new.shape)
        # x_new = self.fc1(x_new)
        # x_new = self.fc2(x_new)
        # x_new = self.fc3(x_new)
        return x_res, x_gcn



def gen_A(num_classes , t , adj_file):
    result = np.load(adj_file, allow_pickle=True)
    result = result.item()
    _adj = result['adj']
    # print(_adj)
    _nums = result['num']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.2 / (_adj.sum(0) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(D, A), D)
    return adj

