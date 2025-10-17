import torch
from torch import nn
from .layer import HGNN_conv
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class HGCN(nn.Module):
    def __init__(self, in_dim, n_hid, dropout=0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout

        self.hgnn = HGNN_conv(in_dim, n_hid)

    def forward(self, x, G):
        x_embed = self.hgnn(x, G)
        x_embed = F.leaky_relu(x_embed, 0.25)

        return x_embed
