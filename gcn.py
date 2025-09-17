# gcn.py (PyTorch版)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def glorot(shape):#初始化张量的权重
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return torch.empty(shape).uniform_(-init_range, init_range)



class GraphConvolution(nn.Module):
    def __init__(self, n_node, input_dim, output_dim, dropout=0., act=F.relu, norm=False):
        """
        n_node: 节点数
        input_dim: 输入特征维度
        output_dim: 第一层输出维度（第二层固定为64）
        dropout: dropout比例（示例中未使用）
        act: 激活函数 用于添加非线性
        norm: 是否使用批归一化
        """

         # 调用父类构造函数完成基础初始化
        super(GraphConvolution, self).__init__()
        self.n_node = n_node    # 存储节点数用于后续计算
        self.dropout = dropout  # 配置正则化强度参数
        self.act = act          # 记录非线性激活函数
        self.norm_flag = norm   # 控制归一化流程的开关

        # 第一层权重：input_dim -> output_dim
        self.weight1 = nn.Parameter(glorot((input_dim, output_dim)))#定义了一个可学习的参数weight1
        # 第二层权重：output_dim -> 64
        self.weight2 = nn.Parameter(glorot((output_dim, 64)))#定义了一个可学习的参数weight2

        if self.norm_flag:
            # 对节点维度做BatchNorm，一般来说节点数较多可采用 BatchNorm1d
            self.bn = nn.BatchNorm1d(n_node)#则对节点维度应用BatchNorm1d归一化操作
            
    def forward(self, inputs, adj):
        """
        inputs: [n_node, input_dim]，一般为 dense tensor
        adj: 邻接矩阵，如果为稀疏矩阵，请传入 torch.sparse_coo_tensor
        """
        # 第一层：线性变换 + 邻接矩阵乘法 + 激活
        x = torch.matmul(inputs, self.weight1)
        if adj.is_sparse:#判断邻接矩阵 adj 是否为稀疏矩阵
            x = torch.sparse.mm(adj, x)#如果是稀疏矩阵，使用稀疏矩阵乘法
        else:
            x = torch.matmul(adj, x)#如果不是稀疏矩阵，使用普通矩阵乘法
        x = self.act(x)

        # 第二层
        x2 = torch.matmul(x, self.weight2)
        if adj.is_sparse:
            x2 = torch.sparse.mm(adj, x2)
        else:
            x2 = torch.matmul(adj, x2)
        outputs = self.act(x2)

        if self.norm_flag:
            outputs = self.bn(outputs)
        return outputs

    def decoder(self, embed):
        """
        对编码结果做内积操作，再经过ReLU激活。
        embed: [n_node, feature_dim]
        输出: 将内积结果 reshape 为 (-1, 1)
        """
        logits = torch.matmul(embed, embed.t())
        logits = logits.view(-1, 1)
        return F.relu(logits)
