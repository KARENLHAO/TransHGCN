# Transorfomer.py (PyTorch版)
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import GraphConvolution
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads

        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)


    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)

        # G_binary = torch.where(G > 0, torch.tensor(1), torch.tensor(0))

        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)

        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = self.attn(X)
        # X = self.AN1(output + X)
        #
        # output = self.l1(X)
        # X = self.AN2(output + X)

        return output



class TransorfomerModel(nn.Module):
    def __init__(self, exp, do_train=False):
        """
        exp: 基因表达矩阵，需为 torch.Tensor，shape = [num_nodes, num_cells]
        do_train: 标记是否处于训练模式（此处仅做示例区分）
        """
        super(TransorfomerModel, self).__init__()
        self.do_train = do_train
        self.num_nodes = exp.shape[0]
        self.num_cells = exp.shape[1]
        self.entry_size = self.num_nodes ** 2
        self.exp = exp  # 固定的基因表达数据

        # 实例化图卷积模块；注意这里 input_dim 用的是 num_cells
        self.gcn = GraphConvolution(n_node=self.num_nodes,
                                    input_dim=self.num_cells,
                                    output_dim=128,
                                    dropout=0.1,
                                    act=nn.LeakyReLU(),
                                    norm=False)


        self.fes_encoder = EncoderLayer(758, 2)


    def forward(self, adj):
        """
        adj: 邻接矩阵，应为稀疏 tensor（torch.sparse_coo_tensor）
        """
        # 对表达数据进行 L2 归一化
        embedding_genes = F.normalize(self.exp, p=2, dim=1)
        new_embedding_genes = self.fes_encoder(embedding_genes)
        final_embedding = self.gcn(new_embedding_genes, adj)
        logits = self.gcn.decoder(final_embedding)

        return logits

    def loss_func(self, logits, lbl_in, msk_in, neg_msk):
        """
        模仿原有 masked_accuracy：计算 (预测值 - 标签) 平方误差后加权平均，再取平方根
        logits, lbl_in, msk_in, neg_msk 均为 tensor
        """
        error = (logits - lbl_in) ** 2
        mask = msk_in + neg_msk
        loss = torch.sqrt(torch.mean(error * mask))
        # 此处 accuracy 与 loss 取值相同，仅作为示例
        return loss, loss
