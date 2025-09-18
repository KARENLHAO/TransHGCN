# Transorfomer.py (PyTorch版)
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn import GraphConvolution
from .scMHNN_model import HGNN_unsupervised
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, n_heads, ouput_dim=None):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads

        self.n_heads = n_heads
        if ouput_dim is None:
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

        self.scMHNN_model = HGNN_unsupervised(
            in_ch=self.num_cells, n_hid=128, dropout=0.1
        )
        self.fuse = nn.Linear(128 + 128, 128)   # 融合 HGNN + Transformer
        self.proj = nn.Linear(self.num_cells, 128)  # 把原始特征压到128维
        self.tau = 0.1  # 对比损失的温度参数

        self.adapt_for_gcn = nn.Linear(128, self.num_cells)

        # 实例化图卷积模块；注意这里 input_dim 用的是 num_cells
        self.gcn = GraphConvolution(n_node=self.num_nodes,
                                    input_dim=self.num_cells,
                                    output_dim=128,
                                    dropout=0.1,
                                    act=nn.LeakyReLU(),
                                    norm=False)


        self.fes_encoder = EncoderLayer(128, 2)


    def forward(self, adj, G):
        # 1) 归一化 + 投影到128维
        x = F.normalize(self.exp, p=2, dim=1)        # [num_nodes, num_cells]
        x_t = self.proj(x)                           # [num_nodes, 128]

        # 2) Transformer 编码
        enc_out = self.fes_encoder(x_t)              # [num_nodes, 128]

        # 3) HGNN 编码
        x_ach, x_pos, x_neg = self.scMHNN_model(x, G)  # 三个都是 [num_nodes, 128]

        # 4) 融合
        fused = self.fuse(torch.cat([enc_out, x_ach], dim=1))  # [N,128]
        fused_for_gcn = self.adapt_for_gcn(fused)              # [N,num_cells]

        # 5) GCN + 解码
        final_embedding = self.gcn(fused_for_gcn, adj)
        logits = self.gcn.decoder(final_embedding)

        # 6) 返回 logits + 对比学习三元组
        return logits, (x_ach, x_pos, x_neg)
    
    def _contrastive_loss(self, x_ach, x_pos, x_neg):
    # 简单 InfoNCE：sim(a,p) 对 sim(a,n) 做 softmax
        def sim(a, b):  # 余弦相似
            a = F.normalize(a, p=2, dim=1)
            b = F.normalize(b, p=2, dim=1)
            return torch.sum(a * b, dim=1)

        pos = sim(x_ach, x_pos) / self.tau
        neg = sim(x_ach, x_neg) / self.tau
        logits = torch.stack([pos, neg], dim=1)             # [N, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
    
    def loss_func(self, out, lbl_in, msk_in, neg_msk, triplet=None, lambda_contrast=0.2):
    # 原有重构/链接预测损失（与你之前一致）
        logits = out if triplet is None else out
        error = (logits - lbl_in) ** 2
        mask = msk_in + neg_msk
        base_loss = torch.sqrt(torch.mean(error * mask))

        # 叠加对比损失
        if triplet is not None:
            x_ach, x_pos, x_neg = triplet
            c_loss = self._contrastive_loss(x_ach, x_pos, x_neg)
            return base_loss + lambda_contrast * c_loss, base_loss
        else:
            return base_loss, base_loss