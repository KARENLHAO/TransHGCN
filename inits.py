import numpy as np
import pandas as pd
from utils import load_data
import scipy.sparse as sp




def adj_to_bias(adj, sizes, nhood = 1):#输入参数包括邻接矩阵 adj、图大小列表 sizes 和邻居阶数 nhood。
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)#初始化一个空数组 mt，用于存储处理后的邻接矩阵。
    # 遍历每个图，将邻接矩阵转换为对角线为 1 的矩阵，并保存到 mt 中。
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        #初始化单位矩阵，并通过矩阵乘法扩展到指定的邻居阶数。
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        #遍历每个节点，如果邻接矩阵的对应位置大于 0，则将对应位置的值设为 1。
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0

    return -1e9 * (1.0-mt)



def load_data1():
    exp_file = 'Data/mESC/TFs+1000/BL--ExpressionData.csv'  # .../


    data_input = pd.read_csv(exp_file, index_col=0)
    geneName = data_input.index
    loader = load_data(data_input)#
    feature = loader.exp_data()
    geneNum = feature.shape[0]

    


    train_file = 'Data/Train_validation_test/mESC 1000/Train_set.csv'  # .../Demo/
    test_file = 'Data/Train_validation_test/mESC 1000/Test_set.csv'
    val_file = 'Data/Train_validation_test/mESC 1000/Validation_set.csv'
    train_data = pd.read_csv(train_file, index_col=0).values
    validation_data = pd.read_csv(val_file, index_col=0).values
    test_data = pd.read_csv(test_file, index_col=0).values

    train_data = train_data[np.lexsort(-train_data.T)]
    train_index = np.sum(train_data[:,2])

    validation_data = validation_data[np.lexsort(-validation_data.T)]
    validation_index = np.sum(validation_data[:, 2])

    test_data = test_data[np.lexsort(-test_data.T)]
    test_index = np.sum(test_data[:, 2])


    logits_train = sp.csr_matrix((train_data[0:train_index,2], (train_data[0:train_index,0] , train_data[0:train_index,1])),shape=(geneNum, geneNum)).toarray()
    neg_logits_train = sp.csr_matrix((np.ones(train_data[train_index:, 2].shape), (train_data[train_index:, 0], train_data[train_index:, 1])),
                                 shape=(geneNum, geneNum)).toarray()
    interaction = logits_train
    interaction = interaction + np.eye(interaction.shape[0])
    interaction = sp.csr_matrix(interaction)
    logits_train = logits_train.reshape([-1, 1])
    neg_logits_train = neg_logits_train.reshape([-1, 1])

    logits_test = sp.csr_matrix((test_data[0:test_index, 2], (test_data[0:test_index, 0], test_data[0:test_index, 1] )),
                                 shape=(geneNum, geneNum)).toarray()
    neg_logits_test = sp.csr_matrix((np.ones(test_data[test_index:, 2].shape), (test_data[test_index:, 0], test_data[test_index:, 1])),
                                shape=(geneNum, geneNum)).toarray()
    logits_test = logits_test.reshape([-1, 1])
    neg_logits_test = neg_logits_test.reshape([-1, 1])
    logits_validation = sp.csr_matrix((validation_data[0:validation_index, 2], (validation_data[0:validation_index, 0], validation_data[0:validation_index, 1])),
                               shape=(geneNum, geneNum)).toarray()
    neg_logits_validation = sp.csr_matrix(
        (np.ones(validation_data[validation_index:, 2].shape), (validation_data[validation_index:, 0], validation_data[validation_index:, 1])),
        shape=(geneNum, geneNum)).toarray()
    logits_validation = logits_validation.reshape([-1, 1])
    neg_logits_validation = neg_logits_validation.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=np.bool_).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool_).reshape([-1, 1])
    validation_mask = np.array(logits_validation[:, 0], dtype=np.bool_).reshape([-1, 1])

    return (
        geneName,
        feature,
        
        logits_train,
        logits_test,
        logits_validation,
        train_mask,
        test_mask,
        validation_mask,
        interaction,
        neg_logits_train,
        neg_logits_test,
        neg_logits_validation,
        train_data,
        validation_data,
        test_data,
    )

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)#将输入的邻接矩阵 adj 转换为稀疏 COO 格式。
    adj_ = adj + sp.eye(adj.shape[0])# 将对角线元素设置为 1，表示节点与自己相连。
    rowsum = np.array(adj_.sum(1))# 计算每个节点的入度，即节点的度数。并生成度矩阵的负平方根对角矩阵
    #对邻接矩阵进行归一化处理：先用 degree_mat_inv_sqrt 左乘 adj_，再右乘 degree_mat_inv_sqrt，并将结果转换为 COO 格式。
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    adj_normalized = adj_normalized.dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):#将稀疏矩阵转换为一个元组，包含坐标、值和形状信息
    if not sp.isspmatrix_coo(sparse_mx):#检查输入稀疏矩阵是否为 COO 格式，如果不是则转换为 COO 格式。
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()#提取稀疏矩阵的行索引和列索引，并将其组合为坐标数组。
    values = sparse_mx.data#提取稀疏矩阵的非零值。
    shape = sparse_mx.shape#获取稀疏矩阵的形状。
    return coords, values, shape

