# main.py (PyTorch版，支持GPU)
import torch
import numpy as np

import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from inits import load_data1, preprocess_graph  # 数据加载、预处理函数保持不变
from model.Transorfomer import TransorfomerModel  # 已转换为 PyTorch 的模型
from utils import masked_accuracy

epochs = 700
batch_size = 64



def evaluate(rel_test_label, pre_test_label):
    """
    计算 ROC 和 PR 曲线下面积。
    rel_test_label: numpy 数组，形状 [n_sample, 3]，其中前两列为索引，第三列为真实标签
    pre_test_label: numpy 数组，形状 [num_nodes, num_nodes]，预测的分数矩阵
    rel_test_label用于提供测试数据的真实标签信息,并与预测分数矩阵 pre_test_label 进行对比，以计算 ROC 和 PR 曲线下面积。
    """
    temp_pre = []
    for i in range(rel_test_label.shape[0]):
        # 提取当前行的前两列作为索引m和n。
        m = int(rel_test_label[i, 0])
        n = int(rel_test_label[i, 1])
        temp_pre.append([m, n, pre_test_label[m, n]])
    temp_pre = np.array(
        temp_pre
    )  # 并从 pre_test_label 中获取对应的预测分数，构造新的数组 temp_pre。

    prec, rec, _ = precision_recall_curve(
        rel_test_label[:, 2], temp_pre[:, 2]
    )  # 计算 PR 曲线的精度和召回率，并计算 PR 曲线下面积 (AUPR)。
    aupr_val = auc(rec, prec)

    fpr, tpr, _ = roc_curve(
        rel_test_label[:, 2], temp_pre[:, 2]
    )  # 计算 ROC 曲线的假阳性率和真阳性率，并计算 ROC 曲线下面积 (AUC)。
    auc_val = auc(fpr, tpr)

    return auc_val, aupr_val, fpr, tpr, rec, prec


def main(device):
    # -------------------- 数据加载与预处理 --------------------
    # load_data1 返回：geneName, feature, logits_train, logits_test, logits_validation,
    # train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation, train_data, validation_data, test_data

    # geneName: 节点名称
    # feature: 节点特征
    # logits_train: 训练集标签 logits 预测分数
    # logits_test: 测试集标签
    # logits_validation: 验证集标签
    # train_mask: 训练集标签 mask筛选特定的数据子集 分别用于标记训练集、测试集和验证集中的有效样本。作用是通过布尔值（True/False）选择性地应用操作到指定的样本上。
    # test_mask: 测试集标签
    # validation_mask: 验证集标签
    # interaction: 邻接矩阵
    # neg_logits_train: 训练集负样本标签 用于在训练阶段计算损失时区分正负样本。
    # neg_logits_test: 测试集负样本标签  用于在测试阶段计算损失时区分正负样本。

    (
        geneName,
        geneNum,
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
    ) = load_data1()

    # 预处理邻接矩阵，得到稀疏矩阵 tuple（coords, values, shape）
    bias_tuple = preprocess_graph(interaction)
    coords, values, shape = bias_tuple
    # 将邻接矩阵转换为 PyTorch 稀疏 tensor
    indices = torch.LongTensor(coords.T).to(device)
    values = torch.FloatTensor(values).to(device)
    adj = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(device)

    # 将 numpy 数据转换为 torch.Tensor，并迁移到 device 上
    feature = torch.FloatTensor(feature).to(device)
    logits_train = torch.FloatTensor(logits_train).to(device)
    logits_test = torch.FloatTensor(logits_test).to(device)
    logits_validation = torch.FloatTensor(logits_validation).to(device)
    # mask 转为 BoolTensor（后续在 loss 计算中转换为 float）
    train_mask = torch.BoolTensor(train_mask).to(device)
    test_mask = torch.BoolTensor(test_mask).to(device)
    validation_mask = torch.BoolTensor(validation_mask).to(device)
    neg_logits_train = torch.FloatTensor(neg_logits_train).to(device)
    neg_logits_test = torch.FloatTensor(neg_logits_test).to(device)
    neg_logits_validation = torch.FloatTensor(neg_logits_validation).to(device)

    # -------------------- 模型构建 --------------------
    # 这里 TransorfomerModel 的构造函数中接收的 exp 参数为 feature
    model = TransorfomerModel(feature, do_train=False).to(device)
    model.train()  # 训练模式

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001
    )  # weight_decay=0.0005 可根据需求添加

    # -------------------- 训练与验证 --------------------
    for epoch in range(epochs):
        t_start = time.time()
        optimizer.zero_grad()

        # 前向传播（训练阶段）
        logits = model(adj)  # 输出形状为 [entry_size, 1]

        # 计算训练损失（masked_accuracy作为损失函数，跟原代码保持一致）
        loss = masked_accuracy(logits, logits_train, train_mask, neg_logits_train)
        loss.backward()
        optimizer.step()

        train_loss_avg = loss.item()  # 示例中仅计算一次

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_logits = model(adj)  # 获取验证集的预测结果
            val_loss = masked_accuracy(
                val_logits, logits_validation, validation_mask, neg_logits_validation
            )
            # 将验证输出 reshape 成 [num_nodes, num_nodes] val_score，以便后续评估。
            val_score = (
                val_logits.view(feature.shape[0], feature.shape[0]).cpu().numpy()
            )
        # 使用 evaluate 函数计算 AUC 和 AUPR
        auc_val, aupr_val, _, _, _, _ = evaluate(validation_data, val_score)
        t_elapsed = time.time() - t_start
        print(
            "Epoch: {:04d} | Train Loss: {:.5f} | Val Loss: {:.5f} | AUC: {:.5f} | AUPR: {:.5f} | Time: {:.5f}s".format(
                epoch, train_loss_avg, val_loss.item(), auc_val, aupr_val, t_elapsed
            )
        )
        model.train()

    print("Finish training.")

    # -------------------- 测试阶段 --------------------
    model.eval()
    with torch.no_grad():
        test_logits = model(adj)
        test_loss = masked_accuracy(
            test_logits, logits_test, test_mask, neg_logits_test
        )
        test_score = test_logits.view(feature.shape[0], feature.shape[0]).cpu().numpy()
    print("Test Loss: {:.5f}".format(test_loss.item()))

    # 返回 geneName, 预测矩阵, 以及 test_data（关系数据，用于后续评价）
    return geneName, test_score, test_data


if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 设置设备：如果有 GPU 则使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    T = 1
    cv_num = 1
    aupr_vec = []
    for t in range(T):
        for i in range(cv_num):
            t_start = time.time()
            geneName, pre_test_label, rel_test_label = main(device)
            print("Run time: {:.4f}s".format(time.time() - t_start))

            # 计算评价指标
            auc_val, aupr_val, fpr, tpr, rec, prec = evaluate(
                rel_test_label, pre_test_label
            )
            aupr_vec.append(aupr_val)
            print("auc: {:.6f}, aupr: {:.6f}".format(auc_val, aupr_val))

            # 可视化 ROC 曲线
            plt.figure()
            plt.plot(fpr, tpr, label="ROC (AUC = {:.4f})".format(auc_val))
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.show()

            # 可视化 Precision-Recall 曲线
            plt.figure()
            plt.plot(rec, prec, label="PR (AUPR = {:.4f})".format(aupr_val))
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.show()



# # main.py (PyTorch版，支持GPU)
# import torch
# import numpy as np
# # import pandas as pd
# import time
# import matplotlib.pyplot as plt
# import argparse
# from sklearn.metrics import precision_recall_curve, roc_curve, auc

# from inits import load_data1, preprocess_graph  # 数据加载、预处理函数保持不变
# # from model.Transorfomer import TransorfomerModel  # 已转换为 PyTorch 的模型
# from utils import masked_accuracy
# from model.scMHNN_model import HGNN_unsupervised




# epochs =700
# batch_size = 64

# parser = argparse.ArgumentParser()
# parser.add_argument("--n_hid", type=int, default=128, help="dimension of hidden layer")
# parser.add_argument("--drop_out", type=float, default=0.1, help="dropout rate")
# args = parser.parse_args()


# def evaluate(rel_test_label, pre_test_label):
#     """
#     计算 ROC 和 PR 曲线下面积。
#     rel_test_label: numpy 数组，形状 [n_sample, 3]，其中前两列为索引，第三列为真实标签
#     pre_test_label: numpy 数组，形状 [num_nodes, num_nodes]，预测的分数矩阵
#     rel_test_label用于提供测试数据的真实标签信息,并与预测分数矩阵 pre_test_label 进行对比，以计算 ROC 和 PR 曲线下面积。
#     """
#     temp_pre = []
#     for i in range(rel_test_label.shape[0]):
#         #提取当前行的前两列作为索引m和n。
#         m = int(rel_test_label[i, 0])
#         n = int(rel_test_label[i, 1])
#         temp_pre.append([m, n, pre_test_label[m, n]])
#     temp_pre = np.array(temp_pre)#并从 pre_test_label 中获取对应的预测分数，构造新的数组 temp_pre。

#     prec, rec, _ = precision_recall_curve(rel_test_label[:, 2], temp_pre[:, 2])#计算 PR 曲线的精度和召回率，并计算 PR 曲线下面积 (AUPR)。
#     aupr_val = auc(rec, prec)

#     fpr, tpr, _ = roc_curve(rel_test_label[:, 2], temp_pre[:, 2])#计算 ROC 曲线的假阳性率和真阳性率，并计算 ROC 曲线下面积 (AUC)。
#     auc_val = auc(fpr, tpr)

#     return auc_val, aupr_val, fpr, tpr, rec, prec

# def main(device):
#     # -------------------- 数据加载与预处理 --------------------
#     # load_data1 返回：geneName, feature, logits_train, logits_test, logits_validation,
#     # train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation, train_data, validation_data, test_data
    
#     #geneName: 节点名称
#     #feature: 节点特征
#     #logits_train: 训练集标签 logits 预测分数
#     #logits_test: 测试集标签
#     #logits_validation: 验证集标签
#     #train_mask: 训练集标签 mask筛选特定的数据子集 分别用于标记训练集、测试集和验证集中的有效样本。作用是通过布尔值（True/False）选择性地应用操作到指定的样本上。
#     #test_mask: 测试集标签
#     #validation_mask: 验证集标签
#     #interaction: 邻接矩阵
#     #neg_logits_train: 训练集负样本标签 用于在训练阶段计算损失时区分正负样本。
#     #neg_logits_test: 测试集负样本标签  用于在测试阶段计算损失时区分正负样本。
  
#     (geneName, feature, logits_train, logits_test, logits_validation,
#      train_mask, test_mask, validation_mask, interaction,
#      neg_logits_train, neg_logits_test, neg_logits_validation,
#      train_data, validation_data, test_data) = load_data1()

#     # 预处理邻接矩阵，得到稀疏矩阵 tuple（coords, values, shape）
#     bias_tuple = preprocess_graph(interaction)
#     coords, values, shape = bias_tuple
#     # 将邻接矩阵转换为 PyTorch 稀疏 tensor
#     indices = torch.LongTensor(coords.T).to(device)
#     values = torch.FloatTensor(values).to(device)
#     adj = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to(device)

#     # 将 numpy 数据转换为 torch.Tensor，并迁移到 device 上
#     feature = torch.FloatTensor(feature).to(device)
#     logits_train = torch.FloatTensor(logits_train).to(device)
#     logits_test = torch.FloatTensor(logits_test).to(device)
#     logits_validation = torch.FloatTensor(logits_validation).to(device)
#     # mask 转为 BoolTensor（后续在 loss 计算中转换为 float）
#     train_mask = torch.BoolTensor(train_mask).to(device)
#     test_mask = torch.BoolTensor(test_mask).to(device)
#     validation_mask = torch.BoolTensor(validation_mask).to(device)
#     neg_logits_train = torch.FloatTensor(neg_logits_train).to(device)
#     neg_logits_test = torch.FloatTensor(neg_logits_test).to(device)
#     neg_logits_validation = torch.FloatTensor(neg_logits_validation).to(device)

#     # -------------------- 模型构建 --------------------
#     # 这里 TransorfomerModel 的构造函数中接收的 exp 参数为 feature
#     model = HGNN_unsupervised(in_ch=feature.size(1), n_hid=args.n_hid, dropout=args.drop_out).to(device)
#     model.train()  # 训练模式

#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # weight_decay=0.0005 可根据需求添加

#     # -------------------- 训练与验证 --------------------
#     for epoch in range(epochs):
#         t_start = time.time()
#         optimizer.zero_grad()
        

#         # 前向传播（训练阶段）
#         # logits = model(feature, adj)  # 输出形状为 [entry_size, 1]
#         # x_ach, x_pos, x_neg = model(feature, adj)
#         # logits = torch.matmul(x_ach, x_pos.t())
#         # logits = logits.view(-1, 1)
#         x_ach, x_pos, x_neg = model(feature, adj)
#         emb = x_ach                                        # 统一用结构分支
#         logits = (emb @ emb.T).reshape(-1, 1) 
#         # 计算训练损失（masked_accuracy作为损失函数，跟原代码保持一致）
#         loss = masked_accuracy(logits, logits_train, train_mask, neg_logits_train)
#         loss.backward()
#         optimizer.step()

#         train_loss_avg = loss.item()  # 示例中仅计算一次

#         # 验证阶段
#         model.eval()
#         with torch.no_grad():
#             # N = x_ach.shape[0]
#             # x_ach, x_pos, x_neg = model(feature, adj)#获取验证集的预测结果
#             # val_logits = (x_ach @ x_ach.t()).reshape(N * N, 1)
#             x_ach, x_pos, x_neg = model(feature, adj)
#             emb = x_ach
#             val_logits = (emb @ emb.T).reshape(-1, 1)
#             val_loss = masked_accuracy(val_logits, logits_validation, validation_mask, neg_logits_validation)
#             # 将验证输出 reshape 成 [num_nodes, num_nodes] val_score，以便后续评估。
#             val_score = val_logits.view(feature.shape[0], feature.shape[0]).cpu().numpy()
#         # 使用 evaluate 函数计算 AUC 和 AUPR
#         auc_val, aupr_val, _, _, _, _ = evaluate(validation_data, val_score)
#         t_elapsed = time.time() - t_start
#         print("Epoch: {:04d} | Train Loss: {:.5f} | Val Loss: {:.5f} | AUC: {:.5f} | AUPR: {:.5f} | Time: {:.5f}s".format(
#             epoch, train_loss_avg, val_loss.item(), auc_val, aupr_val, t_elapsed))
#         model.train()

#     print("Finish training.")

#     # -------------------- 测试阶段 --------------------
#     model.eval()
#     with torch.no_grad():
#         # x_ach, x_pos, x_neg = model(feature, adj)
#         # N = feature.size(0)
#         # test_logits = (x_ach @ x_ach.t()).reshape(N*N, 1)
#         x_ach, x_pos, x_neg = model(feature, adj)
#         emb = x_ach
#         test_logits = (emb @ emb.T).reshape(-1, 1) 
#         test_loss = masked_accuracy(test_logits, logits_test, test_mask, neg_logits_test)
#         test_score = test_logits.view(feature.shape[0], feature.shape[0]).cpu().numpy()
#     print('Test Loss: {:.5f}'.format(test_loss.item()))

#     # 返回 geneName, 预测矩阵, 以及 test_data（关系数据，用于后续评价）
#     return geneName, test_score, test_data

# if __name__ == '__main__':
#     # 设置随机种子，确保结果可复现
#     seed = 123
#     np.random.seed(seed)
#     torch.manual_seed(seed)

#     # 设置设备：如果有 GPU 则使用 GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Using device:", device)

#     T = 1
#     cv_num = 1
#     aupr_vec = []
#     for t in range(T):
#         for i in range(cv_num):
#             t_start = time.time()
#             geneName, pre_test_label, rel_test_label = main(device)
#             print("Run time: {:.4f}s".format(time.time() - t_start))

#             # 计算评价指标
#             auc_val, aupr_val, fpr, tpr, rec, prec = evaluate(rel_test_label, pre_test_label)
#             aupr_vec.append(aupr_val)
#             print("auc: {:.6f}, aupr: {:.6f}".format(auc_val, aupr_val))

#             # 可视化 ROC 曲线
#             plt.figure()
#             plt.plot(fpr, tpr, label="ROC (AUC = {:.4f})".format(auc_val))
#             plt.xlabel("False Positive Rate")
#             plt.ylabel("True Positive Rate")
#             plt.title("ROC Curve")
#             plt.legend()
#             plt.show()

#             # 可视化 Precision-Recall 曲线
#             plt.figure()
#             plt.plot(rec, prec, label="PR (AUPR = {:.4f})".format(aupr_val))
#             plt.xlabel("Recall")
#             plt.ylabel("Precision")
#             plt.title("Precision-Recall Curve")
#             plt.legend()
#             plt.show()


