from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score




class load_data():
    def __init__(self, data, normalize=True):#data（数据）和 normalize 布尔值个布尔类型的参数，用于指示是否对数据进行归一化处理
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):#对输入数据进行归一化处理
        standard = StandardScaler()#对数据进行标准化处理
        epr = standard.fit_transform(data.T)#fit：计算输入数据的均值和标准差。transform：基于计算的均值和标准差对数据进行标准化转换，返回标准化后的结果。

        return epr.T


    def exp_data(self):
        data_feature = self.data.values#获取数据

        if self.normalize:#如果需要对数据进行归一化处理
            data_feature = self.data_normalize(data_feature)#进行归一化处理

        data_feature = data_feature.astype(np.float32)#将数据类型转化为float32类型

        return data_feature



def normalize(expression):
    std = StandardScaler()#对数据进行标准化处理
    epr = std.fit_transform(expression)

    return epr

# def adj2saprse_tensor(adj):
#     coo = adj.tocoo()
#     i = torch.LongTensor([coo.row, coo.col])
#     v = torch.from_numpy(coo.data).float()
#
#     adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
#     return adj_sp_tensor


def Evaluation(y_true, y_pred,flag=False):# 用于计算模型预测结果的 AUC、AUPR 和归一化 AUPR 指标
    #AUC衡量分类器预测正负样本的能力。AUPR 评估模型在不同阈值下的精确率和召回率表现。
    if flag:#是一个布尔参数，用于决定如何处理 y_pred 数据。
        y_p = y_pred[:,-1]# True y_pred 的最后一列会被提取并转换为一维数组。
        y_p = y_p.numpy()
        y_p = y_p.flatten()#多维数组或张量转换为一维数组
    else:
        y_p = y_pred.numpy()
        y_p = y_p.flatten()


    y_t = y_true.astype(int)#将y_true转换为整数类型

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)#计算 ROC 曲线下的 AUC 值


    AUPR = average_precision_score(y_true=y_t,y_score=y_p)#计算 AUPR 值
    AUPR_norm = AUPR/np.mean(y_t)#计算归一化 AUPR 值 使得指标更加平衡，尤其在类别不平衡时更有意义。


    return AUC, AUPR, AUPR_norm

def masked_accuracy(preds, labels, mask, negative_mask):
    """
    preds, labels, mask, negative_mask 均为 torch.Tensor，mask 与 negative_mask 需转为 float 进行计算
    """
    #将 preds 和 labels 转换为浮点数类型以确保计算精度。
    preds = preds.float()
    labels = labels.float()
    error = (preds - labels) ** 2#计算预测值与真实值之间的平方误差
    mask = mask.float() + negative_mask.float()#将 mask 和 negative_mask 拼接起来，并转换为浮点数类型。
    loss = torch.sqrt(torch.mean(error * mask))#计算平方误差的均值，并取平方根作为损失。
    return loss


def ROC(outs, labels, test_arr, label_neg):
    #函数 ROC 的主要功能是计算接收者操作特性（ROC）曲线所需的数据。
    scores = []#用于存储正样本和负样本的得分。
    for i in range(len(test_arr)):#遍历 test_arr，根据 labels 索引从 outs 中提取正样本得分并存入 scores。
        l = test_arr[i]
        scores.append(outs[int(labels[l, 0] - 1), int(labels[l, 1] - 1)])

    for i in range(label_neg.shape[0]):
        scores.append(outs[int(label_neg[i, 0]), int(label_neg[i, 1])])#根据 labels_neg 索引从 outs 中提取负样本得分并存入 scores。
    #创建正样本标签 test_labels 和负样本标签 temp，并将两者垂直堆叠为 test_labels1
    test_labels = np.ones((len(test_arr), 1))
    temp = np.zeros((label_neg.shape[0], 1))
    test_labels1 = np.vstack((test_labels, temp))
    test_labels1 = np.array(test_labels1, dtype=np.bool).reshape([-1, 1])#将 test_labels1 转换为布尔类型，并将其重塑为一列的数组。具体作用如下：
    return test_labels1, scores


