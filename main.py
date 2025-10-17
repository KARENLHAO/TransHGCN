import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from inits import load_data1, preprocess_graph
from model.TransHGCN import TransHGCNModel
from hypergraph_construction import construct_H_with_KNN, generate_G_from_H



epochs = 800
batch_size = 64

parser = argparse.ArgumentParser()
parser.add_argument("--n_hid", type=int, default=128, help="dimension of hidden layer")
parser.add_argument("--drop_out", type=float, default=0.1, help="dropout rate")
parser.add_argument("--edge_type", type=str, default="euclid", help="euclid or pearson")
parser.add_argument( "--is_probH", type=bool, default=False, help="prob edge True or False")
parser.add_argument("--m_prob", type=float, default=1.0, help="m_prob")
parser.add_argument("--K_neigs", type=int, default=5, help="K_neigs")
args = parser.parse_args()


def evaluate(rel_test_label, pre_test_label):
    """
    计算 ROC 和 PR 曲线下面积。
    rel_test_label: numpy 数组，形状 [n_sample, 3]，其中前两列为索引，第三列为真实标签
    pre_test_label: numpy 数组，形状 [num_nodes, num_nodes]
    """
    y_true = rel_test_label[:, 2]
    idx = rel_test_label[:, :2].astype(int)
    y_score = pre_test_label[idx[:, 0], idx[:, 1]]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return auc(fpr, tpr), auc(recall, precision), fpr, tpr, recall, precision


def main(device):
    # -------------------- 数据加载与预处理 --------------------
    (
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
    ) = load_data1()

    # 邻接 -> 稀疏张量 adj
    bias_tuple = preprocess_graph(interaction)
    coords, values, shape = bias_tuple
    indices = torch.LongTensor(coords.T).to(device)
    values = torch.FloatTensor(values).to(device)
    adj = (
        torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        .coalesce()
        .to(device)
    )

    # === 超图 G ===
    # H = build_H_from_knn(exp=pd.DataFrame(feature), k=5, self_loop=True)
    H = construct_H_with_KNN(
        X=feature,
        K_neigs=args.K_neigs,
        is_probH=args.is_probH,
        m_prob=args.m_prob,
        edge_type=args.edge_type,
    )
    G = generate_G_from_H(H)
    
    

    # numpy -> torch
    G = torch.FloatTensor(G).to(device)
    feature = torch.FloatTensor(feature).to(device)
    logits_train = torch.FloatTensor(logits_train).to(device)
    logits_test = torch.FloatTensor(logits_test).to(device)
    logits_validation = torch.FloatTensor(logits_validation).to(device)

    train_mask = torch.BoolTensor(train_mask).to(device)
    test_mask = torch.BoolTensor(test_mask).to(device)
    validation_mask = torch.BoolTensor(validation_mask).to(device)

    neg_logits_train = torch.FloatTensor(neg_logits_train).to(device)
    neg_logits_test = torch.FloatTensor(neg_logits_test).to(device)
    neg_logits_validation = torch.FloatTensor(neg_logits_validation).to(device)

    # -------------------- 模型构建 --------------------
    model = TransHGCNModel(feature, do_train=True).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # -------------------- 训练与验证 --------------------
    for epoch in range(epochs):
        t_start = time.time()
        optimizer.zero_grad()

        # === 前向：现在返回 logits 和对比三元组 ===
        logits= model(adj, G)

        # === 融合损失：重构(base) + 对比 ===
        loss= model.loss_func(
            logits,
            lbl_in=logits_train,
            msk_in=train_mask.float(),
            neg_msk=neg_logits_train,
            triplet=None,
            lambda_contrast=0.2,
        )
        loss.backward()
        optimizer.step()
        

       
        # ma = masked_accuracy(logits, logits_train, train_mask, neg_logits_train)

        # 验证
        model.eval()
        with torch.no_grad():
            val_logits = model(adj, G)
            val_loss = model.loss_func(
                val_logits,
                lbl_in=logits_validation,
                msk_in=validation_mask.float(),
                neg_msk=neg_logits_validation,
                triplet=None,
            )
            val_score = (
                val_logits.view(feature.shape[0], feature.shape[0]).cpu().numpy()
            )

        auc_val, aupr_val, _, _, _, _ = evaluate(validation_data, val_score)
        t_elapsed = time.time() - t_start
        print(
            f"Epoch: {epoch:04d} | Train Loss: {loss.item():.5f} "
            f" Val Loss: {val_loss.item():.5f} "
            f"| AUC: {auc_val:.5f} | AUPR: {aupr_val:.5f} | Time: {t_elapsed:.5f}s"
        )
        model.train()

    print("Finish training.")

    # -------------------- 测试 --------------------
    model.eval()
    with torch.no_grad():
        test_logits = model(adj, G)
        # 如需保持和旧代码一致的指标，也可用 masked_accuracy：
        test_loss= model.loss_func(
            test_logits,
            lbl_in=logits_test,
            msk_in=test_mask.float(),
            neg_msk=neg_logits_test,
            triplet=None,
        )
        test_score = test_logits.view(feature.shape[0], feature.shape[0]).cpu().numpy()
    print("Test Loss: {:.5f}".format(test_loss.item()))

    return geneName, test_score, test_data, test_loss


if __name__ == "__main__":
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 在循环外先准备容器
roc_list = []  # 每个元素: {"fpr":..., "tpr":..., "auc":..., "label":...}
pr_list = []  # 每个元素: {"rec":..., "prec":..., "aupr":..., "label":...}


T = 1
cv_num = 1
auc_vec = []
aupr_vec = []

for t in range(T):
    for i in range(cv_num):
        t_start = time.time()
        geneName, pre_test_label, rel_test_label, test_loss = main(device)
        run_time = time.time() - t_start
        print("Run time: {:.4f}s".format(run_time))

        auc_val, aupr_val, fpr, tpr, rec, prec = evaluate(
            rel_test_label, pre_test_label
        )
        auc_vec.append(auc_val)
        aupr_vec.append(aupr_val)
        print("auc: {:.6f}, aupr: {:.6f}".format(auc_val, aupr_val))

        

        # 保存当前 run 的曲线与数值，便于统一绘图
        run_id = t * cv_num + i + 1
        roc_list.append(
            {"fpr": fpr, "tpr": tpr, "auc": auc_val, "label": f"Run {run_id}"}
        )
        pr_list.append(
            {"rec": rec, "prec": prec, "aupr": aupr_val, "label": f"Run {run_id}"}
        )


# ====== 统一绘图：ROC ======
plt.figure(figsize=(8, 6))
for item in roc_list:
    plt.plot(
        item["fpr"],
        item["tpr"],
        lw=1.8,
        label=f"{item['label']} (AUC={item['auc']:.6f})",
    )
# 参考线
plt.plot([0, 1], [0, 1], "--", lw=1, color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC")
plt.legend()
plt.tight_layout()
plt.savefig("/workspaces/GNNLink-master/image/baseline_AUROC.png", dpi=200)

# ====== 统一绘图：PR ======
plt.figure(figsize=(8, 6))
for item in pr_list:
    plt.plot(
        item["rec"],
        item["prec"],
        lw=1.8,
        label=f"{item['label']} (AUPR={item['aupr']:.6f})",
    )
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("AUPRC")
plt.legend()
plt.tight_layout()
plt.savefig("/workspaces/GNNLink-master/image/baseline_AUPRC.png", dpi=200)
plt.show()
