import numpy as np
import torch
import pandas as pd
from scipy.spatial.distance import cdist


def _to_torch_sparse(mat_scipy, device):
    mat = mat_scipy.tocoo()
    indices = np.vstack((mat.row, mat.col))
    indices = torch.LongTensor(indices).to(device)
    values = torch.FloatTensor(mat.data).to(device)
    return torch.sparse_coo_tensor(indices, values, torch.Size(mat.shape)).coalesce()


def Pear_corr(x):
    x = x.T
    x_pd = pd.DataFrame(x)
    dist_mat = x_pd.corr()
    return dist_mat.to_numpy()

def Eu_dis(x):
    dist_mat = cdist(x, x, "euclid")
    print("dist_mat shape:", dist_mat.shape)
    return dist_mat

# def generate_G_from_H(H, variable_weight=False):
#     H = np.array(H)
#     n_edge = H.shape[1]
#     # the weight of the hyperedge, here we use 1 for each hyperedge
#     W = np.ones(n_edge)
#     # the degree of the node
#     DV = np.sum(H * W, axis=1)
#     # the degree of the hyperedge
#     DE = np.sum(H, axis=0)

#     invDE = np.mat(np.diag(np.power(DE, -1)))
#     DV2 = np.mat(np.diag(np.power(DV, -0.5)))
#     W = np.mat(np.diag(W))
#     H = np.mat(H)
#     HT = H.T

#     if variable_weight:
#         DV2_H = DV2 * H
#         invDE_HT_DV2 = invDE * HT * DV2
#         return DV2_H, W, invDE_HT_DV2
#     else:
#         G = DV2 * H * W * invDE * HT * DV2
#         return G

def generate_G_from_H(H, variable_weight=False, edge_weight=None, eps=1e-12):
    """
    H: (N, E) 节点-超边关联矩阵（numpy ndarray）
    variable_weight: 若为 True，返回分解后的三块以便外部做 H @ W @ invDE @ H^T
    edge_weight: None 或形如 (E,) 的一维权重向量
    eps: 防止除零的微小常数
    """
    # --- 规范输入 ---
    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        raise ValueError(f"H must be a non-empty 2D array, got shape {H.shape}")

    N, E = H.shape

    # --- 超边权 ---
    if edge_weight is None:
        w_vec = np.ones(E, dtype=float)  # (E,)
    else:
        w_vec = np.asarray(edge_weight, dtype=float).reshape(-1)
        if w_vec.shape[0] != E:
            raise ValueError(f"edge_weight length {w_vec.shape[0]} != #edges {E}")

    W = np.diag(w_vec)  # (E, E)

    # --- 度 ---
    # 节点度: 每个节点连接的总权重
    DV = H @ w_vec  # (N,)
    # 超边度: 每条超边包含的节点数（或权重和）
    DE = H.sum(axis=0)  # (E,)

    # --- 逆度（对角）带除零保护 ---
    invDE_diag = np.where(DE > 0, 1.0 / DE, 0.0)  # (E,)
    DV_mhalf_diag = np.where(DV > 0, DV**-0.5, 0.0)  # (N,)

    invDE = np.diag(invDE_diag)  # (E, E)
    DV2 = np.diag(DV_mhalf_diag)  # (N, N)

    HT = H.T  # (E, N)

    if variable_weight:
        # 返回分块，供外部自行做乘法（比如稀疏化时更灵活）
        DV2_H = DV2 @ H  # (N, E)
        invDE_HT_DV2 = invDE @ HT @ DV2  # (E, N)
        # W 仍然是 (E,E)
        return DV2_H, W, invDE_HT_DV2
    else:
        # 完整归一化超图传播矩阵
        G = DV2 @ H @ W @ invDE @ HT @ DV2  # (N, N)
        return G


# Concat the modality-specific hyperedges
# def Multi_omics_hyperedge_concat(*H_list):

#     H = None
#     for h in H_list:
#         if h is not None and h != []:
#             # for the first H appended to fused hypergraph incidence matrix
#             if H is None:
#                 H = h
#             else:
#                 if type(h) != list:
#                     H = np.hstack((H, h))
#                 else:
#                     tmp = []
#                     for a, b in zip(H, h):
#                         tmp.append(np.hstack((a, b)))
#                     H = tmp
#     return H
def Multi_omics_hyperedge_concat(*H_list):
    """安全合并多个模态的超边矩阵（numpy ndarray 或可转为 ndarray 的对象）。
    返回 np.ndarray；若无有效 H 返回空 ndarray（size==0）。
    """
    import numpy as np

    valid = []
    for h in H_list:
        if h is None:
            continue
        arr = np.asarray(h)
        if arr.ndim != 2 or arr.size == 0 or arr.shape[1] == 0:
            continue
        valid.append(arr)
    if not valid:
        return np.array([])

    # 行数一致性检查（节点数）
    rows = {a.shape[0] for a in valid}
    if len(rows) != 1:
        raise ValueError(f"无法合并，节点数不一致: {[a.shape for a in valid]}")

    try:
        return np.concatenate(valid, axis=1)  # 按列拼接超边
    except Exception as e:
        raise ValueError(
            f"无法合并 H_list，形状不匹配: {[a.shape for a in valid]}; 原始错误: {e}"
        )

def construct_H_with_KNN_from_distance(
    dis_mat, k_neig, is_probH=True, m_prob=1, edge_type="euclid"
):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))

    if edge_type == "euclid":
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0
            dis_vec = dis_mat[center_idx]

            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            avg_dis = np.average(dis_vec)
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = np.exp(
                        -(dis_vec[node_idx] ** 2) / (m_prob * avg_dis) ** 2
                    )

                else:
                    H[node_idx, center_idx] = 1.0
        print("use euclid for H construction")

    elif edge_type == "pearson":
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = -999.0
            dis_vec = dis_mat[center_idx]
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            nearest_idx = nearest_idx[::-1]

            avg_dis = np.average(dis_vec)
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    H[node_idx, center_idx] = 1.0 - np.exp(
                        -((dis_vec[node_idx] + 1.0) ** 2)
                    )
                else:
                    H[node_idx, center_idx] = 1.0
        print("use pearson for H construction")

    return H

def construct_H_with_KNN(X, K_neigs=[10], is_probH=True, m_prob=1, edge_type="euclid"):
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    if edge_type == "euclid":
        dis_mat = Eu_dis(X)

    elif edge_type == "pearson":
        dis_mat = Pear_corr(X)

    H =  None
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(
            dis_mat.copy(), k_neig, is_probH, m_prob, edge_type
        )
        H = Multi_omics_hyperedge_concat(H, H_tmp)

    return H
