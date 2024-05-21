import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch


def node_iter(G):
    if float(nx.__version__[:3])<2.0:
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    if float(nx.__version__[:3])>2.1:
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict

def adj_process(adjs):
    g_num, n_num, n_num = adjs.shape
    adjs = adjs.detach()
    for i in range(g_num):
        adjs[i] += torch.eye(n_num).cuda()
        adjs[i][adjs[i]>0.] = 1.
        degree_matrix = torch.sum(adjs[i], dim=-1, keepdim=False)
        degree_matrix = torch.pow(degree_matrix,-1)
        degree_matrix[degree_matrix == float("inf")] = 0.
        degree_matrix = torch.diag(degree_matrix)
        adjs[i] = torch.mm(degree_matrix, adjs[i])
    return adjs


def NormData(adj):
    adj=adj.tolist()
    adj_norm = normalize_adj(adj )
    adj_norm = adj_norm.toarray()
    #adj = adj + sp.eye(adj.shape[0])
    #adj = adj.toarray()
    #feat = feat.toarray()
    return adj_norm



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# 这个函数的作用就是 返回一个稀疏矩阵的非0值坐标、非0值和整个矩阵的shape
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):  # 判断是否为coo_matrix类型
        sparse_mx = sparse_mx.tocoo()     # 返回稀疏矩阵的coo_matrix形式
    # 这个coo_matrix类型 其实就是系数矩阵的坐标形式：（所有非0元素 （row，col））根据row和col对应的索引对应非0元素在矩阵中的位置
    # 其他位置自动补0
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    # vstack 按垂直方向排列 再转置 则每一行的两个元素组成的元组就是非0元素的坐标
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def write_results_to_file(filename, dataset, auc_avg, auc_std, F1_avg, F1_std):
    """
    将参数、数据集、训练结果写入文件
    """
    with open(filename, 'a+') as f:
        if f.tell() == 0:  # 如果文件为空则写入表头
            f.write('Dataset    AUC_avg     AUC_std     F1_avg      F1_std\n')
        f.write(f'{dataset}     {auc_avg}       {auc_std}       {F1_avg}        {F1_std}\n')


def f1_score_all(y_pred, y_true, sign=1):
    y_true = np.array(y_true)
    if sign == 1:
        total_anomalies = int(np.sum(y_true))
        anomaly_indices = np.argsort(y_pred)[-total_anomalies:]
    else:
        total_anomalies = int(np.sum(1 - y_true))
        anomaly_indices = np.argsort(-y_pred)[-total_anomalies:]

    if sign == 1:
     y_pred_label = np.zeros_like(y_pred)
    else:
     y_pred_label = np.ones_like(y_pred)
    y_pred_label[anomaly_indices] = sign

    # 计算真正例(TP)
    TP = np.sum((y_pred_label == sign) & (y_true == sign))

    # 计算假正例(FP)
    FP = np.sum((y_pred_label == sign) & (y_true == 1 - sign))

    # 计算假反例(FN)
    FN = np.sum((y_pred_label == 1 - sign) & (y_true == sign))

    if TP == 0: return 0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1 = 2 * precision * recall / (precision + recall)

    return f1

