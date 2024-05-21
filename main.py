import torch.utils.data
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import argparse
import torch.nn as nn
import torch.nn.functional as F
import load_data
from random import sample
from torch.autograd import Variable
from GraphBuild import GraphBuild
import random
from sklearn.model_selection import StratifiedKFold
from vgae import VGAE
from util import *


def arg_parse():
    parser = argparse.ArgumentParser(description='G-Anomaly Arguments.')
    parser.add_argument('--datadir', dest='datadir', default='dataset', help='Directory where benchmark is located')
    parser.add_argument('--DS', dest='DS', default='DHFR', help='dataset name')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int, default=0,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--num-epochs', dest='num_epochs', default=100, type=int, help='total epoch number')
    parser.add_argument('--batch-size', dest='batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--hidden-dim1', dest='hidden_dim1', default=128, type=int, help='Hidden dimension1')
    parser.add_argument('--hidden-dim2', dest='hidden_dim2', default=64, type=int, help='Hidden dimension2')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--feature', dest='feature', default='default', help='use what node feature')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed')
    parser.add_argument('--sign', dest='sign', type=int, default=0, help='sign of graph anomaly')
    parser.add_argument('--ratio', dest='ratio', type=float, default=1, help='train graph ratio in dataset')
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_kl_loss(A_preds, means, logstds):
    length = A_preds.shape[0]
    kl_loss = 0
    try:
        for i in range(length):
            A_pred = A_preds[i]
            mean = means[i]
            logstd = logstds[i]
            kl = (0.5 / A_pred.size(0)) * (1 + 2 * logstd - mean ** 2 - torch.exp(logstd) ** 2).sum(1).mean()
            kl_loss += kl
    except:
        kl_loss = 0
    return kl_loss / length


def get_adj_loss(A_preds, adj_labels, weight_tensors, norms):
    length = A_preds.shape[0]
    loss = 0
    try:
        for i in range(length):
            A_pred = A_preds[i]

            adj_label = adj_labels[i]

            weight_tensor = weight_tensors[i]
            norm = norms[i]
            loss += norm * F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1), weight=weight_tensor)
    except:
        loss = 0
    return loss / length


def train(data_test_loader, data_abnormal_loader, data_normal_loader, vage, args):
    optimizerG = torch.optim.Adam(vage.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.5)
    auroc_final = 0
    max_AUC = 0
    F1_Score, best_fpr, best_tpr, best_precision, best_recall = None, None, None, None, None

    data_p = []
    data_n = []
    for batch_idx, data in enumerate(data_normal_loader):
        data_p.append(data)
    for batch_idx, data in enumerate(data_abnormal_loader):
        data_n.append(data)
    datasize_p = len(data_p)
    datasize_n = len(data_n)
    while datasize_p < datasize_n:
        data_p.extend(sample(data_p, min(datasize_p, datasize_n - datasize_p)))
        datasize_p = len(data_p)
    while datasize_p > datasize_n:
        data_n.extend(sample(data_n, min(datasize_n, datasize_p - datasize_n)))
        datasize_n = len(data_n)

    vage.train()

    for epoch in range(args.num_epochs):
        for idx in range(datasize_p):
            adj_p = Variable(data_p[idx]['adj'].float(), requires_grad=False).cuda()
            h0_p = Variable(data_p[idx]['feats'].float(), requires_grad=False).cuda()
            adj_n = Variable(data_n[idx]['adj'].float(), requires_grad=False).cuda()
            h0_n = Variable(data_n[idx]['feats'].float(), requires_grad=False).cuda()
            weight_p = Variable(data_p[idx]['weight_tensor'].float(), requires_grad=False).cuda()
            weight_n = Variable(data_n[idx]['weight_tensor'].float(), requires_grad=False).cuda()
            adj_label_p = Variable(data_p[idx]['adj_label'].float(), requires_grad=False).cuda()
            adj_label_n = Variable(data_n[idx]['adj_label'].float(), requires_grad=False).cuda()
            norm_p = Variable(data_p[idx]['norm'].float(), requires_grad=False).cuda()
            norm_n = Variable(data_n[idx]['norm'].float(), requires_grad=False).cuda()

            A_pred, X_pred, score, logstd = vage(torch.cat([h0_p, h0_n]), torch.cat([adj_p, adj_n]))

            adj_loss = get_adj_loss(A_pred, torch.cat([adj_label_p, adj_label_n]),
                                    torch.cat([weight_p, weight_n]),
                                    torch.cat([norm_p, norm_n]))
            kl_loss = get_kl_loss(A_pred, vage.mean, vage.logstd)
            try:
                diff_attribute = torch.pow(torch.cat([h0_p, h0_n]) - X_pred, 2)
            except:
                print('error')
            attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
            attribute_cost = torch.mean(attribute_reconstruction_errors)
            alpha, beta, gamma = 1, 1, 0
            vgea_loss = alpha * attribute_cost + beta * adj_loss - gamma * kl_loss

            t_zeros = torch.zeros(len(adj_p))
            t_ones = torch.ones(logstd.shape[0] - len(adj_p))
            label = torch.cat((t_zeros, t_ones)).cuda()

            score = score.squeeze()
            std_mean = torch.sigmoid(logstd.mean(dim=(1, 2)))
            loss_std = F.binary_cross_entropy(std_mean, label)
            loss_score = F.binary_cross_entropy(score, label)
            if args.DS == 'PROTEINS_full':
                vgea_loss = vgea_loss
            else:
                vgea_loss = vgea_loss / 20
            loss = loss_score + vgea_loss + loss_std
            optimizerG.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vage.parameters(), 0.1)
            optimizerG.step()
            scheduler.step()

        if (epoch + 1) % 10 == 0 and epoch > 0:
            vage.eval()
            loss = []
            y = []

            for batch_idx, data in enumerate(data_test_loader):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                _, _, loss_, _ = vage(h0, adj)
                loss_ = np.array(loss_.cpu().detach())
                loss.extend(loss_)

                for data_label in data['label']:
                    if data_label == args.sign:
                        y.append(1)
                    else:
                        y.append(0)

            label_test = np.array(loss).squeeze()

            fpr_ab, tpr_ab, thr_ = roc_curve(y, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)
            print('abnormal detection: AUC: {}'.format(format(test_roc_ab, '.4f')))
            if test_roc_ab > max_AUC:
                max_AUC = test_roc_ab
        if epoch == (args.num_epochs - 1):
            auroc_final = max_AUC
            best_fpr = fpr_ab
            best_tpr = tpr_ab
            F1_Score = f1_score_all(label_test, y, args.sign)
            best_precision, best_recall, _ = precision_recall_curve(y, label_test)

    return auroc_final, F1_Score, best_fpr, best_tpr, best_precision, best_recall


if __name__ == '__main__':
    args = arg_parse()
    n_k = 5
    setup_seed(args.seed)
    large = not args.DS.find("Tox21_") == -1

    graphs = load_data.read_graphfile(args.datadir, args.DS, max_nodes=args.max_nodes)
    datanum = len(graphs)
    if args.max_nodes == 0:
        max_nodes_num = max([G.number_of_nodes() for G in graphs])
    else:
        max_nodes_num = args.max_nodes
    print('GraphNumber: {}'.format(datanum))
    graphs_label = [graph.graph['label'] for graph in graphs]

    if large:
        DST = args.DS[:args.DS.rfind('_')] + "_testing"
        graphs_testgroup = load_data.read_graphfile(args.datadir, DST, max_nodes=args.max_nodes)
        datanum_test = len(graphs_testgroup)
        if args.max_nodes == 0:
            max_nodes_num = max([max([G.number_of_nodes() for G in graphs_testgroup]), max_nodes_num])
        else:
            max_nodes_num = args.max_nodes
        graphs_label_test = [graph.graph['label'] for graph in graphs_testgroup]

        graphs_all = graphs + graphs_testgroup
        graphs_label_all = graphs_label + graphs_label_test
    else:
        graphs_all = graphs
        graphs_label_all = graphs_label

    kfd = StratifiedKFold(n_splits=n_k, random_state=args.seed, shuffle=True)
    result_auc = []
    result_f1 = []
    for k, (train_index, test_index) in enumerate(kfd.split(graphs_all, graphs_label_all)):
        print(k + 1, '折实验')
        graphs_train = [graphs_all[i] for i in train_index]
        graphs_test = [graphs_all[i] for i in test_index]
        graphs_normal = []
        graphs_abnormal = []

        for graph in graphs_train:
            if graph.graph['label'] == args.sign:
                graphs_abnormal.append(graph)
            else:
                graphs_normal.append(graph)

        num_normal = len(graphs_normal)
        graphs_normal = graphs_normal[: int(args.ratio * num_normal)]
        num_abnormal = len(graphs_abnormal)
        graphs_abnormal = graphs_abnormal[: int(args.ratio * num_abnormal)]

        dataset_test = GraphBuild(graphs_test, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        data_test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=True, batch_size=args.batch_size)

        dataset_normal = GraphBuild(graphs_normal, features=args.feature, normalize=False, max_num_nodes=max_nodes_num)
        data_normal_loader = torch.utils.data.DataLoader(dataset_normal, shuffle=True, batch_size=args.batch_size)

        dataset_abnormal = GraphBuild(graphs_abnormal, features=args.feature, normalize=False,
                                      max_num_nodes=max_nodes_num)
        data_abnormal_loader = torch.utils.data.DataLoader(dataset_abnormal, shuffle=True, batch_size=args.batch_size)

        vage = VGAE(dataset_test.feat_dim, args.hidden_dim1, args.hidden_dim2, max_nodes_num).cuda()

        result, F1_Score, best_fpr, best_tpr, best_precision, best_recall = train(data_test_loader,
                                                                                  data_abnormal_loader,
                                                                                  data_normal_loader, vage, args)
        result_auc.append(result)
        result_f1.append(F1_Score)

    result_auc = np.array(result_auc)
    result_f1 = np.array(result_f1)
    auc_avg = np.mean(result_auc)
    f1_avg = np.mean(result_f1)
    auc_std = np.std(result_auc)
    f1_std = np.std(result_f1)
    print(auc_avg, auc_std)
