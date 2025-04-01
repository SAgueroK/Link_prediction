# coding=UTF8
import networkx as nx
import numpy
import numpy as np
from deepsnap.graph import Graph
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt

from RTime import RTime
from data_utils import get_infectious_graphy

path = 'dataset'
node_num, G_list = get_infectious_graphy(path)
data_list = []
datas = []
d_model = 3
input_size = 64
gcn_output = 8
time_step = 9
epoch = 1
# 将datalist 变成 x * 10 * G 的形式
for i, G in enumerate(G_list):
    if i != 0 and i % 10 == 0:
        data_list.append(datas)
        datas = []
    data = Graph(G)
    data.num_features = d_model
    data.x = torch.ones((data.num_nodes, data.num_features), dtype=torch.float32)
    datas.append(data)
edge_index = [[i for i in range(node_num)], [i for i in range(node_num)]]
division = int(node_num * 0.8)
# 画图
epoch_list = []
auc_list = []
er_list = []


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 64)
        self.conv2 = GCNConv(64, gcn_output)
        self.rtime = RTime(time_step, gcn_output * node_num)
        self.linear = nn.Linear(time_step, 1)

    def forward(self, datas):
        graphy_list = []
        for i in range(time_step):
            x = self.encode(datas[i])
            graphy_list.append(x)
        # seq * node_num * gcn_out
        graphy_list = torch.stack(
            (graphy_list[0], graphy_list[1], graphy_list[2], graphy_list[3], graphy_list[4], graphy_list[5],
             graphy_list[6], graphy_list[7], graphy_list[8]
             ), dim=0)
        # seq * node_num × gcn_out
        graphy_list = graphy_list.reshape(1, time_step, -1)
        graphy_list.to(device='cuda')
        res = self.rtime(graphy_list)
        res = res.reshape(time_step, node_num, gcn_output)
        res = res.permute(1, 2, 0)
        res = res.reshape(node_num * gcn_output, -1)
        res = self.linear(res)
        res = res.reshape(node_num, gcn_output)
        return res

    def encode(self, data):
        data.to(device='cuda')
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = self.conv2(x, data.edge_index)
        x = x.relu()
        # return self.conv2(x, data.edge_index)
        return x

    # def decode(self, z):
    #     # logits = np.zeros((node_num, node_num))
    #     # logits
    #     # for i in edge_index[0]:
    #     #     for j in edge_index[1]:
    #     #         logits[i][j] = (z[i] * z[j]).sum()
    #     logits = (z[edge_index[0]]*z[edge_index[1]]).sum(dim=-1)
    #     return torch.tensor(logits)
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all_er(self, z):
        logits = np.zeros((node_num, node_num))
        for i in range(node_num):
            for j in range(node_num):
                if (z[i] * z[j]).sum() > 0.5:
                    logits[i][j] = 1.0
                else:
                    logits[i][j] = 0.0
        return torch.tensor(logits)

    def decode_all_acu(self, z):
        logits = np.zeros((node_num, node_num))
        for i in range(node_num):
            for j in range(node_num):
                if (z[i] * z[j]).sum() > 0.5:
                    logits[i][j] = 1.0
                if (z[i] * z[j]).sum() < 0.5:
                    logits[i][j] = 0.0
        return torch.tensor(logits)


device = torch.device('cuda')
model = Net().to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def get_all_link_labels(pos_edge_index):
    logits = np.zeros((node_num, node_num))
    for i, _ in enumerate(pos_edge_index[0]):
        logits[pos_edge_index[0][i]][pos_edge_index[1][i]] = 1.0

    return torch.tensor(logits)


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()
    total_loss = 0
    for i in range(20):
        datas = data_list[i]
        neg_edge_index = negative_sampling(
            edge_index=datas[time_step].edge_index, num_nodes=datas[time_step].num_nodes,
            num_neg_samples=datas[time_step].edge_index.size(1),
            force_undirected=True,
        )
        z = model(datas)
        link_logits = model.decode(z, datas[time_step].edge_index, neg_edge_index)
        link_labels = get_link_labels(datas[time_step].edge_index, neg_edge_index)
        # print(link_logits)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.data
        optimizer.step()
        print(loss.data)

    print("train loss:" + str(total_loss))


# @torch.no_grad()
def test():
    model.eval()
    equal = 0.0
    acu = 0.0
    fault = 0.0
    index = 0.0
    for i in range(20):
        datas = data_list[i]
        z = model.forward(datas)
        link_logits = model.decode_all_acu(z)
        link_labels = get_all_link_labels(datas[time_step].edge_index)
        for j, tmp in enumerate(link_logits):
            for k, _ in enumerate(tmp):
                if link_logits[j][k] == link_labels[j][k]:
                    acu += 1
                if link_logits[j][k] == 0.5:
                    equal += 1
                if link_logits[j][k] != link_labels[j][k]:
                    fault += 1
                index += 1
        link_logits = model.decode(z, datas[time_step].edge_index, neg_edge_index)
        link_labels = get_link_labels(datas[time_step].edge_index, neg_edge_index)
        for j, _ in enumerate(link_logits):
            if link_labels[j] == link_logits[j]:
                acu += 1
            index += 1
        for j, _ in enumerate(link_logits):
            if link_logits[j] > 0.5:
                link_logits[j] = 1.0
            else:
                link_logits[j] = 0.0
            # print(link_labels[j])
            # print(link_logits[j])
            if link_labels[j] == link_logits[j]:
                acu += 1
            index += 1

        print("eal loss :" + str(loss.item()))

    er_list.append(fault / index)
    auc_list.append(acu / index)
    print("eal acu:" + str((acu + 0.5 * equal) / index))
    print("eal er:" + str(fault / index))


if __name__ == '__main__':
    for t in range(20):
        train()
        epoch_list.append(t)
    test()
    epoch_list = np.array(epoch_list)
    auc_list = np.array(auc_list)
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.plot(auc_list)
    avg = np.mean(auc_list)
    plt.axhline(y=avg, color='r', linestyle=':')
    plt.savefig('./png/' + "infectious_accuracy" + '.png')
    plt.ylabel('ER')
    plt.xlabel('Epoch')
    plt.plot(er_list)
    avg = np.mean(er_list)
    plt.axhline(y=avg, color='r', linestyle=':')
    plt.savefig('./png/' + "infectious_error" + '.png')
    np.savetxt('./data/infectious/infectious_auc.txt', auc_list)
    np.savetxt('./data/infectious/infectious_er.txt', er_list)

