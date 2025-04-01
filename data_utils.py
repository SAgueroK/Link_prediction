# coding=UTF8
import os

import networkx as nx
import numpy as np
import pandas as pd


def load_data(filePath):
    if not os.path.exists(filePath):
        raise IOError
    else:
        return np.loadtxt(filePath, dtype=int, delimiter=' ')


def get_infectious_graphy(path):
    data = load_data(path)
    G_List = []
    tmp_G = nx.Graph()
    time = -1
    num_node = 0
    for i in data:
        num_node = max(num_node, i[0])
        num_node = max(num_node, i[1])
    for j in range(1, num_node + 1):
        tmp_G.add_node(j)
    count = 0
    max_edge = 0
    for i in data:
        if count > 100:
            break
        if time == -1:
            time = i[3]
        if time != i[3]:
            time = i[3]
            max_edge = max(max_edge, tmp_G.number_of_edges())
            G_List.append(tmp_G)
            tmp_G.clear_edges()
            count += 1

        # 添加无向边
        tmp_G.add_edge(i[0], i[1])
        tmp_G.add_edge(i[1], i[0])
    print(count)
    return num_node, G_List


def tvs_txt():
    df = pd.read_csv('.\data\mooc\mooc_actions.tsv', sep='\t')
    np.savetxt('.\data\mooc\mooc_actions.txt', df)

def npy_to_txt():
    path = './data/hyper/hyper.npy'  # 一个文件夹下多个npy文件，
    txtpath = './data/hyper/hyper.txt'
    input_data = np.load(path)
    np.savetxt(txtpath, input_data)


if __name__ == '__main__':
    get_infectious_graphy('.\data\mooc\mooc_actions.txt')
