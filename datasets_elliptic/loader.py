import os.path

import numpy as np

import pandas as pd


import torch.nn.functional as F


import torch

from torch_geometric.data import Data,DataLoader

from torch_geometric.utils import to_undirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch_geometric.utils as utils

from torch_geometric.transforms import BaseTransform

class MyAug_NodeAttributeMasking(BaseTransform):

    def __init__(self, prob=None):
        self.prob = prob

    def __call__(self, data):
        # data.edge_attr = None
        # batch = data.batch if 'batch' in data else None
        drop_mask = torch.empty(size=(data.x.size(1),), dtype=torch.float32).uniform_(0, 1) < self.prob
        data.x[:, drop_mask] = 0

        return data

def load_data(txs_edgelist, txs_classes, txs_features, save_path):
    # Load Dataframe
    df_edge = pd.read_csv(txs_edgelist)
    df_class = pd.read_csv(txs_classes)
    df_features = pd.read_csv(txs_features, header=None)

    # Setting Column name
    df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in
                                                                                          range(72)]

    print('Number of edges: {}'.format(len(df_edge)))

    all_nodes = list(
        set(df_edge['txId1']).union(set(df_edge['txId2'])).union(set(df_class['txId'])).union(set(df_features['id'])))
    nodes_df = pd.DataFrame(all_nodes, columns=['id']).reset_index()

    print('Number of nodes: {}'.format(len(nodes_df)))

    df_edge = df_edge.join(nodes_df.rename(columns={'id': 'txId1'}).set_index('txId1'), on='txId1', how='inner') \
        .join(nodes_df.rename(columns={'id': 'txId2'}).set_index('txId2'), on='txId2', how='inner', rsuffix='2') \
        .drop(columns=['txId1', 'txId2']) \
        .rename(columns={'index': 'txId1', 'index2': 'txId2'})
    df_edge.head()

    df_class = df_class.join(nodes_df.rename(columns={'id': 'txId'}).set_index('txId'), on='txId', how='inner') \
        .drop(columns=['txId']).rename(columns={'index': 'txId'})[['txId', 'class']]
    df_class.head()

    df_features = df_features.join(nodes_df.set_index('id'), on='id', how='inner') \
        .drop(columns=['id']).rename(columns={'index': 'id'})
    df_features = df_features[['id'] + list(df_features.drop(columns=['id']).columns)]
    df_features.head()

    df_edge_time = df_edge.join(df_features[['id', 'time step']].rename(columns={'id': 'txId1'}).set_index('txId1'),
                                on='txId1', how='left', rsuffix='1') \
        .join(df_features[['id', 'time step']].rename(columns={'id': 'txId2'}).set_index('txId2'), on='txId2',
              how='left',
              rsuffix='2')
    df_edge_time['is_time_same'] = df_edge_time['time step'] == df_edge_time['time step2']
    df_edge_time_fin = df_edge_time[['txId1', 'txId2', 'time step']].rename(
        columns={'txId1': 'source', 'txId2': 'target', 'time step': 'time'})

    df_features.drop(columns=['time step']).to_csv(os.path.join(save_path, 'elliptic_txs_features.csv'),
                                                   index=False,
                                                   header=None)
    df_class.rename(columns={'txId': 'nid', 'class': 'label'})[['nid', 'label']].sort_values(by='nid').to_csv(
        os.path.join(save_path, 'elliptic_txs_classes.csv'), index=False, header=None)
    df_features[['id', 'time step']].rename(columns={'id': 'nid', 'time step': 'time'})[['nid', 'time']].sort_values(
        by='nid').to_csv(os.path.join(save_path, 'elliptic_txs_nodetime.csv'), index=False, header=None)
    df_edge_time_fin[['source', 'target', 'time']].to_csv(
        os.path.join(save_path, 'elliptic_txs_edgelist_timed.csv'),
        index=False, header=None)

    node_label = df_class.rename(columns={'txId': 'nid', 'class': 'label'})[['nid', 'label']].sort_values(
        by='nid').merge(df_features[['id', 'time step']].rename(columns={'id': 'nid', 'time step': 'time'}), on='nid',
                        how='left')
    node_label['label'] = node_label['label'].apply(lambda x: '3' if x == 'unknown' else x).astype(int) - 1
    node_label.head()

    merged_nodes_df = node_label.merge(
        df_features.rename(columns={'id': 'nid', 'time step': 'time'}).drop(columns=['time']), on='nid', how='left')
    merged_nodes_df.head()

    train_dataset = []
    test_dataset = []
    for i in range(49):
        nodes_df_tmp = merged_nodes_df[merged_nodes_df['time'] == i + 1].reset_index()
        nodes_df_tmp['index'] = nodes_df_tmp.index
        df_edge_tmp = df_edge_time_fin.join(
            nodes_df_tmp.rename(columns={'nid': 'source'})[['source', 'index']].set_index('source'), on='source',
            how='inner') \
            .join(nodes_df_tmp.rename(columns={'nid': 'target'})[['target', 'index']].set_index('target'), on='target',
                  how='inner', rsuffix='2') \
            .drop(columns=['source', 'target']) \
            .rename(columns={'index': 'source', 'index2': 'target'})
        x = torch.tensor(np.array(nodes_df_tmp.sort_values(by='index').drop(columns=['index', 'nid', 'label'])),
                         dtype=torch.float)
        edge_index = torch.tensor(np.array(df_edge_tmp[['source', 'target']]).T, dtype=torch.long)
        edge_index = to_undirected(edge_index)
        mask = nodes_df_tmp['label'] != 2
        y = torch.tensor(np.array(nodes_df_tmp['label']))

        if i + 1 < 35:
            data = Data(x=x, edge_index=edge_index, train_mask=mask, y=y)
            train_dataset.append(data)
        else:
            data = Data(x=x, edge_index=edge_index, test_mask=mask, y=y)
            test_dataset.append(data)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_dataset_aug_1 = []
    train_dataset_aug_2 = []
    for data in train_dataset:

        drop_mask_1 = torch.empty(size=(data.x.size(1),), dtype=torch.float32).uniform_(0, 1) < 0.2
        drop_mask_2 = torch.empty(size=(data.x.size(1),), dtype=torch.float32).uniform_(0, 1) < 0.2
        data_tmp_1 = data.clone()
        data_tmp_2 = data.clone()
        data_tmp_1.x[:, drop_mask_1] = 0
        data_tmp_2.x[:, drop_mask_2] = 0
        train_dataset_aug_1.append(data_tmp_1)
        train_dataset_aug_2.append(data_tmp_2)



    train_loader_with_aug = DataLoader(list(zip(train_dataset, train_dataset_aug_1, train_dataset_aug_2)), batch_size=1, shuffle=True)

    return train_dataset, train_dataset_aug_1, train_dataset_aug_2, test_dataset
    # return data, train_loader, test_loader, train_loader_with_aug


def my_inc(self, key, value, *args, **kwargs):
    if key == 'subgraph_edge_index':
        return self.num_subgraph_nodes
    if key == 'subgraph_node_idx':
        return self.num_nodes
    if key == 'subgraph_indicator':
        return self.num_nodes
    elif 'index' in key:
        return self.num_nodes
    else:
        return 0

class GraphDataset(object):
    def __init__(self, dataset, degree=False, k_hop=2, se="gnn", use_subgraph_edge_attr=False,
                 cache_path=None, return_complete_index=False):
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.degree = degree  # True
        self.compute_degree()
        self.abs_pe_list = None
        self.return_complete_index = return_complete_index  # False
        self.k_hop = k_hop  # 2
        self.se = se  # gnn
        self.use_subgraph_edge_attr = use_subgraph_edge_attr  # True
        self.cache_path = cache_path
        if self.se == 'khopgnn':
            Data.__inc__ = my_inc
            self.extract_subgraphs()

    def compute_degree(self):

        # for data in self.dataset:
        #     assert data.edge_index.max() < data.num_nodes

        if not self.degree:
            self.degree_list = None
            return
        self.degree_list = []
        for g in self.dataset:
            deg = 1. / torch.sqrt(1. + utils.degree(g.edge_index[0], g.num_nodes))
            self.degree_list.append(deg)

    def extract_subgraphs(self):
        print("Extracting {}-hop subgraphs...".format(self.k_hop))
        # indicate which node in a graph it is; for each graph, the
        # indices will range from (0, num_nodes). PyTorch will then
        # increment this according to the batch size
        self.subgraph_node_index = []

        # Each graph will become a block diagonal adjacency matrix of
        # all the k-hop subgraphs centered around each node. The edge
        # indices get augumented within a given graph to make this
        # happen (and later are augmented for proper batching)
        self.subgraph_edge_index = []

        # This identifies which indices correspond to which subgraph
        # (i.e. which node in a graph)
        self.subgraph_indicator_index = []

        # This gets the edge attributes for the new indices
        if self.use_subgraph_edge_attr:
            self.subgraph_edge_attr = []

        for i in range(len(self.dataset)):
            if self.cache_path is not None:
                filepath = "{}_{}.pt".format(self.cache_path, i)
                if os.path.exists(filepath):
                    continue
            graph = self.dataset[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicators = []
            edge_index_start = 0

            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
                    node_idx,
                    self.k_hop,
                    graph.edge_index,
                    relabel_nodes=True,
                    num_nodes=graph.num_nodes
                )
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_index_start)
                indicators.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask])  # CHECK THIS DIDN"T BREAK ANYTHING
                edge_index_start += len(sub_nodes)

            if self.cache_path is not None:
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    subgraph_edge_attr = torch.cat(edge_attributes)
                else:
                    subgraph_edge_attr = None
                torch.save({
                    'subgraph_node_index': torch.cat(node_indices),
                    'subgraph_edge_index': torch.cat(edge_indices, dim=1),
                    'subgraph_indicator_index': torch.cat(indicators).type(torch.LongTensor),
                    'subgraph_edge_attr': subgraph_edge_attr
                }, filepath)
            else:
                self.subgraph_node_index.append(torch.cat(node_indices))
                self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
                self.subgraph_indicator_index.append(torch.cat(indicators))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    self.subgraph_edge_attr.append(torch.cat(edge_attributes))
        print("Done!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        data = self.dataset[index]

        # if self.n_features == 1:
        #     data.x = data.x.squeeze(-1)
        # if not isinstance(data.y, list):
        #     data.y = data.y.view(data.y.shape[0], -1)
        n = data.num_nodes
        s = torch.arange(n)
        if self.return_complete_index:
            data.complete_edge_index = torch.vstack((s.repeat_interleave(n), s.repeat(n)))
        data.degree = None
        if self.degree:
            data.degree = self.degree_list[index]
        # data.abs_pe = None
        data.abs_pe = None
        if self.abs_pe_list is not None and len(self.abs_pe_list) == len(self.dataset):
            data.abs_pe = self.abs_pe_list[index]

        # add subgraphs and relevant meta data
        if self.se == "khopgnn":
            if self.cache_path is not None:
                cache_file = torch.load("{}_{}.pt".format(self.cache_path, index))
                data.subgraph_edge_index = cache_file['subgraph_edge_index']
                data.num_subgraph_nodes = len(cache_file['subgraph_node_index'])
                data.subgraph_node_idx = cache_file['subgraph_node_index']
                data.subgraph_edge_attr = cache_file['subgraph_edge_attr']
                data.subgraph_indicator = cache_file['subgraph_indicator_index']
                return data
            data.subgraph_edge_index = self.subgraph_edge_index[index]
            data.num_subgraph_nodes = len(self.subgraph_node_index[index])
            data.subgraph_node_idx = self.subgraph_node_index[index]
            if self.use_subgraph_edge_attr and data.edge_attr is not None:
                data.subgraph_edge_attr = self.subgraph_edge_attr[index]
            data.subgraph_indicator = self.subgraph_indicator_index[index].type(torch.LongTensor)
        else:
            data.num_subgraph_nodes = None
            data.subgraph_node_idx = None
            data.subgraph_edge_index = None
            data.subgraph_indicator = None

        return data

if __name__ == '__main__':
    txs_edgelist = './elliptic_txs_edgelist.csv'
    txs_classes = './elliptic_txs_classes.csv'
    txs_features = './elliptic_txs_features.csv'

    data, train_loader, test_loader, train_loader_with_aug = load_data(txs_edgelist, txs_classes, txs_features)