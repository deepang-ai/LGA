import numpy as np

import pandas as pd


import torch.nn.functional as F


import torch

from torch_geometric.data import Data,DataLoader

from torch_geometric.utils import to_undirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch_geometric.transforms import BaseTransform


def load_data(txs_edgelist, txs_classes, txs_features):
    # Load Dataframe
    df_edge = pd.read_csv(txs_edgelist)
    df_class = pd.read_csv(txs_classes)
    df_features = pd.read_csv(txs_features,header=None)

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

    df_features.drop(columns=['time step']).to_csv('elliptic_bitcoin_dataset_cont/elliptic_txs_features.csv',
                                                   index=False,
                                                   header=None)
    df_class.rename(columns={'txId': 'nid', 'class': 'label'})[['nid', 'label']].sort_values(by='nid').to_csv(
        'elliptic_bitcoin_dataset_cont/elliptic_txs_classes.csv', index=False, header=None)
    df_features[['id', 'time step']].rename(columns={'id': 'nid', 'time step': 'time'})[['nid', 'time']].sort_values(
        by='nid').to_csv('elliptic_bitcoin_dataset_cont/elliptic_txs_nodetime.csv', index=False, header=None)
    df_edge_time_fin[['source', 'target', 'time']].to_csv(
        'elliptic_bitcoin_dataset_cont/elliptic_txs_edgelist_timed.csv',
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

    return train_dataset, test_dataset


if __name__ == '__main__':
    txs_edgelist = './elliptic_txs_edgelist.csv'
    txs_classes = './elliptic_txs_classes.csv'
    txs_features = './elliptic_txs_features.csv'

    train_dataset, test_dataset = load_data(txs_edgelist, txs_classes, txs_features)