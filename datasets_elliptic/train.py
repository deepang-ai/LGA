
import numpy as np
import networkx as nx
import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.manifold import TSNE

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import Parameter
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_undirected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
from loader import load_data, GraphDataset

from datasets_elliptic.lga.models import GraphTransformer
from datasets_elliptic.ethident.model import Ethident
from datasets_elliptic.ethident.encoder import HGATE_encoder
from models import GCN, I2BGNN, GAT, GIN

from torch_geometric.data import Data,DataLoader


GNN_TYPES = [
    'graph', 'graphsage', 'gcn',
    'gin', 'gine',
    'pna', 'pna2', 'pna3', 'mpnn', 'pna4',
    'rwgnn', 'khopgnn', 'gat'
]


def load_args():
    parser = argparse.ArgumentParser(
        description='LGA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--seed', type=int, default=0,
    #                     help='random seed')

    parser.add_argument('--model', type=str, default='LGA', choices=['LGA', 'Ethident', 'GCN', 'GAT', 'GIN',],
                        help='model choices')
    parser.add_argument('--txs-edgelist', type=str, default='./datasets_elliptic/elliptic_txs_edgelist.csv',
                        help='txs edge list')
    parser.add_argument('--txs-classes', type=str, default='./datasets_elliptic/elliptic_txs_classes.csv',
                        help='txs classes')
    parser.add_argument('--txs-features', type=str, default='./datasets_elliptic/elliptic_txs_features.csv',
                        help='txs features')
    parser.add_argument('--save-path', type=str, default='./datasets_elliptic/elliptic_bitcoin_dataset_cont',
                        help='save path')
    parser.add_argument('--k-hop', type=int, default=1, help="number of hops")
    parser.add_argument('--se', type=str, default="khopgnn",  # k-subgraph or k-subtree GNN extractor
                        help='Extractor type: khopgnn, or gnn')

    parser.add_argument('--dataset', type=str, default="ico_wallets/averVolume",
                        help='name of dataset')
    parser.add_argument('--gnn-type', type=str, default='gat',  # Base GNN model
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")


    args = parser.parse_args()
    return args




def main():
    args = load_args()
    txs_edgelist = args.txs_edgelist
    txs_classes = args.txs_classes
    txs_features = args.txs_features

    save_path = args.save_path



    # data, train_loader, test_loader, train_loader_with_aug = load_data(txs_edgelist, txs_classes, txs_features, save_path)

    print("Loading Data...")

    train_dataset, train_dataset_aug_1, train_dataset_aug_2, test_dataset = load_data(txs_edgelist, txs_classes, txs_features, save_path)

    train_dset = GraphDataset(train_dataset, degree=True,
                              k_hop=args.k_hop, se=args.se)
    train_dset_aug_1 = GraphDataset(train_dataset_aug_1, degree=True,
                              k_hop=args.k_hop, se=args.se)
    train_dset_aug_2 = GraphDataset(train_dataset_aug_2, degree=True,
                              k_hop=args.k_hop, se=args.se)
    test_dset= GraphDataset(test_dataset, degree=True,
                              k_hop=args.k_hop, se=args.se)


    train_loader_with_aug = DataLoader(list(zip(train_dset, train_dset_aug_1, train_dset_aug_2)), batch_size=1, shuffle=True)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False)


    if args.model == "LGA":
        model = GraphTransformer(in_size=train_dataset[0].num_node_features,
                                 num_class=2,
                                 d_model=64,
                                 dim_feedforward=int(2 * 64),
                                 dropout=0.1,
                                 num_heads=8,
                                 num_layers=2,
                                 batch_norm=False,
                                 gnn_type=args.gnn_type,
                                 k_hop=args.k_hop,
                                 use_edge_attr=False,
                                 num_edge_features=0,
                                 edge_dim=0,
                                 se=args.se,
                                 deg=None,
                                 in_embed=False,
                                 use_global_pool=False,
                                 edge_embed=False,
                                 global_pool=None,
                                 )

    elif args.model == "Ethident":
        encoder = HGATE_encoder(in_channels=train_dataset[0].num_node_features, hidden_channels=100, out_channels=2,
                                edge_dim=0, num_layers=2,  add_self_loops=True, use_edge_atten=False).to(device)

        model = Ethident(out_channels=2, encoder=encoder).to(device)


    elif args.model == "GAT":
        model = GAT(num_node_features=train_dataset[0].num_node_features, hidden_channels=100, out_channels=2)

    elif args.model == "GCN":
        model = GCN(num_node_features=train_dataset[0].num_node_features, hidden_channels=100, out_channels=2)

    elif args.model == "GIN":
        model = GIN(num_features=train_dataset[0].num_node_features, hidden_dim=100, embedding_dim=100, output_dim=2, n_layers=2)

    else:
        assert 0

    model.to(device)



    patience = 50
    lr = 0.001
    epoches = 1000

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.7, 0.3]).to(device))

    train_losses = []
    val_losses = []
    accuracies = []
    if1 = []
    precisions = []
    recalls = []
    iterations = []

    best = 0

    print("Start Training...")

    for epoch in range(epoches):

        model.train()
        train_loss = 0

        if isinstance(model, GraphTransformer) or isinstance(model, Ethident):
            for data, data_aug_1,  data_aug_2, in train_loader_with_aug:

                data = data.to(device)
                data_aug_1 = data_aug_1.to(device)
                data_aug_2 = data_aug_2.to(device)
                optimizer.zero_grad()
                out = model(data)
                out_aug_1 = model(data_aug_1)
                out_aug_2 = model(data_aug_2)

                tmp = data.train_mask

                loss_un = model.loss_un(out_aug_1[data.train_mask], out_aug_2[data.train_mask])
                loss_su = criterion(out[data.train_mask], data.y[data.train_mask].long())
                loss = 0.1 * loss_un + loss_su
                _, pred = out[data.train_mask].max(dim=1)
                loss.backward()
                train_loss += loss.item() * data.num_graphs
                optimizer.step()
        else:
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out[data.train_mask], data.y[data.train_mask].long())
                _, pred = out[data.train_mask].max(dim=1)
                loss.backward()
                train_loss += loss.item() * data.num_graphs
                optimizer.step()
        train_loss /= len(train_loader.dataset)


        if (epoch + 1) % 1 == 0:
            model.eval()
            ys, preds = [], []
            val_loss = 0
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out[data.test_mask], data.y[data.test_mask].long())
                val_loss += loss.item() * data.num_graphs
                _, pred = out[data.test_mask].max(dim=1)
                ys.append(data.y[data.test_mask].cpu())
                preds.append(pred.cpu())

            y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
            val_loss /= len(test_loader.dataset)
            f1 = f1_score(y, pred, average=None)
            mf1 = f1_score(y, pred, average='micro')
            precision = precision_score(y, pred, average=None)
            recall = recall_score(y, pred, average=None)

            iterations.append(epoch + 1)
            train_losses.append(train_loss)

            val_losses.append(val_loss)
            if1.append(f1[0])
            accuracies.append(mf1)
            precisions.append(precision[0])
            recalls.append(recall[0])

            if f1[0] > best:
                best = f1[0]
                now_best = {"train_loss": train_loss, "val_loss": val_loss, "precision": precision[0], "recall": recall[0], "Illicit f1": f1[0],  "F1:": mf1}
                print("Now Best:", now_best)

            print(
                'Epoch: {:02d}, Train_Loss: {:.4f}, Val_Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Illicit f1: {:.4f}, F1: {:.4f}'.format(
                    epoch + 1, train_loss, val_loss, precision[0], recall[0], f1[0], mf1))



if __name__ == "__main__":
    main()