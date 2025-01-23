import random

import torch
from gensim.models import Word2Vec
import networkx as nx
from datasets_elliptic.graph_embed_loader import load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

class Node2Vec:
    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1.0, q=1.0, workers=1, window_size=10,
                 min_count=0, sg=1):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.window_size = window_size
        self.min_count = min_count
        self.sg = sg

    def _simulate_walks(self, num_walks, walk_length):
        G = self.graph
        walks = []
        nodes = list(G.nodes())
        print('Walking...')
        for walk_iter in range(num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self._node2vec_walk(walk_length=walk_length, start_node=node))
        print('Walks generated.')
        return walks

    def _node2vec_walk(self, walk_length, start_node):
        G = self.graph
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[np.random.choice(len(cur_nbrs))])
                else:
                    prev = walk[-2]
                    next_node = self._biased_sample(cur, prev)
                    walk.append(next_node)
            else:
                break
        return walk

    def _biased_sample(self, cur, prev):
        G = self.graph
        p, q = self.p, self.q
        cur_nbrs = sorted(G.neighbors(cur))
        prev_nbrs = sorted(G.neighbors(prev))
        shared_nbrs = set(cur_nbrs) & set(prev_nbrs)
        weights = []
        for nbr in cur_nbrs:
            if nbr in shared_nbrs:
                weights.append(1 / p)
            elif G.has_edge(nbr, prev):
                weights.append(1)
            else:
                weights.append(1 / q)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        next_node = np.random.choice(cur_nbrs, p=weights)
        return next_node

    def fit(self):
        walks = self._simulate_walks(self.num_walks, self.walk_length)
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, vector_size=self.dimensions, window=self.window_size, min_count=self.min_count,
                         sg=self.sg, workers=self.workers)
        self.embeddings = model.wv

    def get_embeddings(self):
        return self.embeddings.vectors




if __name__ == '__main__':


    txs_edgelist = '../elliptic_txs_edgelist.csv'
    txs_classes = '../elliptic_txs_classes.csv'
    txs_features = '../elliptic_txs_features.csv'

    train_dataset, test_dataset = load_data(txs_edgelist, txs_classes, txs_features)

    model = LogisticRegression(penalty='l2', dual=False, C=1.0, n_jobs=1, random_state=20, max_iter=10000, class_weight={0: 0.7, 1: 0.3})
    # model = RandomForestClassifier(oob_score=True, class_weight={0: 0.7, 1: 0.3})


    for idx, train_data in enumerate(train_dataset):
        print("Train idx:", idx+1)
        edges = train_data.edge_index.T.numpy().tolist()
        G = nx.Graph()
        G.add_edges_from(edges)
        node2vec = Node2Vec(G, dimensions=100, walk_length=2, num_walks=2,  p=1.0, q=1.0, workers=4, window_size=2, min_count=1, sg=0)
        node2vec.fit()
        all_embeddings = node2vec.get_embeddings()

        tmp = train_data.y[train_data.train_mask].numpy().tolist()

        model.fit(all_embeddings[train_data.train_mask], train_data.y[train_data.train_mask].numpy().tolist())


    ys, preds = [], []
    for idx, test_data in enumerate(test_dataset):
        print("Test idx:", idx+1)
        edges = test_data.edge_index.T.numpy().tolist()
        G = nx.Graph()
        G.add_edges_from(edges)
        node2vec = Node2Vec(G, dimensions=100, walk_length=2, num_walks=2,  p=1.0, q=1.0, workers=4, window_size=2, min_count=1, sg=0)
        node2vec.fit()
        all_embeddings = node2vec.get_embeddings()
        y_pred = model.predict(all_embeddings)

        ys.append(test_data.y[test_data.test_mask])
        preds.append(torch.tensor(y_pred[test_data.test_mask]))


    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

    print(y)
    print(pred)

    f1 = f1_score(y, pred, average=None)
    mf1 = f1_score(y, pred, average='micro')
    precision = precision_score(y, pred, average=None)
    recall = recall_score(y, pred, average=None)

    print(
        'Precision: {:.16f}, Recall: {:.16f}, Illicit f1: {:.16f}, F1: {:.16f}'.format(
            precision[0], recall[0], f1[0], mf1))