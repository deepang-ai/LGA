import random

import torch
from gensim.models import Word2Vec
import networkx as nx
from datasets_elliptic.graph_embed_loader import load_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, embedding_size, window_size, num_epochs, num_workers):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers

    def random_walk(self, node):
        walk = [node]
        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) == 0:
                break
            node = random.choice(neighbors)
            walk.append(node)
        return walk

    def generate_walks(self):
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.random_walk(node)
                walks.append(walk)
        return walks

    def learn_embeddings(self, walks):
        model = Word2Vec(walks, vector_size=self.embedding_size, window=self.window_size, min_count=0, sg=1,
                         workers=self.num_workers, epochs=self.num_epochs)
        return model.wv


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
        deepwalk = DeepWalk(G, walk_length=2, num_walks=2, embedding_size=100, window_size=2, num_epochs=10,
                            num_workers=4)
        walks = deepwalk.generate_walks()
        embeddings = deepwalk.learn_embeddings(walks)
        all_embeddings = embeddings.vectors

        model.fit(all_embeddings[train_data.train_mask], train_data.y[train_data.train_mask].numpy().tolist())


    ys, preds = [], []
    for idx, test_data in enumerate(test_dataset):
        print("Test idx:", idx+1)
        edges = test_data.edge_index.T.numpy().tolist()
        G = nx.Graph()
        G.add_edges_from(edges)
        deepwalk = DeepWalk(G, walk_length=2, num_walks=2, embedding_size=100, window_size=2, num_epochs=10,
                            num_workers=4)
        walks = deepwalk.generate_walks()
        embeddings = deepwalk.learn_embeddings(walks)
        all_embeddings = embeddings.vectors
        y_pred = model.predict(all_embeddings)

        ys.append(test_data.y[test_data.test_mask])
        preds.append(torch.tensor(y_pred[test_data.test_mask]))


    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()



    f1 = f1_score(y, pred, average=None)
    mf1 = f1_score(y, pred, average='micro')
    precision = precision_score(y, pred, average=None)
    recall = recall_score(y, pred, average=None)

    print(
        'Precision: {:.16f}, Recall: {:.16f}, Illicit f1: {:.16f}, F1: {:.16f}'.format(
            precision[0], recall[0], f1[0], mf1))