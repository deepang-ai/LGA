#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse, logging
import numpy as np
import struc2vec
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time

import graph
from datasets_elliptic.graph_embed_loader import load_data
logging.basicConfig(filename='struc2vec.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
def parse_args():
	'''
	Parses the struc2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run struc2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=100,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=2,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=2,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=2,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--until-layer', type=int, default=None,
                    	help='Calculation until the layer.')

	parser.add_argument('--iter', default=10, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=16,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	parser.add_argument('--OPT1', default=True, type=bool,
                      help='optimization 1')
	parser.add_argument('--OPT2', default=True, type=bool,
                      help='optimization 2')
	parser.add_argument('--OPT3', default=True, type=bool,
                      help='optimization 3')	
	return parser.parse_args()

def read_graph(edges):
	'''
	Reads the input network.
	'''
	logging.info(" - Loading graph...")
	G = graph.load_edgelist(edges, undirected=True)
	logging.info(" - Graph loaded.")
	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	logging.info("Initializing creation of the representations...")
	# walks = LineSentence('random_walks.txt')
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1, workers=args.workers, epochs=args.iter)
	# model.wv.save_word2vec_format(args.output)
	logging.info("Representations created.")
	
	return model.wv

def exec_struc2vec(args, edges):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''



	if(args.OPT3):
		until_layer = args.until_layer
	else:
		until_layer = None

	G = read_graph(edges)
	G = struc2vec.Graph(G, args.directed, args.workers, untilLayer = until_layer)

	print("Graph loaded.")

	if(args.OPT1):
		G.preprocess_neighbors_with_bfs_compact()
	else:
		G.preprocess_neighbors_with_bfs()

	if(args.OPT2):
		G.create_vectors()
		G.calc_distances(compactDegree = args.OPT1)
	else:
		G.calc_distances_all_vertices(compactDegree = args.OPT1)

	G.create_distances_network()
	print("Distances created.")

	G.preprocess_parameters_random_walk()
	print("Parameters preprocessed.")

	walks = G.simulate_walks(args.num_walks, args.walk_length)


	return walks

def main(args):
	txs_edgelist = '../elliptic_txs_edgelist.csv'
	txs_classes = '../elliptic_txs_classes.csv'
	txs_features = '../elliptic_txs_features.csv'

	train_dataset, test_dataset = load_data(txs_edgelist, txs_classes, txs_features)

	model = LogisticRegression(penalty='l2', dual=False, C=1.0, n_jobs=1, random_state=20, max_iter=10000,
							   class_weight={0: 0.7, 1: 0.3})
	# model = RandomForestClassifier(oob_score=True, class_weight={0: 0.7, 1: 0.3})

	for idx, train_data in enumerate(train_dataset):
		print("Train idx:", idx + 1)
		edges = train_data.edge_index.T.numpy().tolist()
		walks = exec_struc2vec(args, edges)
		embeddings = learn_embeddings(walks)
		all_embeddings = embeddings.vectors

		model.fit(all_embeddings[train_data.train_mask], train_data.y[train_data.train_mask].numpy().tolist())

	ys, preds = [], []

	for idx, test_data in enumerate(test_dataset):
		print("Test idx:", idx + 1)
		edges = test_data.edge_index.T.numpy().tolist()
		walks = exec_struc2vec(args, edges)
		embeddings = learn_embeddings(walks)
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


if __name__ == "__main__":
	args = parse_args()
	main(args)

