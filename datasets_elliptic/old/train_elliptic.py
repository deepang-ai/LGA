import warnings
import torch
import pandas as pd
from datasets_elliptic.old import utils as u
import os
from datasets_elliptic.old.loader import load_data, data_to_pyg
from train import train
import models
from argparse import ArgumentParser


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = ArgumentParser()
parser.add_argument("-d", "--data", dest="data_path", help="Path of data folder")
command_line_args = parser.parse_args()
data_path = command_line_args.data_path

print("Loading configuration from file...")
args = u.get_config()
print("Configuration loaded successfully")
print("=" * 50)
print("Loading graph data...")
data_path = args.data_path if data_path is None else data_path

features, edges = load_data(data_path)
features_noAgg, edges_noAgg = load_data(data_path, noAgg=True)

u.seed_everything(42)

data = data_to_pyg(features, edges)
data_noAgg = data_to_pyg(features_noAgg, edges_noAgg)

print("Graph data loaded successfully")
print("=" * 50)
args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
args.device = 'cpu'
if args.use_cuda:
    args.device = 'cuda'
print("Using CUDA: ", args.use_cuda, "- args.device: ", args.device)



models_to_train = {
    # 'GCN Convolution (tx)': models.GCNConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(
    #     args.device),
    'GCN Convolution (tx+agg)': models.GCNConvolution(args, data.num_features, args.hidden_units).to(args.device),
    # 'GAT Convolution (tx)': models.GATConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(
    #     args.device),
    'GAT Convolution (tx+agg)': models.GATConvolution(args, data.num_features, args.hidden_units).to(args.device),
    # 'SAGE Convolution (tx)': models.SAGEConvolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(
    #     args.device),
    # 'SAGE Convolution (tx+agg)': models.SAGEConvolution(args, data.num_features, args.hidden_units).to(args.device),
    # 'Chebyshev Convolution (tx)': models.ChebyshevConvolution(args, [1, 2], data_noAgg.num_features,
    #                                                           args.hidden_units_noAgg).to(args.device),
    # 'Chebyshev Convolution (tx+agg)': models.ChebyshevConvolution(args, [1, 2], data.num_features,
    #                                                               args.hidden_units).to(args.device),
    # 'GATv2 Convolution (tx)': models.GATv2Convolution(args, data_noAgg.num_features, args.hidden_units_noAgg).to(
    #     args.device),
    # 'GATv2 Convolution (tx+agg)': models.GATv2Convolution(args, data.num_features, args.hidden_units).to(args.device)

    # 'Custom GAT': GAT(num_of_layers=3, num_heads_per_layer=[1, 4, 1],
    #                  num_features_per_layer=[args.num_features, args['hidden_units'],
    #                  args['hidden_units']//2, args['num_classes']], device=args.device).to(args.device)
}

print("Deploy Model Complete!")

compare_illicit = pd.DataFrame(columns=['ethident', 'Precision', 'Recall', 'F1', 'F1 Micro AVG'])
print("Starting training models")
print("=" * 50)



for key, value in models_to_train.items():

    if key.endswith("(tx)"):
        (name, model) = key, value
        data_noAgg = data_noAgg.to(args.device)
        print('-' * 50)
        print(f"Training ethident: {name}")
        print('-' * 50)
        train(args, model, data_noAgg)
        print('-' * 50)
        print(f"Testing ethident: {name}")
        # print('-' * 50)
        # test(ethident, data_noAgg)
        # print('-' * 50)
        # print(f"Computing metrics for ethident: {name}")
        # print('-' * 50)
        # compare_illicit = compare_illicit.append(u.compute_metrics(ethident, name, data_noAgg, compare_illicit),
        #                                          ignore_index=True)
    else:
        (name, model) = key, value
        data = data.to(args.device)
        print('-' * 50)
        print(f"Training ethident: {name}")
        print('-' * 50)
        train(args, model, data)
        print('-' * 50)
        # print(f"Testing ethident: {name}")
        # print('-' * 50)
        # test(ethident, data)
        # print('-' * 50)
        # print(f"Computing metrics for ethident: {name}")
        # compare_illicit = compare_illicit.append(u.compute_metrics(ethident, name, data, compare_illicit), ignore_index=True)
        # print('-' * 50)

compare_illicit.to_csv(os.path.join(data_path, 'metrics.csv'), index=False)
print('Results saved to metrics.csv')

u.plot_results(compare_illicit)

u.aggregate_plot(compare_illicit)
