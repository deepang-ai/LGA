import pandas as pd
import torch
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import numpy as np



def load_data(data_path, noAgg=False):
    # Read edges, features and classes from csv files
    df_edges = pd.read_csv(osp.join(data_path, "elliptic_txs_edgelist.csv"))
    df_features = pd.read_csv(osp.join(data_path, "elliptic_txs_features.csv"), header=None)
    df_classes = pd.read_csv(osp.join(data_path, "elliptic_txs_classes.csv"))

    # Name colums basing on index
    colNames1 = {'0': 'txId', 1: "Time step"}
    colNames2 = {str(ii + 2): "Local_feature_" + str(ii + 1) for ii in range(94)}
    colNames3 = {str(ii + 96): "Aggregate_feature_" + str(ii + 1) for ii in range(72)}

    colNames = dict(colNames1, **colNames2, **colNames3)
    colNames = {int(jj): item_kk for jj, item_kk in colNames.items()}

    # Rename feature columns
    df_features = df_features.rename(columns=colNames)


    if noAgg:
        df_features = df_features.drop(df_features.iloc[:, 96:], axis=1)

    # Map unknown class to '3'
    df_classes.loc[df_classes['class'] == 'unknown', 'class'] = '3'

    # Merge classes and features in one Dataframe
    df_class_feature = pd.merge(df_classes, df_features)

    # print(df_class_feature)

    # Exclude records with unknown class transaction
    df_class_feature = df_class_feature[df_class_feature["class"] != '3']

    # print(df_class_feature["Time step"])

    # Build Dataframe with head and tail of transactions (edges)
    known_txs = df_class_feature["txId"].values
    df_edges = df_edges[(df_edges["txId1"].isin(known_txs)) & (df_edges["txId2"].isin(known_txs))]

    # Build indices for features and edge types
    features_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["txId"].unique()))}
    class_idx = {name: idx for idx, name in enumerate(sorted(df_class_feature["class"].unique()))}

    # Apply index encoding to features
    df_class_feature["txId"] = df_class_feature["txId"].apply(lambda name: features_idx[name])
    df_class_feature["class"] = df_class_feature["class"].apply(lambda name: class_idx[name])

    # Apply index encoding to edges
    df_edges["txId1"] = df_edges["txId1"].apply(lambda name: features_idx[name])
    df_edges["txId2"] = df_edges["txId2"].apply(lambda name: features_idx[name])

    return df_class_feature, df_edges


def data_to_pyg(df_class_feature, df_edges):
    # Define PyTorch Geometric data structure with Pandas dataframe values

    time_step_list = df_class_feature['Time step'].tolist()

    train_mask_list = [1 if i <= 34 else 0 for i in time_step_list]

    val_mask_list = [1 if i > 34 else 0 for i in time_step_list]


    train_mask = torch.tensor(train_mask_list, dtype=torch.bool)

    val_mask = torch.tensor(val_mask_list, dtype=torch.bool)

    edge_index = torch.tensor([df_edges["txId1"].values,
                               df_edges["txId2"].values], dtype=torch.long)
    x = torch.tensor(df_class_feature.iloc[:, 3:].values, dtype=torch.float)
    y = torch.tensor(df_class_feature["class"].values, dtype=torch.long)

    num_ts = 49  # number of timestamps from the paper

    data = Data(x=x, edge_index=edge_index, y=y)
    # data = RandomNodeSplit(num_val=0.15, num_test=0.2)(data)


    data.train_mask = train_mask
    data.val_mask = val_mask

    return data

if __name__ == '__main__':
    data_path = '..'
    feature, edges = load_data(data_path)

    data = data_to_pyg(feature, edges)

    print("Graph data loaded successfully")
