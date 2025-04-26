from torch_geometric.nn import GCNConv, GATConv, GINConv

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Parameter


from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool



class Decoder_linear(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            output_dim: int = 2
    ):
        super().__init__()
        self.layer1 = nn.Linear(embedding_dim, output_dim)

    def forward(self, embedding):
        h = self.layer1(embedding)
        return h
class GIN(nn.Module):
    def __init__(
            self,
            num_features: int,
            hidden_dim: int,
            embedding_dim: int,
            output_dim: int,
            n_layers: int,
            dropout_rate: float = 0
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers



        if n_layers == 1:
            self.gin1 = GINConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                ))

        else:
            self.gin1 = GINConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ))

            self.gin_hidden = nn.ModuleList()
            for _ in range(n_layers - 2):
                self.gin_hidden.append(GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU()
                    )))

            self.gin2 = GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                ))

        self.out = Decoder_linear(embedding_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.gin1(x, edge_index)

        if self.n_layers > 1:
            for layer in self.gin_hidden:
                h = layer(h, edge_index)

            h = self.gin2(h, edge_index)
        out = self.out(h)

        return out

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels, use_skip=False):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.use_skip = use_skip
        if self.use_skip:
            self.weight = nn.init.xavier_normal_(Parameter(torch.Tensor(num_node_features, 2)))


    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        if self.use_skip:
            x = F.softmax(x+torch.matmul(data.x, self.weight), dim=-1)
        else:
            x = F.softmax(x, dim=-1)
        return x

    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels, use_skip=False):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        self.use_skip = use_skip
        if self.use_skip:
            self.weight = nn.init.xavier_normal_(Parameter(torch.Tensor(num_node_features, 2)))


    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        if self.use_skip:
            x = F.softmax(x+torch.matmul(data.x, self.weight), dim=-1)
        else:
            x = F.softmax(x, dim=-1)
        return x

    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x



class GAT2(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels, use_skip=False):
        super(GAT2, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        self.use_skip = use_skip
        if self.use_skip:
            self.weight = nn.init.xavier_normal_(Parameter(torch.Tensor(num_node_features, 2)))


    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        if self.use_skip:
            x = F.softmax(x+torch.matmul(data.x, self.weight), dim=-1)
        else:
            x = F.softmax(x, dim=-1)
        return x

    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x


pooling_dict = {'sum': global_add_pool,
                'mean': global_mean_pool,
                'max': global_max_pool}


class I2BGNN(torch.nn.Module):
    '''
    gcn ethident, in which the messages are aggregated with the edge weights.
    '''

    def __init__(self, in_channels, dim, out_channels, num_layers=2, pooling='max', BN=True, dropout=0.2, which_edge_weight=None):
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.BN = BN
        self.dropout = dropout
        self.which_edge_weight = which_edge_weight

        self.gcs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.drops = torch.nn.ModuleList()
        for i in range(num_layers):
            if i:
                gc = GCNConv(dim, dim)
            else:
                gc = GCNConv(in_channels, dim)
            bn = BatchNorm1d(dim)
            drop = Dropout(p=dropout)

            self.gcs.append(gc)
            self.bns.append(bn)
            self.drops.append(drop)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch,
        # if self.which_edge_weight == 'Volume':
        #     edge_weight = edge_attr[:, 0]
        # else:
        #     edge_weight = edge_attr[:, 1]

        for i in range(self.num_layers):
            x = F.relu(self.gcs[i](x, edge_index, edge_weight=None))
            if self.BN: x = self.bns[i](x)
            if self.dropout: x = self.drops[i](x)

        # x = pooling_dict[self.pooling](x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred







if __name__ == '__main__':
    args = {'num_classes': 2,}
    model = GCN(args, 166, 100).to('cuda')
    print(model)