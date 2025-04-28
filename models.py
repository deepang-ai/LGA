from torch_geometric.nn import GCNConv, GATConv, GINConv

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Parameter

from torch_geometric.nn import MessagePassing
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch_geometric.nn as gnn
from torch_geometric.nn import GAE

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

        self.pooling = gnn.global_mean_pool



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

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(True),
            nn.Linear(embedding_dim, output_dim)
        )



    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.gin1(x, edge_index)

        if self.n_layers > 1:
            for layer in self.gin_hidden:
                h = layer(h, edge_index)

            h = self.gin2(h, edge_index)

        graph_reps = self.pooling(h, batch)

        return h, graph_reps, self.classifier(graph_reps)

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.pooling = gnn.global_mean_pool

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        graph_reps = self.pooling(x, batch)

        return x, graph_reps, self.classifier(graph_reps)


    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x


class GCNTest(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels):
        super(GCNTest, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.pooling = gnn.global_mean_pool

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        graph_reps = self.pooling(x)

        return x, graph_reps, self.classifier(graph_reps)


    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, out_channels, use_skip=False):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.use_skip = use_skip
        if self.use_skip:
            self.weight = nn.init.xavier_normal_(Parameter(torch.Tensor(num_node_features, 2)))
        self.pooling = gnn.global_mean_pool

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        graph_reps = self.pooling(x, batch)


        return x, graph_reps, self.classifier(graph_reps)

    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x


class EdgeFeatureGAT(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.dropout = dropout

        # Node feature transformations
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)

        # Edge feature transformations
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, heads * out_dim),
            nn.ReLU(),
            nn.Linear(heads * out_dim, heads * out_dim)
        )

        # Attention parameters
        self.att = nn.Parameter(torch.Tensor(1, heads, out_dim))

        # Skip connection
        self.skip = nn.Linear(in_dim, heads * out_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.edge_encoder[0].weight)
        nn.init.xavier_uniform_(self.edge_encoder[2].weight)
        nn.init.xavier_uniform_(self.skip.weight)
        nn.init.xavier_normal_(self.att)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [N, in_dim]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, edge_dim]
        """
        # Add self-loops and get their edge attributes (zero vectors)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr,
            fill_value=0.0,
            num_nodes=x.size(0)
        )

        # Transform node features
        x_trans = self.lin(x)  # [N, heads * out_dim]
        x_trans = x_trans.view(-1, self.heads, self.out_dim)  # [N, heads, out_dim]

        # Transform edge features
        edge_attr_trans = self.edge_encoder(edge_attr)  # [E, heads * out_dim]
        edge_attr_trans = edge_attr_trans.view(-1, self.heads, self.out_dim)  # [E, heads, out_dim]

        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x_trans,
            edge_attr=edge_attr_trans,
            size=(x.size(0), x.size(0))
        )

        # Skip connection
        skip = self.skip(x)  # [N, heads * out_dim]
        skip = skip.view(-1, self.heads, self.out_dim)  # [N, heads, out_dim]

        out = out + skip
        out = out.view(-1, self.heads * self.out_dim)  # [N, heads * out_dim]

        return out

    def message(self, x_j, edge_attr, index, ptr, size_i):
        """
        Args:
            x_j: Source node features [E, heads, out_dim]
            edge_attr: Edge features [E, heads, out_dim]
            index: Target node indices [E]
        """
        # Combine node and edge features
        x_j = x_j + edge_attr  # [E, heads, out_dim]

        # Compute attention weights based on edge features
        alpha = (x_j * self.att).sum(dim=-1)  # [E, heads]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index, ptr, size_i)  # [E, heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)  # [E, heads, out_dim]


def softmax(src, index, ptr=None, dim_size=None):
    """
    Softmax operation over nodes for each edge
    """
    src = src - src.max(dim=0, keepdim=True)[0]
    exp_src = torch.exp(src)

    if ptr is not None:
        # For batched graphs
        out = torch.zeros_like(exp_src)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_src), exp_src)
        out = out.index_select(0, index)
    else:
        # For single graphs
        out = torch.zeros_like(exp_src)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_src), exp_src)
        out = out.index_select(0, index)

    return exp_src / (out + 1e-16)


class transGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, out_channels, num_heads=1):
        super(transGAT, self).__init__()
        self.encoder = EdgeFeatureGAT(num_node_features,  hidden_channels, num_edge_features, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, out_channels)
        )
        self.pooling = gnn.global_mean_pool

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.encoder(x, edge_index, edge_attr)

        graph_reps = self.pooling(x, batch)


        return x, graph_reps, self.classifier(graph_reps)

    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x






class AEtransGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, out_channels):
        super(AEtransGAT, self).__init__()
        self.gae = GAE(
            transGAT(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_channels=hidden_channels,
                out_channels=out_channels
            )
        )


    def encoder(self, *args, **kwargs):
        """Forward pass for encoding"""
        return self.gae.encode(*args, **kwargs)

    def decoder(self, *args, **kwargs):
        """Forward pass for decoding"""
        return self.gae.decode(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass"""
        return self.gae(*args, **kwargs)

    def recon_loss(self, *args, **kwargs):
        """Compute reconstruction loss"""
        return self.gae.recon_loss(*args, **kwargs)

    def test(self, *args, **kwargs):
        """Test model"""
        return self.gae.test(*args, **kwargs)


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
        self.lin2 = Linear(dim, dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, out_channels)
        )

        self.pooling = gnn.global_mean_pool

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if self.which_edge_weight == 'T':
            edge_weight = edge_attr[:, 0]
        else:
            edge_weight = edge_attr[:, 1]

        for i in range(self.num_layers):
            x = F.relu(self.gcs[i](x, edge_index, edge_weight=edge_weight))
            if self.BN: x = self.bns[i](x)
            if self.dropout: x = self.drops[i](x)

        # x = pooling_dict[self.pooling](x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        graph_reps = self.pooling(x, batch)

        return x, graph_reps, self.classifier(graph_reps)









if __name__ == '__main__':
    args = {'num_classes': 2,}
    model = GCN(args, 166, 100).to('cuda')
    print(model)