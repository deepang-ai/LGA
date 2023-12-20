#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: dataset.py
@time: 2022/1/15 15:57
@desc: custom dataset class
'''


from typing import Optional, Callable, List

import os
import os.path as osp
import shutil

import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

from load_data import my_read_tu_data

#https://blog.csdn.net/m0_59247846/article/details/118227612?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170167755716800222833552%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170167755716800222833552&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-118227612-null-null.142^v96^pc_search_result_base1&utm_term=InMemoryDataset&spm=1018.2226.3001.4187
class MyBlockChain_TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False, use_node_importance=False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        ####################################
        print(self.data, self.slices)
        #Data(x=[6274, 14888], edge_index=[2, 10875], edge_attr=[10875, 2], y=[146])
        ####################################
        if self.data.x is not None:
            if not use_node_attr and not use_node_importance:         # replace with identity matrix
                self.data.x = torch.eye(self.data.x.size(0))
            elif not use_node_importance:
                num_node_importance = self.num_node_importance
                self.data.x = self.data.x[:,:-num_node_importance]    # drop node importance
            elif not use_node_attr:
                num_node_attributes = self.num_node_attributes
                self.data.x = self.data.x[:, num_node_attributes:]    # drop node attribute



        if self.data.edge_attr is not None and not use_edge_attr:
            # num_edge_attributes = self.num_edge_attributes
            # self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
            self.data.edge_attr = torch.ones_like(self.data.edge_attr)     # replace with 1s matrix





    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, name)

    @property
    def num_node_labels(self) -> int:
        # if self.data.x is None:
        #     return 0
        # for i in range(self.data.x.size(1)):
        #     x = self.data.x[:, i:]
        #     if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
        #         return self.data.x.size(1) - i
        return 0

    @property
    def num_node_importance(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, -i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels - self.num_node_importance

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator', 'graph_labels', 'node_attributes', 'edge_attributes', 'node_importance_labels']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        # url = self.cleaned_url if self.cleaned else self.url
        # folder = osp.join(self.root, self.name)
        # path = download_url(f'{url}/{self.name}.zip', folder)
        # extract_zip(path, folder)
        # os.unlink(path)
        # shutil.rmtree(self.raw_dir)
        # os.rename(osp.join(folder, self.name), self.raw_dir)
        return


    #https://blog.csdn.net/misite_J/article/details/118282680?ops_request_misc=&request_id=&biz_id=102&utm_term=InMemoryDataset.collate()&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-118282680.142^v96^pc_search_result_base1&spm=1018.2226.3001.4187
    def process(self):
        self.data, self.slices = my_read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'



from torch_scatter import scatter
def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data


if __name__ == '__main__':
    from functools import partial
    transform = partial(extract_node_feature, reduce='add')
    dataset_v2 = MyBlockChain_TUDataset(root='./data/eth/ico_wallets/2hop-20/averVolume/', name='ETHG',
                                        use_node_attr=True,
                                        use_node_importance=False,
                                        use_edge_attr= False,
                                        transform=None)
    print(dataset_v2)
    print(dataset_v2[0])
    print(dataset_v2[0].edge_index)

    my_read_tu_data('./data/eth/ico_wallets/2hop-20/averVolume/raw', 'ETHG')



    #####################
    num_edge_list=[]
    num_node_list=[]
    for i in range(len(dataset_v2)):
        num_edge_list.append(len(dataset_v2[i].edge_index[0]))
        num_node_list.append(len(dataset_v2[i].x))

    np.savetxt('./test/num_edge_list.csv', num_edge_list, fmt='%d')
    np.savetxt('./test/num_node_list.csv', num_node_list, fmt='%d')