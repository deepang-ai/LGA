import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from ogb.utils.url import decide_download, download_url, extract_zip
# from ogb.io.read_graph_pyg import read_graph_pyg


########################################
from ogb.io.read_graph_raw import read_binary_graph_raw
from tqdm import tqdm
from torch_geometric.data import Data


def read_csv_graph_raw(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[]):

    '''
    raw_dir: path to the raw directory
    add_inverse_edge (bool): whether to add inverse edge or not

    return: graph_list, which is a list of graphs.
    Each graph is a dictionary, containing edge_index, edge_feat, node_feat, and num_nodes
    edge_feat and node_feat are optional: if a graph does not contain it, we will have None.

    additional_node_files and additional_edge_files must be in the raw directory.
    - The name should be {additional_node_file, additional_edge_file}.csv.gz
    - The length should be num_nodes or num_edges

    additional_node_files must start from 'node_'
    additional_edge_files must start from 'edge_'


    '''

    print('Loading necessary files...')
    print('This might take a while.')
    # loading necessary files
    try:
        edge = pd.read_csv(osp.join(raw_dir, 'edge.csv.gz'), compression='gzip', header=None).values.T.astype(
            np.int64)  # (2, num_edge) numpy array
        num_node_list = \
        pd.read_csv(osp.join(raw_dir, 'num-node-list.csv.gz'), compression='gzip', header=None).astype(np.int64)[
            0].tolist()  # (num_graph, ) python list
        num_edge_list = \
        pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv.gz'), compression='gzip', header=None).astype(np.int64)[
            0].tolist()  # (num_edge, ) python list

    except FileNotFoundError:
        raise RuntimeError('No necessary file')

    try:
        node_feat = pd.read_csv(osp.join(raw_dir, 'node-feat.csv.gz'), compression='gzip', header=None).values
        if 'int' in str(node_feat.dtype):
            node_feat = node_feat.astype(np.int64)
        else:
            # float
            node_feat = node_feat.astype(np.float32)
    except FileNotFoundError:
        node_feat = None

    try:
        edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv.gz'), compression='gzip', header=None).values
        if 'int' in str(edge_feat.dtype):
            edge_feat = edge_feat.astype(np.int64)
        else:
            # float
            edge_feat = edge_feat.astype(np.float32)

    except FileNotFoundError:
        edge_feat = None

    additional_node_info = {}

    for additional_file in additional_node_files:
        assert (additional_file[:5] == 'node_')

        # hack for ogbn-proteins
        if additional_file == 'node_species' and osp.exists(osp.join(raw_dir, 'species.csv.gz')):
            os.rename(osp.join(raw_dir, 'species.csv.gz'), osp.join(raw_dir, 'node_species.csv.gz'))

        temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), compression='gzip', header=None).values

        if 'int' in str(temp.dtype):
            additional_node_info[additional_file] = temp.astype(np.int64)
        else:
            # float
            additional_node_info[additional_file] = temp.astype(np.float32)

    additional_edge_info = {}
    for additional_file in additional_edge_files:
        assert (additional_file[:5] == 'edge_')
        temp = pd.read_csv(osp.join(raw_dir, additional_file + '.csv.gz'), compression='gzip', header=None).values

        if 'int' in str(temp.dtype):
            additional_edge_info[additional_file] = temp.astype(np.int64)
        else:
            # float
            additional_edge_info[additional_file] = temp.astype(np.float32)

    graph_list = []
    num_node_accum = 0
    num_edge_accum = 0

    print('Processing graphs...')
    for num_node, num_edge in tqdm(zip(num_node_list, num_edge_list), total=len(num_node_list)):

        graph = dict()

        ### handling edge
        if add_inverse_edge:
            ### duplicate edge
            duplicated_edge = np.repeat(edge[:, num_edge_accum:num_edge_accum + num_edge], 2, axis=1)     #np.repeat, axis=1表示对行操作，增加的是列
            # np.savetxt('./test/duplicated_edge.csv', duplicated_edge, delimiter=',')
            duplicated_edge[0, 1::2] = duplicated_edge[1, 0::2]                                                   #[i:j:s]s表示步长
            duplicated_edge[1, 1::2] = duplicated_edge[0, 0::2]
            # np.savetxt('./test/duplicated_edge_v2.csv', duplicated_edge, delimiter=',')

            graph['edge_index'] = duplicated_edge

            if edge_feat is not None:
                graph['edge_feat'] = np.repeat(edge_feat[num_edge_accum:num_edge_accum + num_edge], 2, axis=0)
            else:
                graph['edge_feat'] = None

            for key, value in additional_edge_info.items():
                graph[key] = np.repeat(value[num_edge_accum:num_edge_accum + num_edge], 2, axis=0)

        else:
            graph['edge_index'] = edge[:, num_edge_accum:num_edge_accum + num_edge]

            if edge_feat is not None:
                graph['edge_feat'] = edge_feat[num_edge_accum:num_edge_accum + num_edge]
            else:
                graph['edge_feat'] = None

            for key, value in additional_edge_info.items():
                graph[key] = value[num_edge_accum:num_edge_accum + num_edge]

        num_edge_accum += num_edge

        ### handling node
        if node_feat is not None:
            graph['node_feat'] = node_feat[num_node_accum:num_node_accum + num_node]
        else:
            graph['node_feat'] = None

        for key, value in additional_node_info.items():
            graph[key] = value[num_node_accum:num_node_accum + num_node]

        graph['num_nodes'] = num_node
        num_node_accum += num_node

        graph_list.append(graph)

    return graph_list

def read_graph_pyg(raw_dir, add_inverse_edge=False, additional_node_files=[], additional_edge_files=[], binary=False):
    if binary:
        # npz
        graph_list = read_binary_graph_raw(raw_dir, add_inverse_edge)
    else:
        # csv
        graph_list = read_csv_graph_raw(raw_dir, add_inverse_edge, additional_node_files=additional_node_files,
                                        additional_edge_files=additional_edge_files)

    pyg_graph_list = []

    print('Converting graphs into PyG objects...')

    for graph in tqdm(graph_list):
        g = Data()
        g.num_nodes = graph['num_nodes']
        g.edge_index = torch.from_numpy(graph['edge_index'])                      ########################################################################

        del graph['num_nodes']
        del graph['edge_index']

        if graph['edge_feat'] is not None:
            g.edge_attr = torch.from_numpy(graph['edge_feat'])
            del graph['edge_feat']

        if graph['node_feat'] is not None:
            g.x = torch.from_numpy(graph['node_feat'])
            del graph['node_feat']

        for key in additional_node_files:
            g[key] = torch.from_numpy(graph[key])
            del graph[key]

        for key in additional_edge_files:
            g[key] = torch.from_numpy(graph[key])
            del graph[key]

        pyg_graph_list.append(g)

    return pyg_graph_list


##############################

class PygGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        #transform：数据转换函数，接收一个Data对象并返回一个转换后的Data对象，在每一次数据获取过程中都会被执行
        #pre_transform：数据转换函数，接收一个Data对象并返回一个转换后的Data对象，在Data对象被保存到文件前调用
        #pre_filter：检查数据是否要保留的函数，它接收一个Data对象，返回此Data对象是否应该被包含在最终的数据集中，在Data对象被保存到文件前调用
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name  ## original name, e.g., ogbg-molhiv

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col=0,
                                 keep_default_na=False)
            #D:\Apps\anaconda3\envs\SAT\lib\site-packages\ogb\graphproppred
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (
        not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']  ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']

        print("=========================")
        print(self.task_type)
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'

        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)


        self.data, self.slices = torch.load(self.processed_paths[0])

        ################################
        print(self.data, self.slices)
        #Data(num_nodes=38484727, edge_index=[2, 716539644], edge_attr=[716539644, 7], y=[158100, 1])
        ################################

        print(self.meta_info['add_inverse_edge'] == 'True')
        print(self.meta_info['additional node files'])
        print(self.meta_info['additional edge files'])
        print(self.meta_info['binary'])
        print(self.raw_dir)
        data_list_test = read_graph_pyg(self.raw_dir, add_inverse_edge=True,
                                   additional_node_files=[],
                                   additional_edge_files=[], binary=False)


    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype=torch.long), 'valid': torch.tensor(valid_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
    #该方法返回数据集原始文件的文件名列表，原始文件应存在于raw_dir文件夹，否则调用download()函数下载文件到raw_dir文件夹
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
    #该方法返回处理过的数据文件的文件名列表，处理文件应存在于processed_dir文件夹，否则调用process()函数对原始文件进行处理并保存到相应文件夹
        return 'geometric_data_processed.pt'

    def download(self):
    #下载数据集原始文件到raw_dir文件夹
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        ### read pyg graph list
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge,
                                   additional_node_files=additional_node_files,
                                   additional_edge_files=additional_edge_files, binary=self.binary)

        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                                header=None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                          header=None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1, -1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        #collate()方法将列表合并成一个大的Data对象，该过程将所有对象连接成一个大数据对象，同时，返回一个切片字典以便从该对象重建单个小的对象

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])



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
    # pyg_dataset = PygGraphPropPredDataset(name = 'ogbg-molpcba')
    # print(pyg_dataset.num_classes)
    # split_index = pyg_dataset.get_idx_split()
    # print(pyg_dataset)
    # print(pyg_dataset[0])
    # print(pyg_dataset[0].y)
    # print(pyg_dataset[0].y.dtype)
    # print(pyg_dataset[0].edge_index)
    # print(pyg_dataset[split_index['train']])
    # print(pyg_dataset[split_index['valid']])
    # print(pyg_dataset[split_index['test']])


    from functools import partial
    transform = partial(extract_node_feature, reduce='add')
    pyg_dataset = PygGraphPropPredDataset(name='ogbg-ppa',root='./datasets/',transform=transform)
    # print(pyg_dataset.num_classes)
    # split_index = pyg_dataset.get_idx_split()

    # print(pyg_dataset[0].node_is_attributed)
    # print([pyg_dataset[i].x[1] for i in range(100)])
    # print(pyg_dataset[0].y)
    # print(pyg_dataset[0].edge_index)
    # print(pyg_dataset[split_index['train']])
    # print(pyg_dataset[split_index['valid']])
    # print(pyg_dataset[split_index['test']])

    # from torch_geometric.loader import DataLoader
    # loader = DataLoader(pyg_dataset, batch_size=32, shuffle=True)
    # for batch in loader:
    #     print(batch)
    #     print(batch.y)
    #     print(len(batch.y))

    #     exit(-1)

