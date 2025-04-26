import pandas as pd
import os

if __name__ == "__main__":

    all_category = ['exchange', 'ico_wallets', 'mining', 'phish_hack']
    all_sample = ['averVolume', 'Times', 'Volume']

    data_save_path = './multi_classification/'

    # 读取CSV文件
    for sample in all_sample:

        multi_class_edge = []
        multi_class_edge_feat = []
        multi_class_graph_label = []
        multi_class_node_feat = []
        multi_class_num_edge_list = []
        multi_class_num_node_feat = []

        print("Processing " + sample)

        for category in all_category:

            print("Processing " + category)

            date_root = os.path.join(category, sample, 'raw')
            all_csv = os.listdir(date_root)

            for csv_file in all_csv:

                if csv_file == 'edge.csv':
                    edge = pd.read_csv(os.path.join(date_root, csv_file), header=None)
                    edge_list = edge.values.tolist()
                elif csv_file == 'edge-feat.csv':
                    edge_feat = pd.read_csv(os.path.join(date_root, csv_file), header=None)
                    edge_feat_list = edge_feat.values.tolist()
                elif csv_file == 'graph-label.csv':
                    graph_label = pd.read_csv(os.path.join(date_root, csv_file), header=None)
                    graph_label_list = graph_label.values.tolist()
                elif csv_file == 'node-feat.csv':
                    node_feat = pd.read_csv(os.path.join(date_root, csv_file), header=None)
                    node_feat_list = node_feat.values.tolist()
                elif csv_file == 'num-edge-list.csv':
                    num_edge = pd.read_csv(os.path.join(date_root, csv_file), header=None)
                    num_edge_list = num_edge.values.tolist()
                elif csv_file == 'num-node-list.csv':
                    num_node = pd.read_csv(os.path.join(date_root, csv_file), header=None)
                    num_node_list = num_node.values.tolist()


            sample_graph_label = graph_label_list[:65]
            sample_num_edge_list = num_edge_list[:65]
            sample_num_node_list = num_node_list[:65]

            all_num_edge = sum([i[0] for i in sample_num_edge_list])
            all_num_node = sum([i[0] for i in sample_num_node_list])

            sample_edge = edge_list[:all_num_edge]
            sample_edge_feat = edge_feat_list[:all_num_edge]
            sample_node_feat = node_feat_list[:all_num_node]

            if category == 'ico_wallets':
                new_sample_graph_label = [[i[0]-1] for i in sample_graph_label]
            elif category == 'mining':
                new_sample_graph_label = sample_graph_label
            elif category == 'exchange':
                new_sample_graph_label = [[i[0]+1] for i in sample_graph_label]
            elif category == 'phish_hack':
                new_sample_graph_label = [[i[0] + 2] for i in sample_graph_label]

            multi_class_edge.extend(sample_edge)
            multi_class_edge_feat.extend(sample_edge_feat)
            multi_class_graph_label.extend(new_sample_graph_label)
            multi_class_node_feat.extend(sample_node_feat)
            multi_class_num_edge_list.extend(sample_num_edge_list)
            multi_class_num_node_feat.extend(sample_num_node_list)

            # print(sample_num_edge_list)
            # print(len(sample_graph_label))
            # print(len(sample_num_edge_list))
            # print(len(sample_num_node_list))
            #
            # print(len(sample_edge))
            # print(len(sample_edge_feat))
            # print(len(sample_node_feat))

        if not os.path.exists(os.path.join(data_save_path, sample, 'raw')):
            os.makedirs(os.path.join(data_save_path, sample, 'raw'))

        edge = pd.DataFrame(multi_class_edge)
        edge_feat = pd.DataFrame(multi_class_edge_feat)
        graph_label = pd.DataFrame(multi_class_graph_label)
        node_feat = pd.DataFrame(multi_class_node_feat)
        num_edge_list = pd.DataFrame(multi_class_num_edge_list)
        num_node_list = pd.DataFrame(multi_class_num_node_feat)

        edge.to_csv(os.path.join(data_save_path, sample, 'raw', 'edge.csv'), header=False, index=False)
        edge_feat.to_csv(os.path.join(data_save_path, sample, 'raw', 'edge-feat.csv'), header=False, index=False)
        graph_label.to_csv(os.path.join(data_save_path, sample, 'raw', 'graph-label.csv'), header=False, index=False)
        node_feat.to_csv(os.path.join(data_save_path, sample, 'raw', 'node-feat.csv'), header=False, index=False)
        num_edge_list.to_csv(os.path.join(data_save_path, sample, 'raw', 'num-edge-list.csv'), header=False, index=False)
        num_node_list.to_csv(os.path.join(data_save_path, sample, 'raw', 'num-node-list.csv'), header=False, index=False)
