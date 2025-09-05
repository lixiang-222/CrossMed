from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import os
from preprocess.data_load import load_preprocessed_data, save_preprocessed_data
import torch
import torch.nn as nn
from tqdm import tqdm


class RelationWeight(nn.Module):
    """
    RelationEmbedding类用于创建节点类型之间的嵌入矩阵。该类接受源节点类型和目标节点类型的名称列表，
    以及每种节点类型的节点数目字典和嵌入维度，然后初始化节点类型之间的嵌入矩阵。

    参数：
    src_names (list): 源节点类型的名称列表。
    tgt_names (list): 目标节点类型的名称列表。
    num_nodes_dict (dict): 每种节点类型的节点数目字典。例如：{'A': 5, 'B': 3, 'C': 4, 'D': 6}。
    dim (int): 嵌入向量的维度。

    属性：
    embeddings (nn.ParameterDict): 包含所有节点类型之间的嵌入矩阵。键是节点类型对的字符串表示，值是对应的嵌入矩阵。
    edge_types (list): 包含所有边类型的列表，每个元素是一个三元组，表示（源节点类型，'to'，目标节点类型）。

    方法：
    get_embeddings():
        返回一个字典，键是节点类型对的三元组，值是对应的嵌入矩阵。

    get_edge_types():
        返回所有边类型的列表。

    用法示例：
    src_names = ['A']
    tgt_names = ['A', 'B', 'C', 'D']
    num_nodes_dict = {'A': 5, 'B': 3, 'C': 4, 'D': 6}
    dim = 10

    model = RelationEmbedding(src_names, tgt_names, num_nodes_dict, dim)
    embeddings = model.get_embeddings()
    edge_types = model.get_edge_types()

    print("Edge Types:", edge_types)
    for key, value in embeddings.items():
        print(f"{key}: {value.shape}")
    """

    def __init__(self, src_names, tgt_names, num_nodes_dict, causal_model, args):
        super().__init__()
        self.weights = nn.ParameterDict()
        self.edge_types = []
        # 这里的num_nodes_dict是一个字典，包含每个节点类型的数目，例如：{'A': 5, 'B': 3, 'C': 4, 'D': 6}
        for src in src_names:
            for tgt in tgt_names:
                key = (src, 'to', tgt)
                num_src_nodes = num_nodes_dict[src]
                num_tgt_nodes = num_nodes_dict[tgt]
                # todo: 这里要改
                self.weights[str(key)] = nn.Parameter(torch.ones(num_src_nodes, num_tgt_nodes))
                self.edge_types.append(key)

                """下面是改完的"""
                # if src == tgt:
                #     self.weights[str(key)] = nn.Parameter(torch.ones(num_src_nodes, num_tgt_nodes))
                # else:
                #     if args.task == 'drug_rec':
                #         if 'drug' in tgt:
                #             causal_effects_tensor = (
                #                 torch.tensor(causal_model.causal_effects_matrixs[f'{src}_drugs'].values,
                #                              dtype=torch.float32)
                #             )
                #             self.weights[str(key)] = nn.Parameter(causal_effects_tensor)
                #         else:
                #             causal_effects_tensor = (
                #                 torch.tensor(causal_model.causal_effects_matrixs[f'{src}_{tgt}'].values,
                #                              dtype=torch.float32)
                #             )
                #             self.weights[str(key)] = nn.Parameter(causal_effects_tensor)
                #     else:
                #         if 'cond' in tgt:
                #             causal_effects_tensor = (
                #                 torch.tensor(causal_model.causal_effects_matrixs[f'{src}_conditions'].values,
                #                              dtype=torch.float32)
                #             )
                #             self.weights[str(key)] = nn.Parameter(causal_effects_tensor)
                #         else:
                #             causal_effects_tensor = (
                #                 torch.tensor(causal_model.causal_effects_matrixs[f'{src}_{tgt}'].values,
                #                              dtype=torch.float32)
                #             )
                #             self.weights[str(key)] = nn.Parameter(causal_effects_tensor)
                #
                # self.edge_types.append(key)

        # 这里是搞得monitoring-monitoring
        for src in src_names:
            for tgt in src_names:
                if src != tgt:  # 避免重复添加相同类型节点之间的关系
                    key = (tgt, 'to', src)
                    num_tgt_nodes = num_nodes_dict[tgt]
                    num_src_nodes = num_nodes_dict[src]
                    self.weights[str(key)] = nn.Parameter(torch.ones(num_tgt_nodes, num_src_nodes))
                    self.edge_types.append(key)

        # 这里是搞得visit-monitoring
        for tgt in tgt_names:
            for src in src_names:
                if src != tgt:  # 避免重复添加相同类型节点之间的关系
                    key = (tgt, 'to', src)
                    num_tgt_nodes = num_nodes_dict[tgt]
                    num_src_nodes = num_nodes_dict[src]
                    self.weights[str(key)] = nn.Parameter(torch.ones(num_tgt_nodes, num_src_nodes))
                    self.edge_types.append(key)

        # 这里是搞得visit-visit
        for src in tgt_names:
            for tgt in tgt_names:
                if src != tgt:  # 避免重复添加相同类型节点之间的关系
                    key = (tgt, 'to', src)
                    num_tgt_nodes = num_nodes_dict[tgt]
                    num_src_nodes = num_nodes_dict[src]
                    self.weights[str(key)] = nn.Parameter(torch.ones(num_tgt_nodes, num_src_nodes))
                    self.edge_types.append(key)

    def get_edge_types(self):
        return self.edge_types


class VisitGraph(torch.utils.data.Dataset):
    """
    这是一个提前构图的类
    这里为每个visit生成了一个graph，每个graph中包含了多个monitor的信息
    图中每个时间步中monitor一直在变化，visit都相同
    提前构图的过程中，只根据有多少个monitor，构建每个图中的边，以及图中attr的信息，不包含节点本身的信息
    """

    def __init__(self, Tokenizer_visit_event, Tokenizer_monitor_event, dataset, ca, args):
        self.visit_event_token = Tokenizer_visit_event
        self.monitor_event_token = Tokenizer_monitor_event
        self.feature_visit_event_keys = Tokenizer_visit_event.keys()
        self.feature_monitor_event_keys = Tokenizer_monitor_event.keys()
        self.dataset = dataset

        num_nodes_dict = {}
        for feature_key in self.feature_visit_event_keys:
            num_nodes_dict[feature_key] = self.visit_event_token[feature_key].get_vocabulary_size()
        for feature_key in ['lab_item', 'inj_item']:
            num_nodes_dict[feature_key] = self.monitor_event_token[feature_key].get_vocabulary_size()

        feature_names = list(self.feature_visit_event_keys) + ['lab_item', 'inj_item']
        self.relation_attr = (
            RelationWeight(src_names=feature_names, tgt_names=feature_names, num_nodes_dict=num_nodes_dict,
                           causal_model=ca, args=args)
        )

        self.all_data = self._process_()

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def construct_edge_index(self, lab_monitor_length, inj_monitor_length, src_name1, src_name2):
        tgt_name1, tgt_name2, tgt_name3 = self.feature_visit_event_keys

        edge_index = {
            # 对于第一幅图，src1(lab)的图
            # monitoring-visit
            (src_name1, 'to', tgt_name1): [],
            (src_name1, 'to', tgt_name2): [],
            (src_name1, 'to', tgt_name3): [],
            # monitoring-monitoring
            (src_name1, 'to', src_name1): [],
            # visit-monitoring
            (tgt_name1, 'to', src_name1): [],
            (tgt_name2, 'to', src_name1): [],
            (tgt_name3, 'to', src_name1): [],
            # visit-visit
            (tgt_name1, 'to', tgt_name2): [],
            (tgt_name2, 'to', tgt_name1): [],
            (tgt_name1, 'to', tgt_name3): [],
            (tgt_name3, 'to', tgt_name1): [],
            (tgt_name3, 'to', tgt_name2): [],
            (tgt_name2, 'to', tgt_name3): [],

            # 对于第二幅图，src2(inj)的图
            # monitoring-visit
            (src_name2, 'to', tgt_name1): [],
            (src_name2, 'to', tgt_name2): [],
            (src_name2, 'to', tgt_name3): [],
            # monitoring-monitoring
            (src_name2, 'to', src_name2): [],
            # visit-monitoring
            (tgt_name1, 'to', src_name2): [],
            (tgt_name2, 'to', src_name2): [],
            (tgt_name3, 'to', src_name2): [],
            # visit-visit
            (tgt_name1, 'to', tgt_name2): [],
            (tgt_name2, 'to', tgt_name1): [],
            (tgt_name1, 'to', tgt_name3): [],
            (tgt_name3, 'to', tgt_name1): [],
            (tgt_name3, 'to', tgt_name2): [],
            (tgt_name2, 'to', tgt_name3): [],

            # 对于两幅图之间的链接
            (src_name1, 'to', src_name2): [],
            (src_name2, 'to', src_name1): [],
        }

        # 构建lab相关的图
        for i in range(lab_monitor_length):
            """monitoring-visit"""
            # A1->B1
            edge_index[(src_name1, 'to', tgt_name1)].append([i, i])
            # A1->C1
            edge_index[(src_name1, 'to', tgt_name2)].append([i, i])
            # A1->D1
            edge_index[(src_name1, 'to', tgt_name3)].append([i, i])

            """next step"""
            if i < lab_monitor_length - 1:
                """monitoring-monitoring"""
                # A1->A2
                edge_index[(src_name1, 'to', src_name1)].append([i, i + 1])
                """visit-monitoring"""
                # B1->A2
                edge_index[(tgt_name1, 'to', src_name1)].append([i, i + 1])
                # C1->A2
                edge_index[(tgt_name2, 'to', src_name1)].append([i, i + 1])
                # D1->A2
                edge_index[(tgt_name3, 'to', src_name1)].append([i, i + 1])

        # 构建inj相关的图
        for i in range(inj_monitor_length):
            """monitoring-visit"""
            # A1->B1
            edge_index[(src_name2, 'to', tgt_name1)].append([i, i])
            # A1->C1
            edge_index[(src_name2, 'to', tgt_name2)].append([i, i])
            # A1->D1
            edge_index[(src_name2, 'to', tgt_name3)].append([i, i])

            """next step"""
            if i < inj_monitor_length - 1:
                """monitoring-monitoring"""
                # A1->A2
                edge_index[(src_name2, 'to', src_name2)].append([i, i + 1])
                """visit-monitoring"""
                # B1->A2
                edge_index[(tgt_name1, 'to', src_name2)].append([i, i + 1])
                # C1->A2
                edge_index[(tgt_name2, 'to', src_name2)].append([i, i + 1])
                # D1->A2
                edge_index[(tgt_name3, 'to', src_name2)].append([i, i + 1])

        # 构建两幅图之间的链接
        if inj_monitor_length < lab_monitor_length:
            for i in range(inj_monitor_length):
                """monitoring-monitoring"""
                edge_index[(src_name2, 'to', src_name1)].append([i, i])

                """visit-visit"""
                # B1->C1
                edge_index[(tgt_name1, 'to', tgt_name2)].append([i, i])
                # C1->B1
                edge_index[(tgt_name2, 'to', tgt_name1)].append([i, i])
                # B1->D1
                edge_index[(tgt_name1, 'to', tgt_name3)].append([i, i])
                # D1->B1
                edge_index[(tgt_name3, 'to', tgt_name1)].append([i, i])
                # C1->D1
                edge_index[(tgt_name2, 'to', tgt_name3)].append([i, i])
                # D1->C1
                edge_index[(tgt_name3, 'to', tgt_name2)].append([i, i])
        else:
            for i in range(lab_monitor_length):
                """monitoring-monitoring"""
                edge_index[(src_name2, 'to', src_name1)].append([i, i])

                """visit-visit"""
                # B1->C1
                edge_index[(tgt_name1, 'to', tgt_name2)].append([i, i])
                # C1->B1
                edge_index[(tgt_name2, 'to', tgt_name1)].append([i, i])
                # B1->D1
                edge_index[(tgt_name1, 'to', tgt_name3)].append([i, i])
                # D1->B1
                edge_index[(tgt_name3, 'to', tgt_name1)].append([i, i])
                # C1->D1
                edge_index[(tgt_name2, 'to', tgt_name3)].append([i, i])
                # D1->C1
                edge_index[(tgt_name3, 'to', tgt_name2)].append([i, i])

        # 将边列表转换为张量
        for key in edge_index.keys():
            edge_index[key] = torch.tensor(edge_index[key], dtype=torch.long).t().contiguous()

        return edge_index

    def construct_node_features(self, lab_monitor_length, inj_monitor_length, patient, visit_id, lab_key, inj_key):
        graph_node_dict = {}

        # # 添加 monitor 特征
        # x = self.monitor_event_token[feature_key1].batch_encode_2d(
        #     patient[feature_key1][visit_id]
        # )
        # graph_node_dict[feature_key1] = x

        # 添加 lab monitor 特征
        graph_node_dict[lab_key] = []
        for monitor_id in range(lab_monitor_length):
            x = self.monitor_event_token[lab_key].convert_tokens_to_indices(
                patient[lab_key][visit_id][monitor_id]
            )
            graph_node_dict[lab_key].append(x)

        # 添加 inj monitor 特征
        graph_node_dict[inj_key] = []
        for monitor_id in range(inj_monitor_length):
            x = self.monitor_event_token[inj_key].convert_tokens_to_indices(
                patient[inj_key][visit_id][monitor_id]
            )
            graph_node_dict[inj_key].append(x)

        # 添加 visit 特征
        if lab_monitor_length < inj_monitor_length:
            monitor_length = inj_monitor_length
        else:
            monitor_length = lab_monitor_length

        for feature_key in self.feature_visit_event_keys:
            x = self.visit_event_token[feature_key].convert_tokens_to_indices(
                patient[feature_key][visit_id]
            )
            extended_x = [x] * monitor_length
            graph_node_dict[feature_key] = extended_x

        # for key in graph_node_dict.keys():
        #     graph_node_dict[key] = torch.tensor(graph_node_dict[key], dtype=torch.long)

        return graph_node_dict

    def construct_edge_attr(self, graph_node, edge_index, lab_key, inj_key):
        edge_attr = {}
        attrs = self.relation_attr.weights

        for key in edge_index.keys():
            edges = edge_index[key].t().tolist()  # 将边转换为列表
            attr_list = []
            for edge in edges:
                src, tgt = edge
                src_items = graph_node[key[0]][src]  # 获取源节点的项目集合
                tgt_items = graph_node[key[2]][tgt]  # 获取目标节点的项目集合

                # 判断是否为空列表，如果为空则生成同维度的0向量
                if not src_items or not tgt_items:
                    attr_sum = torch.zeros(1)
                else:
                    # 使用高级索引一次性提取嵌入
                    src_items = torch.tensor(src_items)
                    tgt_items = torch.tensor(tgt_items)
                    # TODO:这里可以用mean，sum，或者是我们写在文章里的那种聚合办法，现在写的事mean意思就是完全不用
                    attr_sum = attrs[str(key)][src_items][:, tgt_items].mean()

                attr_list.append(attr_sum)

            if not attr_list:  # 如果attr_list为空，创建一个大小为(0, embeddings[key].size(2))的tensor
                edge_attr[key] = torch.tensor([], dtype=torch.float).contiguous()
            else:
                edge_attr[key] = torch.tensor(attr_list, dtype=torch.float).contiguous()

        return edge_attr

    def _process_(self):
        """
        构图的方法就是把每个visit的图都构出来，然后放到一个list里面
        其中
        node_feature是个假的，这里面只是存了每个nodefeature跟哪些节点是对着的，没有赋予一个真的tensor，具体的嵌入模型里面再给
        edge_index是正常的tensor
        edge_attr这里没有写，也是后面具体模型里面再给
        """
        graph_list = []

        for patient in tqdm(self.dataset):

            # 一个visit_graph_list，里面存的是一个patient的所有visit的图
            visit_graph_list = []
            for visit_id in range(len(patient['procedures'])):
                # 获取 graph_node_dict[feature_key1] 的长度
                lab_monitor_length = len(patient['lab_item'][visit_id])
                inj_monitor_length = len(patient['inj_item'][visit_id])
                # 构造边索引
                edge_index = self.construct_edge_index(lab_monitor_length, inj_monitor_length, 'lab_item', 'inj_item')
                # 构建节点特征（这里没有特征，就是节点里面有哪些event）
                graph_node = self.construct_node_features(lab_monitor_length, inj_monitor_length, patient, visit_id,
                                                          'lab_item', 'inj_item')
                # 构建边特征
                edge_attr = self.construct_edge_attr(graph_node, edge_index, 'lab_item', 'inj_item')

                # 构建 HeteroData 对象
                data = HeteroData()
                # 添加节点特征到 HeteroData 对象
                for key, value in graph_node.items():
                    data[key].x = value
                # 添加边和边的特征，添加到 HeteroData 对象
                for (src, relation, dst), index in edge_index.items():
                    data[(src, relation, dst)].edge_index = index
                    data[(src, relation, dst)].edge_attr = edge_attr[(src, relation, dst)]

                # 添加小图到列表中
                visit_graph_list.append(data)

            # 把图加进去
            graph_list.append({'monitoring_visit_graph': visit_graph_list})

        return graph_list


class PatientGraph(torch.utils.data.Dataset):

    def __init__(self, Tokenizer_visit_event, dataset, ca, args):
        self.visit_event_token = Tokenizer_visit_event
        self.feature_visit_event_keys = Tokenizer_visit_event.keys()
        self.feature_info_keys = ['age', 'weight']
        self.dataset = dataset

        self.relation_attr = nn.ModuleDict()
        num_nodes_dict = {}
        for feature_key in self.feature_visit_event_keys:
            num_nodes_dict[feature_key] = self.visit_event_token[feature_key].get_vocabulary_size()
        for feature_key in self.feature_info_keys:
            # 这里人为规定了60种，体重是0~300，每5个为1种，年龄是1~120，每2个为一种，后面都用分段的方法来判定他的种类
            # 这里写61是为了多一个pad为0的位置
            num_nodes_dict[feature_key] = 1
            names = list(self.feature_visit_event_keys) + [feature_key]
            self.relation_attr[feature_key] = (
                RelationWeight(src_names=[feature_key], tgt_names=names, num_nodes_dict=num_nodes_dict,
                               causal_model=ca, args=args)
            )
        self.all_data = self._process_()

    def classify_value(self, value, value_type):
        if value_type == 'age':
            max_value = 120
            num_classes = 60
        elif value_type == 'weight':
            max_value = 300
            num_classes = 60
        else:
            raise ValueError("Unsupported value type. Please use 'age' or 'weight'.")
        value = float(value)
        if value < 0:
            return 1
        elif value >= max_value:
            return num_classes
        else:
            return int(value // (max_value / num_classes)) + 1

    def construct_edge_index(self, visit_length, src_name):
        tgt_name1, tgt_name2, tgt_name3 = self.feature_visit_event_keys

        edge_index = {
            (src_name, 'to', tgt_name1): [],
            (src_name, 'to', tgt_name2): [],
            (src_name, 'to', tgt_name3): [],
            (src_name, 'to', src_name): [],
            (tgt_name1, 'to', src_name): [],
            (tgt_name2, 'to', src_name): [],
            (tgt_name3, 'to', src_name): [],
        }

        for i in range(visit_length):
            # A1->B1
            edge_index[(src_name, 'to', tgt_name1)].append([i, i])
            # A1->C1
            edge_index[(src_name, 'to', tgt_name2)].append([i, i])
            # A1->D1
            edge_index[(src_name, 'to', tgt_name3)].append([i, i])

            if i < visit_length - 1:
                # A1->A2
                edge_index[(src_name, 'to', src_name)].append([i, i + 1])
                # B1->A2
                edge_index[(tgt_name1, 'to', src_name)].append([i, i + 1])
                # C1->A2
                edge_index[(tgt_name2, 'to', src_name)].append([i, i + 1])
                # D1->A2
                edge_index[(tgt_name3, 'to', src_name)].append([i, i + 1])

        # 将边列表转换为张量
        for key in edge_index.keys():
            edge_index[key] = torch.tensor(edge_index[key], dtype=torch.long).t().contiguous()

        return edge_index

    def construct_node_features(self, visit_length, patient, feature_key1):
        graph_node_dict = {}

        # 先给每种东西建立一个列表
        for feature_key in list(self.feature_visit_event_keys) + [feature_key1]:
            graph_node_dict[feature_key] = []

        for visit_id in range(visit_length):
            # 添加 info 特征
            # x = self.classify_value(patient[feature_key1][visit_id], feature_key1)
            # x = float(patient[feature_key1][visit_id])
            x = -1  # 这里把年龄设置0，不需要年龄具体的数值
            # 这里为了配合后面的东西，把info也变成列表形式，加入的是[x]，但实际上只有一个元素
            graph_node_dict[feature_key1].append([x])

            # 添加 visit 特征
            for feature_key in self.feature_visit_event_keys:
                x = self.visit_event_token[feature_key].convert_tokens_to_indices(
                    patient[feature_key][visit_id]
                )
                graph_node_dict[feature_key].append(x)

        # for key in graph_node_dict.keys():
        #     graph_node_dict[key] = torch.tensor(graph_node_dict[key], dtype=torch.long)

        return graph_node_dict

    def construct_edge_attr(self, graph_node, edge_index, feature_key):
        edge_attr = {}
        attrs = self.relation_attr[feature_key].weights

        for key in edge_index.keys():
            edges = edge_index[key].t().tolist()  # 将边转换为列表
            attr_list = []
            for edge in edges:
                src, tgt = edge
                src_items = graph_node[key[0]][src]  # 获取源节点的项目集合
                tgt_items = graph_node[key[2]][tgt]  # 获取目标节点的项目集合

                # 判断是否为空列表，如果为空则生成同维度的0向量
                if not src_items or not tgt_items:
                    attr_sum = torch.zeros(1)
                else:
                    # 使用高级索引一次性提取嵌入
                    src_items = torch.tensor(src_items)
                    tgt_items = torch.tensor(tgt_items)
                    # TODO:这里可以用mean，sum，或者是我们写在文章里的那种聚合办法
                    if -1 in src_items:
                        attr_sum = attrs[str(key)][:, tgt_items].mean()
                    elif -1 in tgt_items:
                        attr_sum = attrs[str(key)][src_items].mean()
                    else:
                        exit('?')

                attr_list.append(attr_sum)

            if not attr_list:  # 如果attr_list为空，创建一个大小为(0, embeddings[key].size(2))的tensor
                edge_attr[key] = torch.tensor([], dtype=torch.float).contiguous()
            else:
                edge_attr[key] = torch.tensor(attr_list, dtype=torch.float).contiguous()

        return edge_attr

    def _process_(self):
        """
        构图的方法就是把每个visit的图都构出来，然后放到一个list里面
        其中
        node_feature是个假的，这里面只是存了每个nodefeature跟哪些节点是对着的，没有赋予一个真的tensor，具体的嵌入模型里面再给
        edge_index是正常的tensor
        edge_attr这里没有写，也是后面具体模型里面再给
        """
        graph_list = []

        for patient in tqdm(self.dataset):
            patient_graph_dict = {}
            for feature_key in self.feature_info_keys:
                visit_length = len(patient[feature_key])
                # 构造边索引
                edge_index = self.construct_edge_index(visit_length, feature_key)
                # 构建节点特征（这里没有特征，就是节点里面有哪些event）
                graph_node = self.construct_node_features(visit_length, patient, feature_key)
                # 构建边特征
                edge_attr = self.construct_edge_attr(graph_node, edge_index, feature_key)

                # 构建 HeteroData 对象
                data = HeteroData()
                # 添加节点特征到 HeteroData 对象
                for key, value in graph_node.items():
                    data[key].x = value
                # 添加边和边的特征，添加到 HeteroData 对象
                for (src, relation, dst), index in edge_index.items():
                    data[(src, relation, dst)].edge_index = index
                    data[(src, relation, dst)].edge_attr = edge_attr[(src, relation, dst)]

                if feature_key == 'age':
                    patient_graph_dict['age_patient_graph'] = data
                elif feature_key == 'weight':
                    patient_graph_dict['weight_patient_graph'] = data
                else:
                    exit("没有这个东西")

            graph_list.append(patient_graph_dict)

        return graph_list


def dataset_collate(dataset, Tokenizers_event, Tokenizers_monitor, ca, args):
    graph_dataset = VisitGraph(Tokenizers_event, Tokenizers_monitor, dataset.samples, ca, args).all_data
    combined_dataset = combine_datasets(dataset.samples, graph_dataset)
    dataset.samples = combined_dataset
    print("visit图构建完了")

    graph_dataset = PatientGraph(Tokenizers_event, dataset.samples, ca, args).all_data
    combined_dataset = combine_datasets(dataset.samples, graph_dataset)
    dataset.samples = combined_dataset
    print("patient图构建完了")

    return dataset


def combine_datasets(sequence_dataset, graph_dataset):
    combined_dataset = []
    for seq_data, graph_data in zip(sequence_dataset, graph_dataset):
        combined_data = {**seq_data, **graph_data}
        combined_dataset.append(combined_data)
    return combined_dataset


def process_data_with_graph(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, ca, args):
    """数据加入图"""

    # 判断是否是developer
    if args.developer:
        processed_data_path = f'data/{args.dataset}/processed_data/{args.task}/processed_graph_data_developer.pkl'
    else:
        processed_data_path = f'data/{args.dataset}/processed_data/{args.task}/processed_graph_data.pkl'

    # 判断是否有处理好的数据
    if os.path.exists(processed_data_path):
        print("Processed graph data exists, loading directly.")
        task_dataset_with_graph = load_preprocessed_data(processed_data_path)
    else:
        print("Graph data not processed, reconstructing the graph.")
        task_dataset_with_graph = dataset_collate(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, ca,
                                                  args)
        save_preprocessed_data(task_dataset_with_graph, processed_data_path)

    return task_dataset_with_graph
