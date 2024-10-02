import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, GraphNorm
from torch_geometric.data import HeteroData, Batch


def create_common_pad_graph(edge_types):
    data = HeteroData()
    for edge_type in edge_types:
        data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[edge_type].edge_attr = torch.empty((0,), dtype=torch.long)
    return data


def process_visit_graphs(patients_graphs, input_features, device):
    # 初始化一个列表来保存所有图
    all_graphs = []
    i = 0
    # 遍历每个患者及其访问的图
    for visit_graphs in patients_graphs:
        for visit_graph in visit_graphs:
            # 给每个小图赋值
            for key in input_features:
                visit_graph[key].x = input_features[key][i]
            i = i + 1
            all_graphs.append(visit_graph.to(device))

    # 将所有图转换为一个批处理
    batch = Batch.from_data_list(all_graphs)
    return batch


def process_patient_graphs(patients_graphs, input_features, device):
    # 初始化一个列表来保存所有图
    all_graphs = []
    i = 0
    # 遍历每个患者及其访问的图
    for patient_graph in patients_graphs:
        # 给每个小图赋值
        for key in input_features:
            patient_graph[key].x = input_features[key][i]
        i = i + 1
        all_graphs.append(patient_graph.to(device))
    # 将所有图转换为一个批处理
    batch = Batch.from_data_list(all_graphs)
    return batch


class RelationEmbedding(nn.Module):
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

    def __init__(self, src_names, tgt_names, num_nodes_dict, dim):
        super().__init__()
        self.embeddings = nn.ParameterDict()
        self.edge_types = []
        # 这里的num_nodes_dict是一个字典，包含每个节点类型的数目，例如：{'A': 5, 'B': 3, 'C': 4, 'D': 6}
        for src in src_names:
            for tgt in tgt_names:
                key = (src, 'to', tgt)
                num_src_nodes = num_nodes_dict[src]
                num_tgt_nodes = num_nodes_dict[tgt]
                self.embeddings[str(key)] = nn.Parameter(torch.randn(num_src_nodes, num_tgt_nodes, dim))
                self.edge_types.append(key)
        for tgt in tgt_names:
            for src in src_names:
                if src != tgt:  # 避免重复添加相同类型节点之间的关系
                    key = (tgt, 'to', src)
                    num_tgt_nodes = num_nodes_dict[tgt]
                    num_src_nodes = num_nodes_dict[src]
                    self.embeddings[str(key)] = nn.Parameter(torch.randn(num_tgt_nodes, num_src_nodes, dim))
                    self.edge_types.append(key)

    def get_embeddings(self):
        return {eval(key): value for key, value in self.embeddings.items()}

    def get_edge_types(self):
        return self.edge_types


class HeteroGNN(torch.nn.Module):
    def __init__(self, src_names, tgt_names, num_nodes_dict, dim, device):
        super(HeteroGNN, self).__init__()
        self.dim = dim
        self.device = device
        self.relation_embedding = RelationEmbedding(src_names, tgt_names, num_nodes_dict, dim)
        self.edge_types = self.relation_embedding.get_edge_types()
        self.conv1 = self.create_hetero_conv(self.edge_types, dim)

    def create_hetero_conv(self, edge_types, dim):
        convs = {}
        for edge_type in edge_types:
            convs[edge_type] = GATv2Conv(dim, dim, edge_dim=1, add_self_loops=False)
        return HeteroConv(convs, aggr='sum')

    def forward(self, data, length):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        out_dict = x_dict

        graph_norm = {key: GraphNorm(self.dim).to(self.device) for key in x_dict.keys()}

        for _ in range(length - 1):
            # 卷积操作
            out_dict = self.conv1(out_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            # out_dict = self.conv1(out_dict, edge_index_dict)
            # ReLU 激活
            out_dict = {key: F.relu(x) for key, x in out_dict.items()}
            # 残差相加
            out_dict = {key: (out_dict[key] + x_dict[key]) / 2 for key in out_dict.keys()}
            # 图归一化
            out_dict = {key: graph_norm[key](x) for key, x in out_dict.items()}

        return out_dict
