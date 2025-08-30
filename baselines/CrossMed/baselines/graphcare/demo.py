import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class PatientGraphModel(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(PatientGraphModel, self).__init__()
        self.conv1 = GCNConv(embedding_dim, 128)
        self.conv2 = GCNConv(128, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # GCN 第一层
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        # GCN 第二层
        x = self.conv2(x, edge_index)
        return x


# 创建图节点和边
def create_patient_graph(patient_data, patient_embeddings):
    """
    patient_data: 包含患者就诊事件的数据字典，例如 'procedures', 'conditions', 'drugs_hist'
    patient_embeddings: 对应每个事件的嵌入表示, Tensor, 形状为 (visit, event, embedding_dim)
    """
    all_nodes = []  # 保存所有节点的嵌入
    edge_index = []  # 边的索引
    event_to_node_map = {}  # 记录相同事件在不同就诊中的映射
    node_index = 0  # 当前节点的索引计数

    # 遍历每个就诊中的所有事件
    for visit_idx, visit_events in enumerate(
            zip(patient_data['procedures'], patient_data['conditions'], patient_data['drugs_hist'])):
        visit_event_indices = []  # 存储本次就诊中所有事件的索引

        for event_type_idx, event_type in enumerate(visit_events):  # 遍历每种事件类型
            for event in event_type:  # 遍历每个具体事件
                # 将事件嵌入加入到节点列表中
                all_nodes.append(patient_embeddings[visit_idx][event_type_idx])

                # 构建就诊内全连接边
                for prev_event_idx in visit_event_indices:
                    edge_index.append([prev_event_idx, node_index])
                    edge_index.append([node_index, prev_event_idx])

                visit_event_indices.append(node_index)

                # 如果该事件在其他次就诊中出现过，构建跨就诊边
                if event in event_to_node_map:
                    for previous_occurrence in event_to_node_map[event]:
                        edge_index.append([previous_occurrence, node_index])
                        edge_index.append([node_index, previous_occurrence])

                # 更新事件的映射
                if event not in event_to_node_map:
                    event_to_node_map[event] = []
                event_to_node_map[event].append(node_index)

                node_index += 1

    # 将所有节点和边转换为Tensor
    all_nodes_tensor = torch.stack(all_nodes)  # 形状为 (total_events, embedding_dim)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=all_nodes_tensor, edge_index=edge_index_tensor)


# 模拟数据并创建图
# 模拟数据并创建图
def create_patient_graph_data(batch_data):
    patient_emb_dict = {}

    for patient_idx in range(len(batch_data['patient_id'])):
        # 当前患者的所有就诊事件
        patient_data = {
            'procedures': batch_data['procedures'][patient_idx],
            'conditions': batch_data['conditions'][patient_idx],
            'drugs_hist': batch_data['drugs_hist'][patient_idx]
        }

        # 获取嵌入：根据不同事件类型获取对应的嵌入表
        patient_embeddings = []
        for visit_idx, visit_events in enumerate(
                zip(patient_data['procedures'], patient_data['conditions'], patient_data['drugs_hist'])):
            visit_embeddings = []

            # 遍历每种事件类型，获取相应的嵌入表
            for event_type_idx, (procedure, condition, drug) in enumerate(visit_events):
                procedure_embed = patient_emb_dict_origin['procedures'][int(procedure[0])]  # 根据 procedure 获取嵌入
                condition_embed = patient_emb_dict_origin['conditions'][int(condition[0])]  # 根据 condition 获取嵌入
                drug_embed = patient_emb_dict_origin['drugs_hist'][drug[0]]  # 根据 drug 获取嵌入

                # 合并当前就诊中的事件嵌入
                visit_embeddings.append([procedure_embed, condition_embed, drug_embed])

            # 将该次就诊的嵌入添加到患者嵌入列表中
            patient_embeddings.append(visit_embeddings)

        # 将嵌入转为 tensor
        patient_embeddings = torch.tensor(patient_embeddings, dtype=torch.float)

        # 创建患者图
        graph_data = create_patient_graph(patient_data, patient_embeddings)

        # 使用GCN更新图表示
        gcn_model = PatientGraphModel(embedding_dim=graph_data.x.shape[1], output_dim=128)
        updated_representation = gcn_model(graph_data)

        # 存储更新后的患者表示
        patient_emb_dict[batch_data['patient_id'][patient_idx]] = updated_representation

    return patient_emb_dict


# 假设 batch_data 中包含患者的就诊数据
embedding_dim = 100  # 事件嵌入的维度

# 初始化每种事件的 embedding
condition_embeddings = torch.rand(100, embedding_dim)  # 对 condition 创建嵌入
procedure_embeddings = torch.rand(100, embedding_dim)  # 对 procedure 创建嵌入
drug_embeddings = torch.rand(100, embedding_dim)       # 对 drug 创建嵌入

# 用一个字典保存各个事件类型的 embedding
patient_emb_dict_origin = {
    'conditions': condition_embeddings,  # 每种 condition 的嵌入表
    'procedures': procedure_embeddings,  # 每种 procedure 的嵌入表
    'drugs_hist': drug_embeddings        # 每种 drug 的嵌入表
}

batch_data = {
    'patient_id': ['109', '291'],
    'visit_id': ['158995', '125726'],
    'procedures': [
        [['54'], ['91'], ['89', '63', '58', '54', '222', '70', '216'], ['58'], ['58'], ['58', '54', '222'],
         ['99', '98', '216', '63', '58'], ['58', '216'], ['58'], ['54'], ['58'], ['58', '222']],
        [['45', '63', '47', '61', '231']]
    ],
    'conditions': [
        [['99', '156', '158', '159'], ['99', '122', '156', '158'], ['99', '158'], ['99', '158'], ['99', '158'],
         ['99', '158'], ['99', '158'], ['99', '158'], ['99', '158'], ['238'], ['99', '158'], ['661']],
        [['158', '99']]
    ],
    'drugs_hist': [
        [['V06D', 'C05A', 'C02D'], ['N05B', 'C02D'], ['C07A', 'A06A'], ['N02A', 'B01A'], ['C07A', 'B01A'],
         ['C07A', 'C09X']],
        [['N02A', 'B01A']]
    ]
}

# 创建患者图并处理
patient_graph_embeddings = create_patient_graph_data(batch_data)
