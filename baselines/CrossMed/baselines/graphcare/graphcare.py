"""这个model的构图方法是每个visit构建一个图，然后把所有patient所有visit聚合为一个batch，一个batch_data聚合为一个batch_graph"""
import torch.nn as nn
import torch.nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GraphCare(nn.Module):
    def __init__(
            self,
            Tokenizers_visit_event,
            output_size,
            device,
            embedding_dim=128,
            dropout=0.7
    ):
        super(GraphCare, self).__init__()
        self.embedding_dim = embedding_dim
        self.visit_event_token = Tokenizers_visit_event

        self.feature_visit_event_keys = Tokenizers_visit_event.keys()
        self.dropout = torch.nn.Dropout(p=dropout)

        self.device = device

        self.embeddings = nn.ModuleDict()
        # 为每种event（包含monitor和visit）添加一种嵌入
        for feature_key in self.feature_visit_event_keys:
            tokenizer = self.visit_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        self.visit_gru = nn.ModuleDict()
        # 为每种visit_event添加一种gru
        for feature_key in self.feature_visit_event_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        # 新增: 定义GCN层
        self.gcn1 = GCNConv(self.embedding_dim, self.embedding_dim).to(self.device)
        self.gcn2 = GCNConv(self.embedding_dim, self.embedding_dim).to(self.device)
        self.fc_patient = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(3 * self.embedding_dim, output_size)
        )

        self.gru_layers = nn.ModuleDict()
        # 为每种feature添加一种gru
        for feature_key in self.feature_visit_event_keys:
            self.add_gru_layer(feature_key)

    def add_gru_layer(self, feature_key: str):
        self.gru_layers[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

    def forward(self, batch_data):
        batch_size = len(batch_data['visit_id'])
        patient_event_dict = {}
        patient_emb_dict_origin = {}

        """step0：得到cond, proc, drug的表示"""
        for feature_key in self.feature_visit_event_keys:
            x = self.visit_event_token[feature_key].batch_encode_3d(
                batch_data[feature_key], max_length=(400, 1024)
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event)

            x1 = self.dropout(self.embeddings[feature_key](x))
            # (patient, visit, event, embedding_dim)

            patient_event_dict[feature_key] = x
            # dict{feature_key: (patient, visit, event)}

            patient_emb_dict_origin[feature_key] = x1
            # dict{feature_key: (patient, visit, event, embedding_dim)}

        """step1：构图（基本没法还原原文）"""
        # 构图和构边
        edge_index_list = []  # 存储所有边的列表
        all_nodes_list = []  # 存储所有节点嵌入

        event_to_node_map = {}  # 记录每个事件的节点索引，用于跨就诊构边
        node_index = 0  # 全局节点索引计数器

        # 遍历每个患者的所有就诊事件
        # TODO: 这里只能放一个病人，batch=1
        for patient_idx in range(patient_emb_dict_origin['procedures'].shape[0]):  # 遍历每个病人
            for visit_idx in range(patient_emb_dict_origin['procedures'].shape[1]):  # 遍历每次就诊
                visit_event_indices = []  # 存储当前就诊的事件节点索引

                # 遍历当前患者的每种事件类型（cond, proc, drug）
                for feature_key in self.feature_visit_event_keys:
                    for event_idx in range(patient_emb_dict_origin[feature_key].shape[2]):
                        # 获取事件名称
                        event_name = int(patient_event_dict[feature_key][patient_idx, visit_idx, event_idx])
                        if event_name == 0:
                            continue
                        # 获取事件嵌入
                        event_embedding = patient_emb_dict_origin[feature_key][patient_idx, visit_idx, event_idx]

                        all_nodes_list.append(event_embedding)

                        # 记录当前事件节点的索引
                        current_node_index = node_index
                        visit_event_indices.append(current_node_index)
                        node_index += 1

                        # 如果该事件在之前的其他就诊中出现过，构建跨就诊边
                        event_key = (feature_key, event_name)  # 用事件类型和事件索引唯一标识一个事件

                        if event_key in event_to_node_map:
                            for previous_node_index in event_to_node_map[event_key]:
                                edge_index_list.append([previous_node_index, current_node_index])  # 跨就诊边
                                edge_index_list.append([current_node_index, previous_node_index])  # 双向边

                        # 更新事件到节点的映射
                        if event_key not in event_to_node_map:
                            event_to_node_map[event_key] = []
                        event_to_node_map[event_key].append(current_node_index)

                # 构建当前就诊的全连接边
                for i in range(len(visit_event_indices)):
                    for j in range(i + 1, len(visit_event_indices)):
                        edge_index_list.append([visit_event_indices[i], visit_event_indices[j]])  # 就诊内全连接
                        edge_index_list.append([visit_event_indices[j], visit_event_indices[i]])  # 双向边

        # 将节点和边转为 tensor
        all_nodes_tensor = torch.stack(all_nodes_list)  # (total_events, embedding_dim)
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()  # (2, num_edges)
        graph_data = Data(x=all_nodes_tensor, edge_index=edge_index_tensor).to(self.device)

        """step2：图更新"""
        updated_node_embeddings = self.gcn1(graph_data.x, graph_data.edge_index)
        updated_node_embeddings = torch.relu(updated_node_embeddings)  # 加ReLU激活
        updated_node_embeddings = self.gcn2(updated_node_embeddings, graph_data.edge_index)

        # 新增: 整理更新后的节点回到 patient_emb_dict_origin 的结构中
        patient_emb_dict_update = {key: torch.zeros_like(patient_emb_dict_origin[key]) for key in
                                           patient_emb_dict_origin}

        node_idx = 0  # 重置索引计数器
        for patient_idx in range(patient_emb_dict_origin['procedures'].shape[0]):  # 遍历每个病人
            for visit_idx in range(patient_emb_dict_origin['procedures'].shape[1]):  # 遍历每次就诊
                for feature_key in self.feature_visit_event_keys:
                    for event_idx in range(patient_emb_dict_origin[feature_key].shape[2]):
                        event_name = int(patient_event_dict[feature_key][patient_idx, visit_idx, event_idx])
                        if event_name == 0:
                            continue

                        # 将更新后的嵌入放回对应位置
                        patient_emb_dict_update[feature_key][patient_idx, visit_idx, event_idx] = \
                        updated_node_embeddings[node_idx]
                        node_idx += 1

        patient_emb = []
        """step3：时序融合"""
        for feature_key in self.feature_visit_event_keys:
            x = patient_emb_dict_update[feature_key].squeeze(0)
            # x = (visit, event, embedding_dim)
            x = x.sum(dim=1)
            # x = (visit, embedding_dim)
            output, hidden = self.gru_layers[feature_key](x)
            patient_emb.append(hidden)

        patient_emb = torch.cat(patient_emb, dim=-1)
        logits = self.fc_patient(patient_emb)
        # (patient, label_size)
        return logits
