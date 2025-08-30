"""这个model的构图方法是每个visit构建一个图，然后把所有patient所有visit聚合为一个batch，一个batch_data聚合为一个batch_graph"""
import torch.nn as nn
import torch.nn
from models.grpah_netV1 import HeteroGNN, process_visit_graphs, process_patient_graphs


def extract_and_transpose(tensor_list):
    """
    提取每个张量的最后一个序列并转置
    :param tensor_list: list of tensors
    :return: 处理后的张量列表
    """
    processed_tensors = []
    for tensor in tensor_list:
        last_seq = tensor[:, -1:, :]  # 提取最后一个序列
        transposed_seq = last_seq.transpose(0, 1)  # 转置
        processed_tensors.append(transposed_seq)
    return processed_tensors


def split_tensor(tensor, lengths, max_len):
    """
    将聚合的张量拆分为原始形状
    :param tensor: 聚合的张量
    :param lengths: 每个batch的长度
    :param max_len: 最大长度
    :return: 拆分后的张量列表
    """
    index = 0
    outputs = []

    for length in lengths:
        output_tensor = tensor[index:index + length]
        outputs.append(output_tensor)
        index += length

    outputs = [x[:, :max_len, :] for x in outputs]
    return outputs


def aggregate_tensors(tensors, device):
    """
    将多个张量聚合到最大长度
    :param tensors: list of tensors
    :return: 聚合后的张量, 每个batch的长度
    """
    max_len = max([x.size(1) for x in tensors])
    padded_inputs = []
    lengths = []

    for x in tensors:
        lengths.append(x.size(0))
        padding = torch.zeros(x.size(0), max_len - x.size(1), x.size(2)).to(device)
        padded_x = torch.cat((x, padding), dim=1)
        padded_inputs.append(padded_x)

    aggregated_tensor = torch.cat(padded_inputs, dim=0)
    return aggregated_tensor, lengths


def concatenate_patient_embeddings(patient_emb_dict):
    """
    将 patient_emb_dict 中的所有张量在最后一个维度上拼接起来。

    参数:
    patient_emb_dict (dict of torch.Tensor): 包含每个特征的患者嵌入字典

    返回:
    torch.Tensor: 拼接后的张量
    """
    # 提取所有张量
    emb_list = list(patient_emb_dict.values())
    # 在最后一个维度上拼接
    patient_emb = torch.cat(emb_list, dim=-1)
    return patient_emb


def pad_tensors(tensors):
    """
    将不同尺寸的tensor填充成相同尺寸的大tensor。

    参数:
    tensors (list of torch.Tensor): 输入的多个张量列表，每个张量的形状为 (visit, monitor, dim)

    返回:
    torch.Tensor: 填充后的大张量，形状为 (batch, max_visits, max_monitors, dim)
    """
    # 找到各个维度的最大值
    max_visits = max(t.size(0) for t in tensors)
    max_monitors = max(t.size(1) for t in tensors)
    dim = tensors[0].size(2)

    # 初始化一个大的tensor，用0填充
    batch_size = len(tensors)
    padded_tensor = torch.zeros((batch_size, max_visits, max_monitors, dim))

    # 将各个tensor填充到大tensor中
    for i, tensor in enumerate(tensors):
        v, m, d = tensor.size()
        padded_tensor[i, :v, :m, :] = tensor

    return padded_tensor


def expand_tensors(visit_emb_list, target_shape):
    """
    将visit_emb_list中的每个张量扩展到目标形状。

    参数:
    visit_emb_list (list of torch.Tensor): 输入的张量列表，每个张量的形状为 (16, 22, 128)
    target_shape (tuple): 目标张量形状 (16, 22, 77, 128)

    返回:
    list of torch.Tensor: 扩展后的张量列表，每个张量的形状为 (16, 22, 77, 128)
    """
    expanded_list = []
    for tensor in visit_emb_list:
        # 在第三个维度上增加一个维度 (16, 22, 1, 128)
        tensor_expanded = tensor.unsqueeze(2)
        # 将第三个维度扩展到目标形状 (16, 22, 77, 128)
        tensor_expanded = tensor_expanded.expand(-1, -1, target_shape[2], -1)
        expanded_list.append(tensor_expanded)
    return expanded_list


class CrossMed4(nn.Module):
    def __init__(
            self,
            Tokenizers_visit_event,
            Tokenizers_monitor_event,
            output_size,
            device,
            embedding_dim=128,
            dropout=0.7
    ):
        super(CrossMed4, self).__init__()
        self.embedding_dim = embedding_dim
        self.visit_event_token = Tokenizers_visit_event
        self.monitor_event_token = Tokenizers_monitor_event

        self.feature_visit_event_keys = Tokenizers_visit_event.keys()
        self.feature_monitor_event_keys = Tokenizers_monitor_event.keys()
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

        for feature_key in self.feature_monitor_event_keys:
            tokenizer = self.monitor_event_token[feature_key]
            self.embeddings[feature_key] = nn.Embedding(
                tokenizer.get_vocabulary_size(),
                self.embedding_dim,
                padding_idx=tokenizer.get_padding_index(),
            )

        self.visit_gru = nn.ModuleDict()
        # 为每种visit_event添加一种gru
        for feature_key in self.feature_visit_event_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in self.feature_monitor_event_keys:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in ['weight', 'age']:
            self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.monitor_gru = nn.ModuleDict()
        # 为每种monitor_event添加一种gru
        for feature_key in self.feature_monitor_event_keys:
            self.monitor_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        for feature_key in self.feature_visit_event_keys:
            self.monitor_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.patient_info_fc = nn.ModuleDict()
        # 为每种病人信息添加一个全连接层
        for feature_key in ['weight', 'age']:
            self.patient_info_fc[feature_key] = nn.Linear(1, self.embedding_dim)

        item_num = int(len(Tokenizers_monitor_event.keys()) / 2) + 3 + 2
        self.fc_visit = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(5 * self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
            nn.Linear(self.embedding_dim, 1),
        )

        self.fc_patient = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(item_num * self.embedding_dim, output_size)
        )

        self.gnn = nn.ModuleDict()
        num_nodes_dict = {}
        for feature_key in self.feature_visit_event_keys:
            num_nodes_dict[feature_key] = self.visit_event_token[feature_key].get_vocabulary_size()

        # 给visit-monitoring交互的图构图
        num_nodes_dict['lab_item'] = self.monitor_event_token['lab_item'].get_vocabulary_size()
        num_nodes_dict['inj_item'] = self.monitor_event_token['inj_item'].get_vocabulary_size()
        names = list(self.feature_visit_event_keys) + ['lab_item', 'inj_item']
        self.gnn['monitoring_visit_graph'] \
            = HeteroGNN(['lab_item', 'inj_item'], names, num_nodes_dict, embedding_dim, device)

        # 给两个patient构图
        for feature_key in ['weight', 'age']:
            num_nodes_dict[feature_key] = 61
            names = list(self.feature_visit_event_keys) + [feature_key]
            self.gnn[feature_key] = HeteroGNN([feature_key], names, num_nodes_dict, embedding_dim, device)

        self.patient_state_gru = torch.nn.GRU(1, 1, batch_first=True)

    def forward(self, batch_data):

        batch_size = len(batch_data['visit_id'])
        # patient_emb_list = []
        patient_emb_dict_origin = {}
        patient_emb_dict = {}
        patient_emb_dict_reg = {}

        """处理cond, proc, drug"""
        for feature_key in self.feature_visit_event_keys:
            x = self.visit_event_token[feature_key].batch_encode_3d(
                batch_data[feature_key], max_length=(400, 1024)
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event)

            x = self.dropout(self.embeddings[feature_key](x))
            # (patient, visit, event, embedding_dim)

            x = torch.sum(x, dim=2)
            # (patient, visit, embedding_dim)

            patient_emb_dict_origin[feature_key] = x
            # dict{feature_key: (patient, visit, embedding_dim)}

        """处理lab, inj"""
        # lab_key = 'lab_item'
        # inj_key = 'inj_item'
        # graph_key = 'monitoring_visit_graph'

        # 初始化多就诊字典 最后里面装的是dict{5 *list[patient * (visit, monitor, embedding_dim)]}
        patient_emb_list_dict = {'lab_item': [], 'inj_item': []}
        for feature_key in self.feature_visit_event_keys:
            patient_emb_list_dict[feature_key] = []

        visit_lengths = []
        max_monitors = {'lab_item': None, 'inj_item': None}
        # 迭代处理每一对
        feature_paris = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))

        for feature_key1, feature_key2 in feature_paris:
            monitor_emb_list = []
            # 先聚合monitor层面，生成batch_size个病人的多次就诊的表征，batch_size * (1, visit, monitor, embedding)
            for patient in range(batch_size):
                x1 = self.monitor_event_token[feature_key1].batch_encode_3d(
                    batch_data[feature_key1][patient], max_length=(400, 1024)
                )
                x1 = torch.tensor(x1, dtype=torch.long, device=self.device)
                x2 = self.monitor_event_token[feature_key2].batch_encode_3d(
                    batch_data[feature_key2][patient], max_length=(400, 1024)
                )
                x2 = torch.tensor(x2, dtype=torch.long, device=self.device)
                # (visit, monitor, event)

                x1 = self.embeddings[feature_key1](x1)
                x2 = self.embeddings[feature_key2](x2)
                # (visit, monitor, event, embedding_dim)

                x = self.dropout(torch.mul(x1, x2))
                # (visit, monitor, event, embedding_dim)

                x = torch.sum(x, dim=2)
                # (visit, monitor, embedding_dim)

                monitor_emb_list.append(x)

            """把monitor的数据变成指定样式"""
            aggregated_monitor_tensor, visit_lengths = aggregate_tensors(monitor_emb_list, self.device)
            # (patient * visit, monitor, embedding_dim) 这里不是乘法，而是将多个visit累加

            _, max_monitor_temp, _ = aggregated_monitor_tensor.size()
            max_monitors[feature_key1] = max_monitor_temp

            """把visit的数据变成指定样式"""
            patient_emb_list_dict[feature_key1] = aggregated_monitor_tensor

        # 处理其他feature key的数据并加入到patient_emb_list_dict中
        for feature_key in self.feature_visit_event_keys:
            x = patient_emb_dict_origin[feature_key]
            # (patient, visit, embedding_dim)
            padded_inputs = []
            for i, length in enumerate(visit_lengths):
                patient_visit_tensor = x[i, :length, :]  # 获取每个patient的所有visit
                # (visit, embedding_dim)
                repeated_tensor = patient_visit_tensor.unsqueeze(1).repeat(1, max(max_monitors.values()), 1)
                # (visit, monitor, embedding_dim)
                padded_inputs.append(repeated_tensor)

            aggregated_feature_tensor = torch.cat(padded_inputs, dim=0).to(self.device)
            # (patient * visit, monitor, embedding_dim) 这里不是乘法，而是将多个visit累加
            patient_emb_list_dict[feature_key] = aggregated_feature_tensor

        graphs = batch_data['monitoring_visit_graph']

        # pad_graph = create_common_pad_graph(self.gnn[feature_key1].edge_types)

        batch_graph = process_visit_graphs(graphs, patient_emb_list_dict, self.device)

        # 进行图神经网络传播
        patient_emb_list_dict_out = self.gnn['monitoring_visit_graph'](batch_graph, max(max_monitors.values()))

        # 初始化结果字典
        result_dict = {'lab_item': [], 'inj_item': []}
        for feature_key in self.feature_visit_event_keys:
            result_dict[feature_key] = []

        # 将 patient_emb_list_dict_out 的向量按每个 visit 的维度拆分，并直接拼接
        for feature_key in ['lab_item', 'inj_item']:
            split_values = torch.split(patient_emb_list_dict_out[feature_key], max_monitors[feature_key], dim=0)
            stacked_values = [split_value.unsqueeze(0) for split_value in split_values]
            result_dict[feature_key] = torch.cat(stacked_values, dim=0)

        for feature_key in self.feature_visit_event_keys:
            split_values = torch.split(patient_emb_list_dict_out[feature_key], max(max_monitors.values()), dim=0)
            stacked_values = [split_value.unsqueeze(0) for split_value in split_values]
            result_dict[feature_key] = torch.cat(stacked_values, dim=0)

        for key in result_dict.keys():
            x = result_dict[key]
            output, hidden = self.monitor_gru[key](x)
            # output: (patient * visit, monitor, embedding_dim), hidden:(1, patient * visit, embedding_dim)

            """这里搞regression的representation"""
            if key not in patient_emb_dict_reg:
                patient_emb_dict_reg[key] = hidden.squeeze(dim=0)  # 如果 `key` 不存在，创建新的条目
            else:
                patient_emb_dict_reg[key] = patient_emb_dict_reg[key] + hidden.squeeze(dim=0)  # 如果 `key` 存在，进行加法操作
            """这里处理完了regression的第一部分"""

            # 拆分gru的输出
            if key == 'lab_item' or key == 'inj_item':
                max_monitors_temp = max_monitors[key]
            else:
                max_monitors_temp = max(max_monitors.values())
            split_outputs = split_tensor(output, visit_lengths, max_monitors_temp)

            # 提取最后一个序列并转置
            visit_emb_list = extract_and_transpose(split_outputs)
            # list[batch * (1,visit,dim)]

            # 开始搞visit层面的
            aggregated_visit_tensor, lengths = aggregate_tensors(visit_emb_list, self.device)

            output, hidden = self.visit_gru[key](aggregated_visit_tensor)
            # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)

            if key not in patient_emb_dict:
                patient_emb_dict[key] = hidden.squeeze(dim=0)  # 如果 `key` 不存在，创建新的条目
            else:
                patient_emb_dict[key] = patient_emb_dict[key] + hidden.squeeze(dim=0)  # 如果 `key` 存在，进行加法操作
                # (patient, embedding_dim)

        """这里搞regression的第二部分，输出reg结果"""
        patient_emb_reg = concatenate_patient_embeddings(patient_emb_dict_reg)
        logits_reg1 = self.fc_visit(patient_emb_reg).view(-1)
        # (patient * visit,) 这里不是乘法，而是将多个visit累加

        """这里搞patient state的直接传递"""
        # 确定第二层的最大长度
        max_length = max(len(item) for sublist in batch_data['patient_state'] for item in sublist) - 1

        # 填充每个第三层列表至最大长度
        padded_data = []
        for sublist in batch_data['patient_state']:
            for item in sublist:
                # 删除每个列表的最后一个元素
                truncated_item = item[:-1]
                # 用 0 填充至最大长度
                padded_item = truncated_item + [0] * (max_length - len(truncated_item))
                padded_data.append(padded_item)

        # 将数据转换为Tensor
        input_tensor = torch.tensor(padded_data, dtype=torch.float32).unsqueeze(-1).to(self.device)

        _, logits_reg2 = self.patient_state_gru(input_tensor)

        logits_reg2 = logits_reg2.view(-1)

        logits_reg = (logits_reg1 + logits_reg2) / 2
        """到这里处理完reg"""

        """处理weight, age"""
        for feature_key1 in ['weight', 'age']:
            x = batch_data[feature_key1]

            # 找出最长列表的长度
            max_length = max(len(sublist) for sublist in x)
            # 将每个子列表的元素转换为浮点数，并使用0对齐长度
            x = [[float(item) for item in sublist] + [0] * (max_length - len(sublist)) for sublist in x]
            # (patient, visit)

            x = torch.tensor(x, dtype=torch.float, device=self.device)
            # (patient, visit)

            num_patients, num_visits = x.shape
            x = x.view(-1, 1)  # 变成 (patient * visit, 1)

            # 创建一个掩码用于标记输入为0的位置
            mask = (x == 0)

            x = self.dropout(self.patient_info_fc[feature_key1](x))

            # 对输入为0的位置输出也设为0
            x = x * (~mask)
            # (patient * visit, embedding_dim)

            x = x.view(num_patients, num_visits, -1)
            # (patient, visit, embedding_dim)

            """开始构图"""
            _, max_visit, _ = x.size()
            patient_emb_list_dict = {feature_key1: x}

            # 处理其他feature key的数据并加入到patient_emb_list_dict中
            for feature_key in self.feature_visit_event_keys:
                x = patient_emb_dict_origin[feature_key]
                patient_emb_list_dict[feature_key] = x

            if feature_key1 in 'age_patient_graph':
                feature_graph = 'age_patient_graph'
            else:
                feature_graph = 'weight_patient_graph'

            graphs = batch_data[feature_graph]

            batch_graph = process_patient_graphs(graphs, patient_emb_list_dict, self.device)
            # batch_graph = Batch.from_data_list(graphs)

            # 进行图神经网络传播
            patient_emb_list_dict_out = self.gnn[feature_key1](batch_graph, max_visit)

            # 初始化结果字典
            result_dict = {feature_key1: []}
            for feature_key in self.feature_visit_event_keys:
                result_dict[feature_key] = []

            # 将 patient_emb_list_dict_out 的向量按每个 visit 的维度拆分，并直接拼接
            for key, value in patient_emb_list_dict_out.items():
                split_values = torch.split(value, max_visit, dim=0)
                stacked_values = [split_value.unsqueeze(0) for split_value in split_values]
                result_dict[key] = torch.cat(stacked_values, dim=0)

            # 给每个key做gru
            for key in result_dict.keys():
                x = result_dict[key]
                output, hidden = self.visit_gru[key](x)
                # output: (patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)

                if key not in patient_emb_dict:
                    patient_emb_dict[key] = hidden.squeeze(dim=0)  # 如果 `key` 不存在，创建新的条目
                else:
                    patient_emb_dict[key] = patient_emb_dict[key] + hidden.squeeze(dim=0)  # 如果 `key` 存在，进行加法操作
                    # (patient, embedding_dim)

        patient_emb = concatenate_patient_embeddings(patient_emb_dict)
        # (patient, 6 * embedding_dim)

        logits = self.fc_patient(patient_emb)
        # (patient, label_size)
        return logits_reg, logits






