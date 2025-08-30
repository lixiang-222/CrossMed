import torch.nn as nn
import torch.nn.functional as F
import torch.nn
from pyhealth.medcode import InnerMap
from torch.nn.utils.rnn import unpack_sequence
from baselines.Seqmodels import *


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


class Refine(nn.Module):
    def __init__(
            self,
            Tokenizers_visit_event,
            Tokenizers_monitor_event,
            output_size,
            device,
            embedding_dim=128,
            dropout=0.7
    ):
        super(Refine, self).__init__()
        self.embedding_dim = embedding_dim
        self.visit_event_token = Tokenizers_visit_event
        self.monitor_event_token = Tokenizers_monitor_event

        self.feature_visit_evnet_keys = Tokenizers_visit_event.keys()
        self.feature_monitor_event_keys = Tokenizers_monitor_event.keys()
        self.dropout = torch.nn.Dropout(p=dropout)

        self.device = device

        self.embeddings = nn.ModuleDict()
        # 为每种event（包含monitor和visit）添加一种嵌入
        for feature_key in self.feature_visit_evnet_keys:
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

        # 为每种visit-level的event添加一种TransformerLayer
        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_visit_evnet_keys:
            self.transformer[feature_key] = TransformerLayer(heads=2,
                                                             feature_size=embedding_dim, dropout=dropout, num_layers=2)
        for feature_key in self.feature_monitor_event_keys:
            self.transformer[feature_key] = TransformerLayer(heads=2,
                                                             feature_size=embedding_dim, dropout=dropout, num_layers=2)
        for feature_key in ['weight', 'age']:
            self.transformer[feature_key] = TransformerLayer(heads=2,
                                                             feature_size=embedding_dim, dropout=dropout, num_layers=2)

        # self.visit_gru = nn.ModuleDict()
        # # 为每种visit_event添加一种gru
        # for feature_key in self.feature_visit_evnet_keys:
        #     self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        # for feature_key in self.feature_monitor_event_keys:
        #     self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        # for feature_key in ['weight', 'age']:
        #     self.visit_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.monitor_gru = nn.ModuleDict()
        # 为每种monitor_event添加一种gru
        for feature_key in self.feature_monitor_event_keys:
            self.monitor_gru[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.fc_age = nn.Linear(1, self.embedding_dim)
        self.fc_weight = nn.Linear(1, self.embedding_dim)
        # self.fc_inj_amt = nn.Linear(1, self.embedding_dim)

        self.fc_patient = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(self.embedding_dim, output_size)
        )

    def forward(self, batch_data):

        batch_size = len(batch_data['visit_id'])
        patient_emb_list = []

        """处理lab, inj"""
        feature_paris = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))
        # 迭代处理每一对
        for feature_key1, feature_key2 in feature_paris:
            monitor_emb_list = []
            # 先聚合monitor层面，生成batch_size个病人的多次就诊的表征，batch_size * (1, visit, embedding)
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

                x1 = self.dropout(self.embeddings[feature_key1](x1))
                x2 = self.dropout(self.embeddings[feature_key2](x2))
                # (visit, monitor, event, embedding_dim)

                x = torch.mul(x1, x2)
                # (visit, monitor, event, embedding_dim)

                x = torch.sum(x, dim=2)
                # (visit, monitor, embedding_dim)

                monitor_emb_list.append(x)

            # 聚合多次的monitor
            aggregated_monitor_tensor, lengths = aggregate_tensors(monitor_emb_list, self.device)
            # (patient * visit, monitor, embedding_dim) 这里不是乘法，而是将多个visit累加

            output, hidden = self.monitor_gru[feature_key1](aggregated_monitor_tensor)
            # output: (patient * visit, monitor, embedding_dim), hidden:(1, patient * visit, embedding_dim)

            # 拆分gru的输出
            max_len = max([x.size(1) for x in monitor_emb_list])
            split_outputs = split_tensor(output, lengths, max_len)

            # 提取最后一个序列并转置
            visit_emb_list = extract_and_transpose(split_outputs)
            # list[batch * (1,visit,dim)]

            # 开始搞visit层面的
            aggregated_visit_tensor, lengths = aggregate_tensors(visit_emb_list, self.device)

            output, hidden = self.transformer[feature_key1](aggregated_visit_tensor)
            # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)

            patient_emb_list.append(hidden.unsqueeze(0))
            # (patient, event)

        """处理cond, proc, drug"""
        for feature_key in self.feature_visit_evnet_keys:
            x = self.visit_event_token[feature_key].batch_encode_3d(
                batch_data[feature_key],
            )
            # (patient, visit, event)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.any(x != 0, dim=2)
            _, x = self.transformer[feature_key](x, mask)
            patient_emb_list.append(x.unsqueeze(dim=0))

        """处理weight, age"""
        for feature_key in ['weight', 'age']:
            x = batch_data[feature_key]

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

            if feature_key == 'weight':
                x = self.dropout(self.fc_weight(x))
            elif feature_key == 'age':
                x = self.dropout(self.fc_age(x))
            # 对输入为0的位置输出也设为0
            x = x * (~mask)
            # (patient * visit, embedding_dim)

            x = x.view(num_patients, num_visits, -1)
            # (patient, visit, embedding_dim)

            output, hidden = self.transformer[feature_key](x)
            # output:(patient, visit, embedding_dim), hidden:(1, patient, embedding_dim)

            patient_emb_list.append(hidden.unsqueeze(0))

        patient_emb = torch.cat(patient_emb_list, dim=0)
        # (patient, 7 * embedding_dim)
        patient_emb = torch.sum(patient_emb, dim=0)

        logits = self.fc_patient(patient_emb)
        # (patient, label_size)
        return logits
