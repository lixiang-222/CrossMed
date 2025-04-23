import torch.nn
from pyhealth.medcode import InnerMap
from torch.nn.utils.rnn import unpack_sequence

from baselines.Seqmodels import *


class Transformer(nn.Module):
    def __init__(
            self,
            Tokenizers,
            output_size,
            device,
            embedding_dim=128,
            dropout=0.5
    ):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.device = device
        # 为每种feature添加一种嵌入
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        # 为每种feature添加一种TransformerLayer
        self.transformer = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.transformer[feature_key] = TransformerLayer(heads=2,
                                                             feature_size=embedding_dim, dropout=dropout, num_layers=2
                                                             )

        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

    def forward(self, batchdata):
        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
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
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        return logits

class GRU(nn.Module):
    def __init__(
            self,
            Tokenizers,
            output_size,
            device,
            embedding_dim=128,
            dropout=0.7
    ):
        super(GRU, self).__init__()
        self.embedding_dim = embedding_dim
        self.event_token = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.gru_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.rnn_dropout = torch.nn.Dropout(p=dropout)

        self.device = device

        # 为每种feature添加一种嵌入
        for feature_key in self.feature_keys:
            self.add_embedding_layer(feature_key)

        # 为每种feature添加一种gru
        for feature_key in self.feature_keys:
            self.add_gru_layer(feature_key)

        self.fc = nn.Linear(2 * 3 * self.embedding_dim, output_size)

    def add_embedding_layer(self, feature_key: str):
        tokenizer = self.event_token[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

    def add_gru_layer(self, feature_key: str):
        self.gru_layers[feature_key] = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

    def forward(self, batchdata):
        patient_num = len(batchdata['conditions'])
        patient_emb = []
        for patient in range(patient_num):
            all_emb_seq = {
                'conditions': [],
                'procedures': [],
                'drugs_hist': [],
            }
            # 生成三个主要event的嵌入
            for feature_key in self.feature_keys:
                x = self.event_token[feature_key].batch_encode_2d(batchdata[feature_key][patient])
                x = torch.tensor(x, dtype=torch.long, device=self.device)
                # (visit,event)
                x = self.rnn_dropout(self.embeddings[feature_key](x))
                # (visit, event, embedding_dim)
                x = torch.sum(x, dim=1).unsqueeze(dim=0)
                # (patient, visit, embedding_dim)
                all_emb_seq[feature_key].append(x)

            output1, hidden1 = self.gru_layers['conditions'](torch.cat(all_emb_seq['conditions'], dim=1))
            output2, hidden2 = self.gru_layers['procedures'](torch.cat(all_emb_seq['procedures'], dim=1))
            output3, hidden3 = self.gru_layers['drugs_hist'](torch.cat(all_emb_seq['drugs_hist'], dim=1))
            output1 = output1[:, -1, :].unsqueeze(0)
            output2 = output2[:, -1, :].unsqueeze(0)
            output3 = output3[:, -1, :].unsqueeze(0)

            seq_repr = torch.cat([output1, output2, output3], dim=-1)
            last_repr = torch.cat([hidden1, hidden2, hidden3], dim=-1)
            patient_repr = torch.cat([seq_repr, last_repr], dim=-1)

            patient_emb.append(patient_repr)

        patient_emb = torch.cat(patient_emb, dim=0)
        # (patient, label_size)
        logits = self.fc(patient_emb).squeeze(dim=1)
        # obtain y_true, loss, y_prob
        return logits



class MLP(nn.Module):
    def __init__(
            self,
            Tokenizers_visit_event,
            Tokenizers_monitor_event,
            output_size,
            device,
            embedding_dim=128,
            dropout=0.7
    ):
        super(MLP, self).__init__()
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

        # 移除 GRU，改用 MLP 或直接聚合
        self.visit_mlp = nn.ModuleDict()
        for feature_key in self.feature_visit_evnet_keys:
            self.visit_mlp[feature_key] = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        for feature_key in self.feature_monitor_event_keys:
            self.visit_mlp[feature_key] = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        for feature_key in ['weight', 'age']:
            self.visit_mlp[feature_key] = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.monitor_mlp = nn.ModuleDict()
        for feature_key in self.feature_monitor_event_keys:
            self.monitor_mlp[feature_key] = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.fc_age = nn.Linear(1, self.embedding_dim)
        self.fc_weight = nn.Linear(1, self.embedding_dim)

        item_num = int(len(Tokenizers_monitor_event.keys()) / 2) + 3 + 2
        self.fc_patient = nn.Sequential(
            torch.nn.ReLU(),
            nn.Linear(item_num * self.embedding_dim, output_size)
        )

    def forward(self, batch_data):
        batch_size = len(batch_data['visit_id'])
        patient_emb_list = []

        """处理 lab, inj（monitor_event）"""
        feature_paris = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))
        for feature_key1, feature_key2 in feature_paris:
            monitor_emb_list = []
            for patient in range(batch_size):
                x1 = self.monitor_event_token[feature_key1].batch_encode_3d(
                    batch_data[feature_key1][patient], max_length=(400, 1024)
                )
                x1 = torch.tensor(x1, dtype=torch.long, device=self.device)
                x2 = self.monitor_event_token[feature_key2].batch_encode_3d(
                    batch_data[feature_key2][patient], max_length=(400, 1024)
                )
                x2 = torch.tensor(x2, dtype=torch.long, device=self.device)

                x1 = self.dropout(self.embeddings[feature_key1](x1))
                x2 = self.dropout(self.embeddings[feature_key2](x2))

                x = torch.mul(x1, x2)  # (visit, monitor, event, embedding_dim)
                x = torch.sum(x, dim=2)  # (visit, monitor, embedding_dim)

                # 直接对 monitor 维度求和（或平均）
                x = torch.sum(x, dim=1)  # (visit, embedding_dim)
                x = self.monitor_mlp[feature_key1](x)  # (visit, embedding_dim)

                monitor_emb_list.append(x.unsqueeze(0))  # (1, visit, embedding_dim)

            # 聚合所有病人的 visit 数据
            aggregated_visit_tensor = torch.cat(monitor_emb_list, dim=0)  # (batch, visit, embedding_dim)
            
            # 直接对 visit 维度求和（或平均）
            patient_emb = torch.sum(aggregated_visit_tensor, dim=1)  # (batch, embedding_dim)
            patient_emb_list.append(patient_emb)

        """处理 cond, proc, drug（visit_event）"""
        for feature_key in self.feature_visit_evnet_keys:
            x = self.visit_event_token[feature_key].batch_encode_3d(batch_data[feature_key])
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.dropout(self.embeddings[feature_key](x))  # (patient, visit, event, embedding_dim)
            x = torch.sum(x, dim=2)  # (patient, visit, embedding_dim)

            # 直接对 visit 维度求和（或平均）
            patient_emb = torch.sum(x, dim=1)  # (patient, embedding_dim)
            patient_emb = self.visit_mlp[feature_key](patient_emb)  # (patient, embedding_dim)
            patient_emb_list.append(patient_emb)

        """处理 weight, age"""
        for feature_key in ['weight', 'age']:
            x = batch_data[feature_key]
            max_length = max(len(sublist) for sublist in x)
            x = [[float(item) for item in sublist] + [0] * (max_length - len(sublist)) for sublist in x]
            x = torch.tensor(x, dtype=torch.float, device=self.device)  # (patient, visit)

            num_patients, num_visits = x.shape
            x = x.view(-1, 1)  # (patient * visit, 1)
            mask = (x == 0)

            if feature_key == 'weight':
                x = self.dropout(self.fc_weight(x))
            elif feature_key == 'age':
                x = self.dropout(self.fc_age(x))
            x = x * (~mask)  # (patient * visit, embedding_dim)
            x = x.view(num_patients, num_visits, -1)  # (patient, visit, embedding_dim)

            # 直接对 visit 维度求和（或平均）
            patient_emb = torch.sum(x, dim=1)  # (patient, embedding_dim)
            patient_emb = self.visit_mlp[feature_key](patient_emb)  # (patient, embedding_dim)
            patient_emb_list.append(patient_emb)

        """拼接所有特征"""
        patient_emb = torch.cat(patient_emb_list, dim=-1)  # (patient, item_num * embedding_dim)
        logits = self.fc_patient(patient_emb)  # (patient, output_size)
        return logits




class RETAIN(nn.Module):
    def __init__(self, Tokenizers, output_size, device,
                 embedding_dim: int = 128, dropout=0.5
                 ):
        super(RETAIN, self).__init__()
        self.embedding_dim = embedding_dim
        Tokenizers = {k: Tokenizers[k] for k in list(Tokenizers)[1:]}
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()

        # add feature RETAIN layers
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
        self.retain = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.retain[feature_key] = RETAINLayer(feature_size=embedding_dim, dropout=dropout)

        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)
        self.device = device

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

    def forward(self, batchdata):

        patient_emb = []
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            x = self.embeddings[feature_key](x)
            x = torch.sum(x, dim=2)
            mask = torch.sum(x, dim=2) != 0
            x = self.retain[feature_key](x, mask)
            patient_emb.append(x)
        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits


class StageNet(nn.Module):
    def __init__(self, Tokenizers, output_size, device, embedding_dim: int = 128,
                 chunk_size: int = 128,
                 levels: int = 3,
                 ):
        super(StageNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.levels = levels
        Tokenizers = {k: Tokenizers[k] for k in list(Tokenizers)[1:]}
        self.feature_keys = Tokenizers.keys()

        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()

        self.stagenet = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)
            self.stagenet[feature_key] = StageNetLayer(
                input_dim=embedding_dim,
                chunk_size=self.chunk_size,
                levels=self.levels,
            )
        self.fc = nn.Linear(
            len(self.feature_keys) * self.chunk_size * self.levels, output_size
        )
        self.device = device

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )

    def forward(self, batchdata):
        patient_emb = []
        distance = []
        mask_dict = {}
        for feature_key in self.feature_keys:
            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )

            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, event, embedding_dim)
            x = self.embeddings[feature_key](x)
            # (patient, visit, embedding_dim)
            x = torch.sum(x, dim=2)
            # (patient, visit)
            mask = torch.any(x != 0, dim=2)
            mask_dict[feature_key] = mask
            time = None
            x, _, cur_dis = self.stagenet[feature_key](x, time=time, mask=mask)
            patient_emb.append(x)
            distance.append(cur_dis)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits


class KAME(nn.Module):
    def __init__(self, Tokenizers, output_size, device,
                 embedding_dim: int = 128, dataset='mimic3'
                 ):
        super(KAME, self).__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.feat_tokenizers = Tokenizers
        self.embeddings = nn.ModuleDict()
        self.linear_layers = nn.ModuleDict()
        self.feature_keys = Tokenizers.keys()
        self.parent_dictionary = {'cond_hist': InnerMap.load("ICD9CM"), 'procedures': InnerMap.load("ICD9PROC")}
        self.compatability = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1, bias=False),
        )

        self.knowledge_map = nn.ModuleDict()
        for feature_key in ['cond_hist', 'procedures']:
            self.knowledge_map[feature_key] = nn.Linear(embedding_dim, embedding_dim, bias=False)

        for feature_key in self.feature_keys:
            self.add_feature_transform_layer(feature_key)

        self.rnn = nn.ModuleDict()
        for feature_key in self.feature_keys:
            if feature_key.endswith('_parent'):
                continue
            self.rnn[feature_key] = nn.GRU(input_size=self.embedding_dim,
                                           hidden_size=self.embedding_dim,
                                           batch_first=True, bidirectional=False)

        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    def embed_code_with_parent(self, x, feature_key):
        # x: (patient, visit, event)
        max_visit = x.shape[1]
        out = []
        out_mask = []
        for patient in x:
            mask = []
            patient_embed = []
            for visit in patient:
                if visit.sum() == 0:
                    num_pad = max_visit - len(patient_embed)
                    mask.extend([0] * num_pad)
                    visit_embed = torch.zeros(self.embedding_dim, device=self.device)
                    patient_embed.extend([visit_embed] * num_pad)
                    break
                visit = visit[visit != 0]
                mask.append(1)
                events = self.feat_tokenizers[feature_key].convert_indices_to_tokens(visit.tolist())
                basic_embeds = self.embeddings[feature_key](visit)
                visit_embed = torch.zeros(self.embedding_dim, device=self.device)
                for embed, event in zip(basic_embeds, events):
                    try:
                        parents = self.parent_dictionary[feature_key].get_ancestors(event)
                    except:
                        visit_embed += embed
                        continue
                    parents = self.feat_tokenizers[feature_key + '_parent'].convert_tokens_to_indices(parents)
                    parents = torch.tensor(parents, dtype=torch.long, device=self.device)
                    parents_embed = self.embeddings[feature_key + '_parent'](parents)
                    parents_embed = torch.cat([parents_embed, embed.reshape(1, -1)], dim=0)
                    embed_ = torch.stack([embed] * len(parents_embed))
                    compat_score = self.compatability(torch.cat([embed_, parents_embed], dim=1))
                    compat_score = torch.softmax(compat_score, dim=0)
                    embed = torch.sum(compat_score * parents_embed, dim=0)
                    visit_embed += embed
                patient_embed.append(visit_embed)
            patient_embed = torch.stack(patient_embed)
            out.append(patient_embed)
            out_mask.append(mask)
        out = torch.stack(out)
        out_mask = torch.tensor(out_mask, dtype=torch.int, device=self.device)
        return out, out_mask

    def embed_code(self, x, feature_key):
        # x: (patient, visit, event)
        max_visit = x.shape[1]
        out = []
        out_mask = []
        for patient in x:
            mask = []
            patient_embed = []
            for visit in patient:
                if visit.sum() == 0:
                    num_pad = max_visit - len(patient_embed)
                    mask.extend([0] * num_pad)
                    visit_embed = torch.zeros(self.embedding_dim, device=self.device)
                    patient_embed.extend([visit_embed] * num_pad)
                    break
                visit = visit[visit != 0]
                mask.append(1)
                embeds = self.embeddings[feature_key](visit)
                visit_embed = torch.sum(embeds, dim=0)
                patient_embed.append(visit_embed)
            patient_embed = torch.stack(patient_embed)
            out.append(patient_embed)
            out_mask.append(mask)
        out = torch.stack(out)
        out_mask = torch.tensor(out_mask, dtype=torch.int, device=self.device)
        return out, out_mask

    def get_parent_embeddings(self, x, feature_key):
        out = []
        for patient in x:
            if patient == []:
                out.append(torch.zeros(self.embedding_dim, device=self.device))
                continue
            parent = set()
            for code in patient:
                try:
                    parent.update(self.parent_dictionary[feature_key].get_ancestors(code))
                except:
                    continue
            parent = list(parent)
            parent = self.feat_tokenizers[feature_key + '_parent'].convert_tokens_to_indices(parent)
            parent = torch.tensor(parent, dtype=torch.long, device=self.device)
            parent = self.embeddings[feature_key + '_parent'](parent)
            out.append(parent)
        return out

    def forward(self, batchdata):
        patient_emb = []
        patient_parent = {}
        for feature_key in self.feature_keys:
            if feature_key.endswith('_parent'):
                continue
            if feature_key != 'drugs':
                if feature_key == 'cond_hist':
                    x = list(map(lambda y: y[-2] if len(y) > 1 else y[-1], batchdata[feature_key]))
                else:
                    x = list(map(lambda y: y[-1], batchdata[feature_key]))
                patient_parent[feature_key] = self.get_parent_embeddings(x, feature_key)

            x = self.feat_tokenizers[feature_key].batch_encode_3d(
                batchdata[feature_key],
            )
            # (patient, visit, event)
            x = torch.tensor(x, dtype=torch.long, device=self.device)
            # (patient, visit, embedding_dim)
            if feature_key != 'drugs':
                x, mask = self.embed_code_with_parent(x, feature_key)
            else:
                x, mask = self.embed_code(x, feature_key)

            visit_len = mask.sum(dim=1)
            visit_len[visit_len == 0] = 1
            visit_len = visit_len.cpu()
            x = pack_padded_sequence(x, visit_len, batch_first=True, enforce_sorted=False)
            x, _ = self.rnn[feature_key](x)
            x = unpack_sequence(x)
            x = list(map(lambda x: x[-1], x))
            x = torch.stack(x)
            mask = (mask.sum(dim=1).reshape(-1, 1) != 0)
            x = x * mask
            patient_emb.append(x)

        tmp_patient_emb = torch.sum(torch.stack(patient_emb), dim=0)
        for key in patient_parent.keys():
            knowledge_embed = patient_parent[key]
            mask = list(map(lambda x: 0 if (x == 0).all() else 1, knowledge_embed))
            knowledge_embed = [self.knowledge_map[key](x) for x in knowledge_embed]
            patient_knowledge_embed = []
            for patient, basic_embed, mask_ in zip(knowledge_embed, tmp_patient_emb, mask):
                if mask_ == 0:
                    patient_knowledge_embed.append(torch.zeros(self.embedding_dim, device=self.device))
                    continue
                weight = torch.matmul(patient, basic_embed)
                weight = torch.softmax(weight, dim=0).reshape(-1, 1)
                patient = torch.sum(weight * patient, dim=0)
                patient_knowledge_embed.append(patient)
            patient_knowledge_embed = torch.stack(patient_knowledge_embed)
            patient_emb.append(patient_knowledge_embed)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)
        return logits

    def add_feature_transform_layer(self, feature_key: str):
        tokenizer = self.feat_tokenizers[feature_key]
        self.embeddings[feature_key] = nn.Embedding(
            tokenizer.get_vocabulary_size(),
            self.embedding_dim,
            padding_idx=tokenizer.get_padding_index(),
        )
