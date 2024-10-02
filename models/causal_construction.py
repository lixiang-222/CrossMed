import os
import pandas as pd
import statsmodels.api as sm
from dowhy import CausalModel
from tqdm import tqdm
import dill
import numpy as np
import warnings


class CausalAnalysis:
    def __init__(self, Tokenizer_visit_event, Tokenizer_monitor_event, raw_data, args):
        # 包含了多个df
        self.dfs = {}
        # 包含了多个causal_effects_matrix
        self.causal_effects_matrixs = {}

        self.visit_event_token = Tokenizer_visit_event
        self.monitor_event_token = Tokenizer_monitor_event
        self.feature_visit_event_keys = Tokenizer_visit_event.keys()
        self.feature_monitor_event_keys = Tokenizer_monitor_event.keys()

        self.preprocess_data(raw_data.samples, args)
        self.learn_causal_effects(args)

    def preprocess_data(self, raw_data, args):
        # 此函数将从 raw_data 中处理并生成 self.df
        if args.developer:
            # 开发模式，加载小数据集
            data_path = f'data/{args.dataset}/processed_data/{args.task}/data_matrix_developer.pkl'
        else:
            data_path = f'data/{args.dataset}/processed_data/{args.task}/data_matrix.pkl'

        # 检查文件是否存在
        if os.path.exists(data_path):
            # 从df_path加载数据
            self.dfs = dill.load(open(data_path, 'rb'))
            print(f"Data Matrix loaded from {data_path}")
        else:
            # 如果文件不存在，开始处理数据
            print(f"Data Matrix not found at {data_path}. Starting data processing.")

            feature_pairs = list(zip(*[iter(self.feature_monitor_event_keys)] * 2))
            for monitor_feature_key1, monitor_feature_key2 in feature_pairs:
                for visit_feature_key in self.feature_visit_event_keys:
                    # 生成列的长度
                    col_len1 = self.monitor_event_token[monitor_feature_key1].get_vocabulary_size()
                    col_len2 = self.visit_event_token[visit_feature_key].get_vocabulary_size()

                    # 两个特殊情况
                    if visit_feature_key == 'drugs_hist':
                        visit_feature_key = 'drugs'
                    elif visit_feature_key == 'cond_hist':
                        visit_feature_key = 'conditions'

                    # 生成每个列名
                    columns = ([f'{monitor_feature_key1}_{i + 1}' for i in range(col_len1)] +
                               [f'{visit_feature_key}_{i + 1}' for i in range(col_len2)])
                    df = pd.DataFrame(columns=columns)

                    for patient in tqdm(raw_data, desc=f'Processing {monitor_feature_key1} to {visit_feature_key}'):
                        # 获取最后一个visit_id
                        last_visit_id = len(patient['procedures']) - 1
                        # 知道monitor有多长
                        monitor_length = len(patient[monitor_feature_key1][last_visit_id])
                        if args.task == "drug_rec":
                            if visit_feature_key == 'drugs':
                                # 为drug
                                filtered_visit_events = [event for event in patient[visit_feature_key] if
                                                         event in self.visit_event_token[
                                                             'drugs_hist'].vocabulary.token2idx]
                                visit_events = self.visit_event_token['drugs_hist'].convert_tokens_to_indices(
                                    filtered_visit_events)
                            else:
                                # 为cond和proc
                                visit_events = self.visit_event_token[visit_feature_key].convert_tokens_to_indices(
                                    patient[visit_feature_key][last_visit_id]
                                )
                        else:
                            if visit_feature_key == 'conditions':
                                # 为condition
                                filtered_visit_events = [event for event in patient[visit_feature_key] if
                                                         event in self.visit_event_token[
                                                             'cond_hist'].vocabulary.token2idx]
                                visit_events = self.visit_event_token['cond_hist'].convert_tokens_to_indices(
                                    filtered_visit_events)
                            else:
                                # 为proc和drug
                                visit_events = self.visit_event_token[visit_feature_key].convert_tokens_to_indices(
                                    patient[visit_feature_key][last_visit_id]
                                )

                        for monitor_id in range(monitor_length):
                            monitor_events1 = self.monitor_event_token[monitor_feature_key1].convert_tokens_to_indices(
                                patient[monitor_feature_key1][last_visit_id][monitor_id]
                            )
                            monitor_events2 = self.monitor_event_token[monitor_feature_key2].convert_tokens_to_indices(
                                patient[monitor_feature_key2][last_visit_id][monitor_id]
                            )

                            # 创建一个新行的Series
                            new_row = pd.Series(dtype=float)

                            # {visit_feature_key}_{visit_events}的列变为1
                            for event in visit_events:
                                new_row[f"{visit_feature_key}_{event}"] = 1

                            # {monitor_feature_key1}_{monitor_events1}的列变为{monitor_events2}
                            for m_event1, m_event2 in zip(monitor_events1, monitor_events2):
                                new_row[f"{monitor_feature_key1}_{m_event1}"] = m_event2

                            # 填充新行的空值为0
                            new_row = new_row.reindex(df.columns, fill_value=0)

                            # 将新行添加到 DataFrame
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                    # 将DataFrame添加到字典中
                    self.dfs[f'{monitor_feature_key1}_{visit_feature_key}'] = df

            for patient_info_key in ['age', 'weight']:
                for visit_feature_key in self.feature_visit_event_keys:
                    col_len2 = self.visit_event_token[visit_feature_key].get_vocabulary_size()

                    # 两个特殊情况
                    if visit_feature_key == 'drugs_hist':
                        visit_feature_key = 'drugs'
                    elif visit_feature_key == 'cond_hist':
                        visit_feature_key = 'conditions'

                    columns = ([patient_info_key] +
                               [f'{visit_feature_key}_{i + 1}' for i in range(col_len2)])
                    df = pd.DataFrame(columns=columns)

                    for patient in tqdm(raw_data, desc=f'Processing {patient_info_key} to {visit_feature_key}'):
                        # 获取最后一个visit_id
                        last_visit_id = len(patient['procedures']) - 1
                        if args.task == "drug_rec":
                            if visit_feature_key == 'drugs':
                                # 为drug
                                filtered_visit_events = [event for event in patient[visit_feature_key] if
                                                         event in self.visit_event_token[
                                                             'drugs_hist'].vocabulary.token2idx]
                                visit_events = self.visit_event_token['drugs_hist'].convert_tokens_to_indices(
                                    filtered_visit_events)
                            else:
                                # 为cond和proc
                                visit_events = self.visit_event_token[visit_feature_key].convert_tokens_to_indices(
                                    patient[visit_feature_key][last_visit_id]
                                )
                        else:
                            if visit_feature_key == 'conditions':
                                # 为condition
                                filtered_visit_events = [event for event in patient[visit_feature_key] if
                                                         event in self.visit_event_token[
                                                             'cond_hist'].vocabulary.token2idx]
                                visit_events = self.visit_event_token['cond_hist'].convert_tokens_to_indices(
                                    filtered_visit_events)
                            else:
                                # 为proc和drug
                                visit_events = self.visit_event_token[visit_feature_key].convert_tokens_to_indices(
                                    patient[visit_feature_key][last_visit_id]
                                )

                        # 创建一个新行的Series
                        new_row = pd.Series(dtype=float)

                        # {visit_feature_key}_{visit_events}的列变为1
                        for event in visit_events:
                            new_row[f"{visit_feature_key}_{event}"] = 1

                        new_row[patient_info_key] = patient[patient_info_key][last_visit_id]

                        # 填充新行的空值为0
                        new_row = new_row.reindex(df.columns, fill_value=0)

                        # 将新行添加到 DataFrame
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                    # 将DataFrame添加到字典中
                    self.dfs[f'{patient_info_key}_{visit_feature_key}'] = df

            dill.dump(self.dfs, open(data_path, 'wb'))
            print(f"Data matrix processed and saved to {data_path}")

    def learn_causal_effects(self, args):
        # 忽略特定的警告
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.PerfectSeparationWarning)

        if args.developer:
            data_path = f'data/{args.dataset}/processed_data/{args.task}/causal_matrix_developer.pkl'
        else:
            data_path = f'data/{args.dataset}/processed_data/{args.task}/casual_matrix.pkl'

        if os.path.exists(data_path):
            # 从df_path加载数据
            self.causal_effects_matrixs = dill.load(open(data_path, 'rb'))
            print(f"Causal Matrix loaded from {data_path}")
        else:
            # 如果文件不存在，开始处理数据
            print(f"Causal Matrix not found at {data_path}. Starting data processing.")

            for type_a in ['lab_item', 'inj_item', 'age', 'weight']:
                for type_b in self.visit_event_token:
                    if type_b == 'drugs_hist':
                        type_b = 'drugs'
                    elif type_b == 'cond_hist':
                        type_b = 'conditions'

                    # 此函数从self.df中学习typea-typeb的关系，并存储结果到self.causal_effects_matrix
                    items_a = [col for col in self.dfs[f'{type_a}_{type_b}'].columns if col.startswith(type_a)]
                    items_b = [col for col in self.dfs[f'{type_a}_{type_b}'].columns if col.startswith(type_b)]
                    # todo：这里赋值的是一个全1的矩阵，提交之前要把这里改一下
                    results_matrix = pd.DataFrame(1, index=items_a, columns=items_b)

                    # results_matrix = pd.DataFrame(index=items_a, columns=items_b)
                    # # 使用tqdm显示进度条
                    # for item_a in tqdm(items_a, desc=f"Processing {type_a}_{type_b}"):
                    #     for item_b in items_b:
                    #         model = CausalModel(data=self.dfs[f'{type_a}_{type_b}'].astype(float),
                    #                             treatment=item_a, outcome=item_b)
                    #
                    #         # 识别因果效应
                    #         identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                    #
                    #         # 估计因果效应
                    #         estimate = model.estimate_effect(
                    #             identified_estimand,
                    #             method_name="backdoor.generalized_linear_model",
                    #             method_params={"glm_family": sm.families.Binomial()}
                    #         )
                    #
                    #         # 存储结果到矩阵中
                    #         results_matrix.loc[item_a, item_b] = estimate.value

                    self.causal_effects_matrixs[f'{type_a}_{type_b}'] = results_matrix

            dill.dump(self.causal_effects_matrixs, open(data_path, 'wb'))
            print(f"Causal matrix processed and saved to {data_path}")

    def get_causal_effect(self, typea, ida, typeb, idb):
        # 此函数从self.causal_effects_matrix中找到指定的因果效应值
        a_column = f"{typea}_{ida}"
        b_column = f"{typeb}_{idb}"

        causal_effect = self.causal_effects_matrixs[f'{typea}_{typeb}'][a_column][b_column]
        return causal_effect
