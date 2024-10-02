import os
import dill
from preprocess.drug_recommendation_mimic34_fn import *
from preprocess.diag_prediction_mimic34_fn import *

from joblib import load

from OverWrite_mimic3 import MIMIC3Dataset
from OverWrite_mimic4 import MIMIC4Dataset


# 定义保存数据的函数
def save_preprocessed_data(data, filepath):
    print("Saving data...")
    dill.dump(data, open(filepath, 'wb'))
    print(f"Data saved to {filepath}")


def load_preprocessed_data(filepath):
    data = dill.load(open(filepath, 'rb'))
    print(f"Data loaded from {filepath}")
    return data


def load_dataset(dataset, root, tables=None, task_fn=None, dev=False):
    if dataset == 'mimic3':
        dataset = MIMIC3Dataset(
            root=root,
            dev=dev,
            tables=tables,
            # NDC->ATC3的编码映射
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                          "ICD9CM": "CCSCM",
                          "ICD9PROC": "CCSPROC"
                          },
            refresh_cache=False
        )

    elif dataset == 'mimic4':
        dataset = MIMIC4Dataset(
            root=root,
            dev=dev,
            tables=tables,
            code_mapping={
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                "ICD9CM": "CCSCM",
                "ICD9PROC": "CCSPROC",
                "ICD10CM": "CCSCM",
                "ICD10PROC": "CCSPROC",
            },
            refresh_cache=False,
        )
    else:
        return load(root)
    # mimic3_sample1 = dataset
    # mimic3_sample2 = dataset.set_task(task_fn=task_fn)
    # mimic4_sample1 = dataset
    # mimic4_sample2 = dataset.set_task(task_fn=task_fn)
    # print(dataset.stat()) 这个不是很准，不知道为什么
    return dataset.set_task(task_fn=task_fn)


def preprocess_data(args):
    """数据预处理"""

    raw_data_path = f"data/{args.dataset}/raw_data"

    if args.developer:
        processed_data_path = f'data/{args.dataset}/processed_data/{args.task}/processed_developer_data.pkl'
    else:
        processed_data_path = f'data/{args.dataset}/processed_data/{args.task}/processed_data.pkl'

    if os.path.exists(processed_data_path):
        task_dataset = load_preprocessed_data(processed_data_path)
    else:
        print(f"数据不存在，开始做{args.dataset}数据集在{args.task}任务的数据预处理")
        # 数据预处理逻辑
        if args.dataset == 'mimic3':
            if args.task == 'drug_rec':
                task_dataset = load_dataset(args.dataset,
                                            tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                                    "INPUTEVENTS_MV"],
                                            root=raw_data_path,
                                            task_fn=drug_recommendation_mimic3_fn,
                                            dev=args.developer)
            elif args.task == 'diag_pred':
                task_dataset = load_dataset(args.dataset,
                                            tables=['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS', "LABEVENTS",
                                                    "INPUTEVENTS_MV"],
                                            root=raw_data_path,
                                            task_fn=diag_prediction_mimic3_fn,
                                            dev=args.developer)
            else:
                raise ValueError("检查一下这个task")
        elif args.dataset == 'mimic4':
            if args.task == 'drug_rec':
                task_dataset = load_dataset(args.dataset,
                                            tables=['diagnoses_icd', 'procedures_icd', 'prescriptions', 'labevents',
                                                    'inputevents_mv'],
                                            root=raw_data_path,
                                            task_fn=drug_recommendation_mimic4_fn,
                                            dev=False)
            elif args.task == 'diag_pred':
                task_dataset = load_dataset(args.dataset,
                                            tables=['diagnoses_icd', 'procedures_icd', 'prescriptions', 'labevents',
                                                    'inputevents_mv'],
                                            root=raw_data_path,
                                            task_fn=diag_prediction_mimic4_fn,
                                            dev=False)
            else:
                raise ValueError("检查一下这个task")
        else:
            raise ValueError("检查一下dataset，没有这个数据集")

        save_preprocessed_data(task_dataset, processed_data_path)

    return task_dataset
