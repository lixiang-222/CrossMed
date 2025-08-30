import random
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.medcode import InnerMap
from pyhealth.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Subset


# 此文件修改的地方为67行加入了LABEVENTS以及将cond_hist对应变为drugs_hist

def batch_to_multihot(label, num_labels: int) -> torch.tensor:
    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot

def log_params(dataset, args, log_path):
    argsDict = args.__dict__
    with open(log_path, 'w') as log_file:
        log_file.write(f'parameter of dataset:\n')
        log_file.write(f'{dataset} \n')
        log_file.write('--------------------------------------\n')
        log_file.write(f'args params:\n')
        for eachArg, value in argsDict.items():
            log_file.writelines(eachArg + ' : ' + str(value) + '\n')
        log_file.write('--------------------------------------\n')


def log_results(epoch, run_time, train_loss, val_loss, metrics, log_path):
    with open(log_path, 'a') as log_file:
        log_file.write(f'Epoch {epoch + 1}\n')
        log_file.write(f'Train Loss: {train_loss:.4f}\n')
        log_file.write(f'Validation Loss: {val_loss:.4f}\n')
        log_file.write(f'F1: {metrics["f1"]:.4f}, '
                       f'Jaccard: {metrics["jaccard"]:.4f}, '
                       f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                       f'PR-AUC: {metrics["pr_auc"]:.4f}, '
                       f'Run_Time: {run_time:.3f}\n')
        if (epoch + 1) % 20 == 0:
            log_file.write(f'{epoch + 1} epoch Model saved !!!\n')
        log_file.write('--------------------------------------\n')

def log_outmemory(data, error_log, log_path):
    with open(log_path, 'a') as log_file:
        log_file.write(f'{error_log}\n')
        log_file.write('--------------------------------------\n')


def prepare_labels(
        labels,
        label_tokenizer: Tokenizer,
) -> torch.Tensor:
    labels_index = label_tokenizer.batch_encode_2d(
        labels, padding=False, truncation=False
    )
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = batch_to_multihot(labels_index, num_labels)
    return labels


# def get_init_tokenizers(task_dataset, keys=['drugs_hist', 'procedures', 'drugs']):
#     Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
#     return Tokenizers
def get_init_tokenizers(task_dataset, keys=None):
    Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
    return Tokenizers


def get_parent_tokenizers(task_dataset, keys=['cond_hist', 'procedures']):
    parent_tokenizers = {}
    dictionary = {'cond_hist': InnerMap.load("ICD9CM"), 'procedures': InnerMap.load("ICD9PROC")}
    for feature_key in keys:
        assert feature_key in dictionary.keys()
        tokens = task_dataset.get_all_tokens(feature_key)
        parent_tokens = set()
        for token in tokens:
            try:
                parent_tokens.update(dictionary[feature_key].get_ancestors(token))
            except:
                continue
        parent_tokenizers[feature_key + '_parent'] = Tokenizer(tokens=list(parent_tokens), special_tokens=["<pad>"])
    return parent_tokenizers


def seq_dataloader(dataset, split_ratio=[0.75, 0.1, 0.15], batch_size=64):
    train_dataset, val_dataset, test_dataset = split_by_patient(dataset, split_ratio)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 这三句我都不知道干啥的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


# 纯gpt写的
def get_sample_loader(data_loader, sample_size):
    sample_size = round(len(data_loader.dataset) * sample_size)
    dataset = data_loader.dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)  # 随机打乱索引
    sample_indices = indices[:sample_size]  # 取前 sample_size 个索引
    subset = Subset(dataset, sample_indices)
    sample_loader = DataLoader(subset, batch_size=data_loader.batch_size, shuffle=True,
                               collate_fn=data_loader.collate_fn)
    return sample_loader


def plot_losses(epoch_list, train_losses, val_losses, png_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_losses, label='Train Loss')
    plt.plot(epoch_list, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # 使用 numpy 找到 val_losses 的最小值及其索引
    min_val_loss_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_loss_idx]
    min_epoch = epoch_list[min_val_loss_idx]
    
    # 标注 val_losses 的最低点
    plt.scatter(min_epoch, min_val_loss, color='red', zorder=5)  # 使用红色点标注最低点
    plt.text(min_epoch, min_val_loss, f'Min Val Loss\n({min_epoch:.0f}, {min_val_loss:.4f})', 
             horizontalalignment='right', verticalalignment='bottom', fontsize=9, color='red')

    # 设置纵坐标范围
    plt.autoscale(True)
    # 保存绘图
    plt.savefig(png_path)
    plt.close()


def code_level(labels, predicts):
    labels = np.array(labels)
    total_labels = np.where(labels == 1)[0].shape[0]
    top_ks = [10, 20, 30, 40, 50, 60, 70, 80]
    total_correct_preds = []
    for k in top_ks:
        correct_preds = 0
        for i, pred in enumerate(predicts):
            index = np.argsort(-pred)[:k]
            for ind in index:
                if labels[i][ind] == 1:
                    correct_preds = correct_preds + 1
        total_correct_preds.append(float(correct_preds))

    total_correct_preds = np.array(total_correct_preds) / total_labels
    return total_correct_preds


def visit_level(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    top_ks = [10, 20, 30, 40, 50, 60, 70, 80]
    precision_at_ks = []
    for k in top_ks:
        precision_per_patient = []
        for i in range(len(labels)):
            actual_positives = np.sum(labels[i])
            denominator = min(k, actual_positives)
            top_k_indices = np.argsort(-predicts[i])[:k]
            true_positives = np.sum(labels[i][top_k_indices])
            precision = true_positives / denominator if denominator > 0 else 0
            precision_per_patient.append(precision)
        average_precision = np.mean(precision_per_patient)
        precision_at_ks.append(average_precision)
    return precision_at_ks


# Calculate the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_parameters(model):
    print(f"{'Module':<30} {'Parameters':<15}")
    print('-' * 45)
    total_params = 0
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += module_params
        print(f"{name:<30} {module_params:<15,}")

    # Print total parameters
    print('-' * 45)
    print(f"{'Total Parameters':<30} {total_params:<15,}")


def print_dataset_parameters(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, label_size, args):
    patient_num = len(task_dataset.patient_to_index)
    visit_num = len(task_dataset.visit_to_index)
    if args.task == 'drug_rec':
        cond_num = len(Tokenizers_visit_event['conditions'].vocabulary)
        drug_num = label_size
    else:  # args.task == 'diag_pred'
        cond_num = label_size
        drug_num = len(Tokenizers_visit_event['drugs'].vocabulary)
    proc_num = len(Tokenizers_visit_event['procedures'].vocabulary)
    lab_num = len(Tokenizers_monitor_event['lab_item'].vocabulary)
    inj_num = len(Tokenizers_monitor_event['inj_item'].vocabulary)

    cond = 0
    proc = 0
    drug = 0
    max_cond = 0
    max_proc = 0
    max_drug = 0
    max_visit = 0
    for visit in task_dataset.samples:
        if args.task == 'drug_rec':
            cond += len(visit['conditions'][-1])
            proc += len(visit['procedures'][-1])
            drug += len(visit['drugs'])
            if len(visit['conditions'][-1]) > max_cond:
                max_cond = len(visit['conditions'][-1])
            if len(visit['procedures'][-1]) > max_proc:
                max_proc = len(visit['procedures'][-1])
            if len(visit['drugs']) > max_drug:
                max_drug = len(visit['drugs'])
            if len(visit['conditions']) > max_visit:
                max_visit = len(visit['conditions'])
        elif args.task == 'diag_pred':
            cond += len(visit['conditions'])
            proc += len(visit['procedures'][-1])
            drug += len(visit['drugs'][-1])
            if len(visit['conditions'][-1]) > max_cond:
                max_cond = len(visit['conditions'][-1])
            if len(visit['procedures'][-1]) > max_proc:
                max_proc = len(visit['procedures'][-1])
            if len(visit['drugs']) > max_drug:
                max_drug = len(visit['drugs'])
            if len(visit['conditions']) > max_visit:
                max_visit = len(visit['conditions'])

    avg_visit = visit_num / patient_num
    avg_cond = cond / visit_num
    avg_proc = proc / visit_num
    avg_drug = drug / visit_num

    output = 'patient_num: {}\nvisit_num: {}\ncond_num: {}\ndrug_num: {}\nproc_num: {}\nlab_num: {}\ninj_num:{} ' \
             '\navg_visit: {}\navg_cond: {}\navg_proc: {}\navg_drug: {}\nmax_cond:{}\nmax_proc:{}\nmax_drug:{}' \
             '\nmax_visit: {}'.format(
        patient_num, visit_num, cond_num, drug_num, proc_num, lab_num, inj_num, avg_visit, avg_cond, avg_proc,
        avg_drug,max_cond,max_proc,max_drug, max_visit)
    return output


# 计算敏感性
def calculate_sensitivity(label, predict):
    if predict.shape != label.shape:
        raise ValueError("predict 和 label 的形状必须一致")

    # 计算真阳性和真实阳性
    true_positives = np.sum((predict == 1) & (label == 1), axis=1)
    total_positives = np.sum(label == 1, axis=1)

    # 计算敏感性，避免除零错误
    sensitivity = np.divide(true_positives, total_positives, out=np.zeros_like(true_positives, dtype=float),
                            where=total_positives != 0)

    return sensitivity

# 计算特异性
def calculate_specificity(label, predict):
    if predict.shape != label.shape:
        raise ValueError("predict 和 label 的形状必须一致")

    # 计算真阴性和真实阴性
    true_negatives = np.sum((predict == 0) & (label == 0), axis=1)
    total_negatives = np.sum(label == 0, axis=1)

    # 计算特异性，避免除零错误
    specificity = np.divide(true_negatives, total_negatives, out=np.zeros_like(true_negatives, dtype=float),
                            where=total_negatives != 0)

    return specificity
