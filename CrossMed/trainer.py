import numpy as np
import torch
import torch.nn.functional as F
from pyhealth.metrics import binary_metrics_fn
from tqdm import tqdm
from torch import autograd
# from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset
from baselines.CrossMed.utils import prepare_labels, get_sample_loader, visit_level, code_level, calculate_sensitivity, \
    calculate_specificity, prepare_pre_labels


def training_pre(data_loader, model, optimizer, label_name, device):
    model.train()
    train_loss = 0
    with tqdm(total=len(data_loader), desc="Training", unit="batch") as pbar:
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            label_state = prepare_pre_labels(data['patient_state']).to(device)

            out = model(data, pretrain=True)
            loss = F.mse_loss(out, label_state)
            # if torch.isnan(loss).any() or torch.isinf(loss).any():
            #     exit("损失中有NaN或Inf值！")
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            avg_loss = train_loss / (batch_idx + 1)

            # 网上搜的，清理显存，不然显存会叠起来，100g也不够用
            del data, out, loss
            if device != torch.device('cpu'):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    return avg_loss


def evaluating_pre(data_loader, model, label_name, device):
    model.eval()
    val_loss = 0
    y_t_all, y_p_all = [], []
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluating", unit="batch") as pbar:
            for batch_idx, data in enumerate(data_loader):
                label = prepare_pre_labels(data[label_name]).to(device)

                out = model(data, pretrain=True)
                loss = F.mse_loss(out, label)

                val_loss += loss.detach().cpu().numpy()
                avg_loss = val_loss / (batch_idx + 1)

                # y_t = label.cpu().numpy()
                # y_p = torch.sigmoid(out).detach().cpu().numpy()
                # y_t_all.append(y_t)
                # y_p_all.append(y_p)

                # 网上搜的，清理显存，不然显存会叠起来，100g也不够用
                del data, out, loss
                if device != torch.device('cpu'):
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")

            # y_true = np.concatenate(y_t_all, axis=0)
            # y_prob = np.concatenate(y_p_all, axis=0)
            # y_pred = np.where(y_prob > 0.5, 1, 0)

            # code_level_results = code_level(y_true, y_pred)
            # visit_level_results = visit_level(y_true, y_pred)

            # sensitivity = calculate_sensitivity(y_true, y_pred)
            # specificity = calculate_specificity(y_true, y_pred)
            # sensitivity = np.mean(sensitivity)
            # specificity = np.mean(specificity)

            # y_true = y_true.ravel()
            # y_prob = y_prob.ravel()
            #
            # metrics = binary_metrics_fn(y_true, y_prob,
            #                             metrics=["f1", "jaccard", "roc_auc", "pr_auc"])

    return avg_loss


def training(data_loader, model, label_tokenizer, optimizer, label_name, device):
    model.train()

    train_regression_loss = 0
    train_classification_loss = 0
    train_combined_loss = 0

    avg_regression_loss = 0
    avg_classification_loss = 0
    avg_combined_loss = 0

    with tqdm(total=len(data_loader), desc="Training", unit="batch") as pbar:
        for batch_idx, data in enumerate(data_loader):
            """这里开始搞monitoring层面的regression训练"""
            optimizer.zero_grad()
            label_state = prepare_pre_labels(data['patient_state']).to(device)

            if type(data) == dict:
                label = prepare_labels(data[label_name], label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0][label_name], label_tokenizer).to(device)

            out_reg, out_cls = model(data)

            if out_reg.shape != label_state.shape:
                out_reg = torch.zeros_like(label_state, device=device)

            loss_reg = F.mse_loss(out_reg, label_state)
            loss_cls = F.binary_cross_entropy_with_logits(out_cls, label)
            loss = 0.1 * loss_reg + 0.9 * loss_cls

            loss.backward()
            optimizer.step()

            train_regression_loss = train_regression_loss + loss_reg.item()
            avg_regression_loss = train_regression_loss / (batch_idx + 1)

            train_classification_loss = train_classification_loss + loss_cls.item()
            avg_classification_loss = train_classification_loss / (batch_idx + 1)

            train_combined_loss = train_combined_loss + loss.item()
            avg_combined_loss = train_combined_loss / (batch_idx + 1)

            # 网上搜的，清理显存，不然显存会叠起来，100g也不够用
            del data, out_reg, out_cls, loss
            if device != torch.device('cpu'):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

            # 更新进度条
            pbar.update(1)
            pbar.set_postfix(reg_loss=f"{avg_regression_loss:.4f}",
                             cls_loss=f"{avg_classification_loss:.4f}",
                             cmb_loss=f"{avg_combined_loss:.4f}")

    return avg_regression_loss, avg_classification_loss, avg_combined_loss


def evaluating(data_loader, model, label_tokenizer, label_name, device):
    model.eval()

    avg_regression_loss = 0
    avg_classification_loss = 0
    avg_combined_loss = 0

    val_regression_loss = 0
    val_classification_loss = 0
    val_combined_loss = 0

    y_t_all, y_p_all = [], []
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluating", unit="batch") as pbar:
            for batch_idx, data in enumerate(data_loader):

                """这里开始搞monitoring层面的regression训练"""
                label_state = prepare_pre_labels(data['patient_state']).to(device)
                if type(data) == dict:
                    label = prepare_labels(data[label_name], label_tokenizer).to(device)
                else:
                    label = prepare_labels(data[0][label_name], label_tokenizer).to(device)

                out_reg, out_cls = model(data)
                if out_reg.shape != label_state.shape:
                    out_reg = torch.zeros_like(label_state, device=device)
                
                loss_reg = F.mse_loss(out_reg, label_state)
                loss_cls = F.binary_cross_entropy_with_logits(out_cls, label)

                val_regression_loss = val_regression_loss + loss_reg.item()
                avg_regression_loss = val_regression_loss / (batch_idx + 1)

                val_classification_loss = val_classification_loss + loss_cls.item()
                avg_classification_loss = val_classification_loss / (batch_idx + 1)

                loss = 0.1 * loss_reg + 0.9 * loss_cls
                val_combined_loss = val_combined_loss + loss.item()
                avg_combined_loss = val_combined_loss / (batch_idx + 1)

                y_t = label.cpu().numpy()
                y_p = torch.sigmoid(out_cls).detach().cpu().numpy()
                y_t_all.append(y_t)
                y_p_all.append(y_p)

                # 网上搜的，清理显存，不然显存会叠起来，100g也不够用
                del data, out_reg, out_cls, loss
                if device != torch.device('cpu'):
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix(reg_loss=f"{avg_regression_loss:.4f}",
                                 cls_loss=f"{avg_classification_loss:.4f}",
                                 cmb_loss=f"{avg_combined_loss:.4f}")

            y_true = np.concatenate(y_t_all, axis=0)
            y_prob = np.concatenate(y_p_all, axis=0)
            y_pred = np.where(y_prob > 0.5, 1, 0)

            code_level_results = code_level(y_true, y_pred)
            visit_level_results = visit_level(y_true, y_pred)

            sensitivity = calculate_sensitivity(y_true, y_pred)
            specificity = calculate_specificity(y_true, y_pred)
            sensitivity = np.mean(sensitivity)
            specificity = np.mean(specificity)

            y_true = y_true.ravel()
            y_prob = y_prob.ravel()

            metrics = binary_metrics_fn(y_true, y_prob,
                                        metrics=["f1", "jaccard", "roc_auc", "pr_auc"])

    return avg_regression_loss, avg_classification_loss, avg_combined_loss, metrics, code_level_results, visit_level_results, sensitivity, specificity


def testing(data_loader, test_epochs, model, label_tokenizer, sample_size, label_name, device):
    results = []
    for epoch in range(test_epochs):
        print(f'\nTesting Epoch {epoch + 1}/{test_epochs}')
        sample_loader = get_sample_loader(data_loader, sample_size)

        _, _, _, metrics, code_level_results, visit_level_results, sensitivity, specificity = evaluating(sample_loader,
                                                                                                         model,
                                                                                                         label_tokenizer,
                                                                                                         label_name,
                                                                                                         device)

        # 打印结果
        print(f'F1: {metrics["f1"]:.4f}, '
              f'Jaccard: {metrics["jaccard"]:.4f}, '
              f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
              f'PR-AUC: {metrics["pr_auc"]:.4f}, '
              f'sensitivity: {sensitivity:.4f}, '
              f'specificity: {specificity:.4f}'
              )

        results.append([metrics["f1"], metrics["jaccard"], metrics["roc_auc"],
                        metrics["pr_auc"]] + list(code_level_results) + list(visit_level_results) +
                       [sensitivity, specificity])

    results = np.array(results)
    mean, std = results.mean(axis=0), results.std(axis=0)
    metric_list = ['F1', 'Jaccard', 'ROC-AUC', 'PR-AUC',
                   'code-10', 'code-20', 'code-30', 'code-40', 'code-50', 'code-60', 'code-70', 'code-80',
                   'visit-10', 'visit-20', 'visit-30', 'visit-40', 'visit-50', 'visit-60', 'visit-70', 'visit-80',
                   'sensitivity', 'specificity']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])

    return outstring
