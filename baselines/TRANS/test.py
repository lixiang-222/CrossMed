
import argparse
from dgl.data import split_dataset
from baselines.TRANS.models.Model import TRANS
from Task import initialize_task
from utils import *
from baselines.baselines import *
from my_baselines.my_baselines import *
from trainer import training, evaluating, testing, EarlyStopper
from preprocess.data_load import preprocess_data
from models.graph_construction import process_data_with_graph
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from rdkit import Chem
from pyhealth.medcode import ATC
from typing import Any, Dict, List, Tuple, Optional, Union
from torch_geometric import loader
from datetime import datetime
from models.model import PersonalMed
from models.Base2_1 import Base2_1
from models.Base2_1_1 import Base2_1_1
from models.base2_1_2 import Base2_1_2
from models.base2_1_3for_LX import Base2_1_3
from models.mlp import MLP
from models.TimeSpace import TimeSpace
from models.gnn import drug_sdf_db
import pickle
import os
import time
import dill
from baselines.TRANS.models.Model import TRANS
from Task import initialize_task
from baselines.TRANS.data.Task import MMDataset
from baselines.TRANS.utils import mm_dataloader

def run_Trans(args):
    set_random_seed(args.seed)
    print('{}--{}--{}--{}'.format(args.model, args.task, args.dataset, args.batch_size))
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")

    # 数据读取
    task_dataset = preprocess_data(args)

    # 任务定义
    Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = initialize_task(task_dataset, args)

    # 切分数据
    train_loader, val_loader, test_loader = seq_dataloader(task_dataset, batch_size=args.batch_size)

    mdataset = MMDataset(task_dataset, Tokenizers_visit_event, dim=args.dim, device=device, task=args.task,
                             trans_dim=4)
    trainset, validset, testset = split_dataset(mdataset)
    train_loader, val_loader, test_loader = mm_dataloader(trainset, validset, testset, batch_size=args.batch_size)
    model = TRANS(Tokenizers_visit_event, args.dim, label_size, device, args.task)


    if args.task == "drug_rec":
        label_name = 'drugs'
    elif args.task == "drug_rec_ts":
        label_name = 'drugs'
    elif args.task == "diag_pred_ts":
        label_name = 'conditions'
    else:
        label_name = 'conditions'

    # 打印数据集的统计信息
    dataset_output = print_dataset_parameters(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, label_size, args)
    print('parameter of dataset:', dataset_output)

    # 打印模型参数的，需要可以开
    # print('number of parameters', count_parameters(model))
    # print_model_parameters(model)

    # 保存checkpoint的路径
    folder_path = './logs/{}_{}_{}_{}_{}'.format(args.model, args.task, args.dataset, args.batch_size, args.notes)
    os.makedirs(folder_path, exist_ok=True)
    ckpt_path = f'{folder_path}/best_model.ckpt'
    png_path = f'{folder_path}/loss.png'
    txt_path = f'{folder_path}/final_result.txt'
    log_txt_path = f'{folder_path}/log.txt'
    log_outmemory_txt_path = f'{folder_path}/log_outmemory.txt'

    jaccard_ckpt_path = f'{folder_path}/best_model_jaccard.ckpt'
    final_jaccard_model_log = f'{folder_path}/final_result_jaccard.txt'

    if not args.test:
        # 记录 loss 的列表
        epoch_list = []
        train_losses = []
        val_losses = []

        log_params(dataset_output, args, log_txt_path)

        print('--------------------Begin Training--------------------')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        if args.scheduler:
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best = float('inf')  # 无限大
        best_jaccard = float('-inf')
        best_model = None
        best_model_jaccard = None
        for epoch in range(args.epochs):
            start_time = time.time()
            print(f'\nTraining Epoch {epoch + 1}/{args.epochs}')
            model = model.to(device)

            train_loss = training(train_loader, model, label_tokenizer, optimizer, label_name, log_outmemory_txt_path, device)
            val_loss, metrics, code_level_results, visit_level_results, sensitivity, specificity \
                = evaluating(val_loader, model, label_tokenizer, label_name, device)
            
            end_time = time.time()
            run_time = end_time - start_time

            # 对两个ndarray进行格式化
            code_level_results = ', '.join(map(lambda x: f"{x:.4f}", code_level_results))
            visit_level_results = ', '.join(map(lambda x: f"{x:.4f}", visit_level_results))

            # 打印结果
            print(f'F1: {metrics["f1"]:.4f}, '
                  f'Jaccard: {metrics["jaccard"]:.4f}, '
                  f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                  f'PR-AUC: {metrics["pr_auc"]:.4f}, '
                  f'code_level: {code_level_results}, '
                  f'visit_level: {visit_level_results},'
                  f'sensitivity: {sensitivity}, '
                  f'specificity: {specificity}'
                  )

            # 记录结果到 log.txt
            log_results(epoch, run_time, train_loss, val_loss, metrics, log_txt_path)

            if val_loss < best:
                best = val_loss
                best_model = model.state_dict()
            
            if metrics["jaccard"] > best_jaccard:
                best_jaccard = metrics["jaccard"]
                best_model_jaccard = model.state_dict()

            if (epoch + 1) % 20 == 0:
                torch.save(best_model, ckpt_path)
                torch.save(best_model_jaccard, jaccard_ckpt_path)

            # 记录损失
            epoch_list.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 每个epoch都绘制一次，绘制损失曲线
            plot_losses(epoch_list, train_losses, val_losses, png_path)

            # 学习率递减
            if args.scheduler:
                scheduler.step()

        # 这里本来可以每个epoch都保存一次，但是太大了，所以只保存一次
        torch.save(best_model, ckpt_path)
        torch.save(best_model_jaccard, jaccard_ckpt_path)

    print('--------------------Begin Testing--------------------')
    # 读取最新的model
    best_model = torch.load(ckpt_path)
    model.load_state_dict(best_model)
    model = model.to(device)

    # 开始测试
    sample_size = 0.8  # 国际惯例选取0.8
    outstring = testing(test_loader, args.test_epochs, model, label_tokenizer, sample_size, label_name, device)

    # 输出结果
    print("\nFinal test result:")
    print(outstring)
    with open(txt_path, 'w+') as file:
        file.write("model_path:")
        file.write(ckpt_path)
        file.write('\n')
        file.write(outstring)
    

    # 读取最新的model_jaccard
    best_model_jaccard = torch.load(jaccard_ckpt_path)
    model.load_state_dict(best_model_jaccard)
    model = model.to(device)

    outstring_jaccard = testing(test_loader, args.test_epochs, model, label_tokenizer, sample_size, label_name, device)

    # 输出结果
    print("\nFinal test result(jaccard):")
    print(outstring_jaccard)
    with open(final_jaccard_model_log, 'w+') as file:
        file.write("model_path:")
        file.write(jaccard_ckpt_path)
        file.write('\n')
        file.write(outstring_jaccard)

if __name__ == '__main__':
    run_Trans(args)
