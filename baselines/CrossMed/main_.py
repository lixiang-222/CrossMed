import argparse
from baselines.CrossMed.Task import initialize_task
from baselines.CrossMed.baselines.graphcare.graphcare import GraphCare
from baselines.CrossMed.utils import *
from baselines.CrossMed.utils import add_patient_state
from baselines.CrossMed.baselines.baseline import *
from baselines.CrossMed.baselines.refine_simple import Refine
# from my_baselines.my_baselines import *
from baselines.CrossMed.trainer import training, evaluating, testing, training_pre, evaluating_pre
from baselines.CrossMed.preprocess.data_load import preprocess_data
from baselines.CrossMed.models.graph_construction import process_data_with_graph
from baselines.CrossMed.baselines.graphcare.graph_construction4Graphcare import process_data_with_graph_for_graphcare
# from models.causal_construction import CausalAnalysis
# from baselines.CrossMed.models.model import CrossMed
# from baselines.CrossMed.models.model1 import CrossMed1
from baselines.CrossMed.models.model4 import CrossMed4
# from baselines.CrossMed.models.demo1 import CrossMed4  # only visit
import os
from baselines.CrossMed.baselines.drug_rec import drug_rec_from_pyhealth
from baselines.CrossMed.baselines.TRANS.models.Model import TRANS
from baselines.CrossMed.baselines.TRANS.data.Task import MMDataset
from baselines.CrossMed.baselines.TRANS.utils import mm_dataloader
# from dgl.data import split_dataset

# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
# parser.add_argument('--test_epochs', type=int, default=10, help='Number of epochs to test.')
# parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
# parser.add_argument('--model', type=str, default="CrossMed4",
#                     help='Transformer, RETAIN, StageNet, KAME, TRANS, GRU, REFINE'
#                          'SafeDrug ,GAMENet, micron, MoleRec, GRASP, CausalMed'
#                          'CrossMed')
# parser.add_argument('--device_id', type=int, default=1, help="choose a gpu id")
# parser.add_argument('--seed', type=int, default=222)
# parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
# parser.add_argument('--task', type=str, default="drug_rec", choices=['drug_rec', 'diag_pred'])
# parser.add_argument('--batch_size', type=int, default=4, help='batch size')
# parser.add_argument('--dim', type=int, default=128, help='embedding dim')
# parser.add_argument('--dropout', type=float, default=0.7, help='dropout rate')
# parser.add_argument('--developer', type=bool, default=True, help='developer mode')
# parser.add_argument('--test', type=bool, default=False, help='test mode')
# parser.add_argument('--pretrain', type=bool, default=False, help='need pretrained model')

# args = parser.parse_args()


def main(args):
    if args.developer:
        args.epochs = 3
        args.test_epochs = 2
        args.batch_size = 2
    set_random_seed(args.seed)
    print('{}--{}--{}--{}'.format(args.model, args.task, args.dataset, args.batch_size))
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")

    # 数据读取
    task_dataset = preprocess_data(args)

    # 任务定义
    Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = initialize_task(task_dataset, args)

    task_dataset_with_graph = \
        process_data_with_graph(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, None, args)

    # 新搞得一个东西，用于回归模型
    task_dataset_with_graph = add_patient_state(task_dataset_with_graph)

    # 切分数据
    train_loader, val_loader, test_loader = seq_dataloader(task_dataset_with_graph, batch_size=args.batch_size)

    """Model definition"""

    if args.model == 'Transformer':
        model = Transformer(Tokenizers_visit_event, label_size, device)

    elif (args.model == 'SafeDrug' or args.model == 'GAMENet' or args.model == 'micron'
          or args.model == 'MoleRec' or args.model == 'GRASP'):
        drug_rec_from_pyhealth(args.model, args.dataset, args.developer)

    elif args.model == 'CrossMed4':
        model = CrossMed4(Tokenizers_visit_event, Tokenizers_monitor_event, label_size, device)
    else:
        print("This model is not available.")
        return

    if args.task == "drug_rec":
        label_name = 'drugs'
    else:
        label_name = 'conditions'

    # Print the dataset statistics.
    print('parameter of dataset:',
          print_dataset_parameters(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, label_size, args))

    # If you need to print the model parameters, you can enable it.
    # print('number of parameters', count_parameters(model))
    # print_model_parameters(model)

    # 保存checkpoint的路径
    folder_path = './logs/{}_{}_{}_{}'.format(args.model, args.task, args.dataset, args.batch_size)
    os.makedirs(folder_path, exist_ok=True)
    ckpt_path = f'{folder_path}/best_model.ckpt'
    ckpt_pretrain_path = f'{folder_path}/best_pretrained_model.ckpt'
    png_path = f'{folder_path}/loss.png'
    png_pretrain_path = f'{folder_path}/loss_pretrain.png'
    txt_path = f'{folder_path}/final_result.txt'
    log_txt_path = f'{folder_path}/log.txt'
    log_pretrain_txt_path = f'{folder_path}/log_pretrain.txt'

    if not args.test:

        # 记录 loss 的列表
        epoch_list = []
        train_losses = []
        val_losses = []

        print('--------------------Begin Training--------------------')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best = float('inf')  # 无限大
        best_model = None
        for epoch in range(args.epochs):
            print(f'\nTraining Epoch {epoch + 1}/{args.epochs}')
            model = model.to(device)

            train_reg_loss, train_cls_loss, train_loss \
                = training(train_loader, model, label_tokenizer, optimizer, label_name, device)
            val_reg_loss, val_cls_loss, val_loss, metrics, code_level_results, visit_level_results, sensitivity, specificity \
                = evaluating(val_loader, model, label_tokenizer, label_name, device)

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
            log_results(epoch, train_reg_loss, train_cls_loss, train_loss, val_reg_loss, val_cls_loss, val_loss,
                        metrics, log_txt_path)

            if val_loss < best:
                best = val_loss
                best_model = model.state_dict()

            # 记录损失
            epoch_list.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # 每个epoch都绘制一次，绘制损失曲线
            plot_losses(epoch_list, train_losses, val_losses, png_path)

        # 这里本来可以每个epoch都保存一次，但是太大了，所以只保存一次
        torch.save(best_model, ckpt_path)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--test_epochs', type=int, default=10, help='Number of epochs to test.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--model', type=str, default="CrossMed4",
                        help='Transformer, RETAIN, StageNet, KAME, TRANS, GRU, REFINE'
                            'SafeDrug ,GAMENet, micron, MoleRec, GRASP, CausalMed'
                            'CrossMed')
    parser.add_argument('--device_id', type=int, default=1, help="choose a gpu id")
    parser.add_argument('--seed', type=int, default=222)
    parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
    parser.add_argument('--task', type=str, default="drug_rec", choices=['drug_rec', 'diag_pred'])
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--dim', type=int, default=128, help='embedding dim')
    parser.add_argument('--dropout', type=float, default=0.7, help='dropout rate')
    parser.add_argument('--developer', type=bool, default=True, help='developer mode')
    parser.add_argument('--test', type=bool, default=False, help='test mode')
    parser.add_argument('--pretrain', type=bool, default=False, help='need pretrained model')

    args = parser.parse_args()
    main(args)
