import argparse
from Task import initialize_task
from utils import *
from baselines.baseline import *
from trainer import training, evaluating, testing
from preprocess.data_load import preprocess_data
from models.graph_construction import process_data_with_graph
from models.model import CrossMed
import os
from baselines.drug_rec import drug_rec_from_pyhealth
from baselines.TRANS.models.Model import TRANS
from baselines.TRANS.data.Task import MMDataset
from baselines.TRANS.utils import mm_dataloader
from dgl.data import split_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--test_epochs', type=int, default=10, help='Number of epochs to test.')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
parser.add_argument('--model', type=str, default="CrossMed",
                    help='Transformer, RETAIN, StageNet, KAME, TRANS, GRU, REFINE'
                         'SafeDrug ,GAMENet, micron, MoleRec, GRASP, CausalMed'
                         'CrossMed')
parser.add_argument('--device_id', type=int, default=0, help="choose a gpu id")
parser.add_argument('--seed', type=int, default=222)
parser.add_argument('--dataset', type=str, default="mimic3", choices=['mimic3', 'mimic4'])
parser.add_argument('--task', type=str, default="drug_rec", choices=['drug_rec', 'diag_pred'])
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--dim', type=int, default=128, help='embedding dim')
parser.add_argument('--dropout', type=float, default=0.7, help='dropout rate')
parser.add_argument('--developer', type=bool, default=True, help='developer mode')
parser.add_argument('--test', type=bool, default=False, help='test mode')

args = parser.parse_args()


def main(args):
    if args.developer:
        args.epochs = 3
        args.test_epochs = 2
        args.batch_size = 2
    set_random_seed(args.seed)
    print('{}--{}--{}--{}'.format(args.model, args.task, args.dataset, args.batch_size))
    cuda_id = "cuda:" + str(args.device_id)
    device = torch.device(cuda_id if torch.cuda.is_available() else "cpu")

    # Data loading
    task_dataset = preprocess_data(args)

    # Task definition
    Tokenizers_visit_event, Tokenizers_monitor_event, label_tokenizer, label_size = initialize_task(task_dataset, args)

    task_dataset_with_graph = \
        process_data_with_graph(task_dataset, Tokenizers_visit_event, Tokenizers_monitor_event, None, args)

    # Split the data.
    train_loader, val_loader, test_loader = seq_dataloader(task_dataset_with_graph, batch_size=args.batch_size)

    """Model definition"""
    # TODO
    #  Most of the baselines are defined here.
    #  For REFINE and CausalMed, you can directly refer to the code provided in their respective papers.
    # Get the baseline working.
    if args.model == 'Transformer':
        model = Transformer(Tokenizers_visit_event, label_size, device)

    elif args.model == 'GRU':
        model = GRU(Tokenizers_visit_event, label_size, device)

    elif args.model == 'RETAIN':
        model = RETAIN(Tokenizers_visit_event, label_size, device)

    elif args.model == 'KAME':
        Tokenizers_visit_event.update(get_parent_tokenizers(task_dataset))
        model = KAME(Tokenizers_visit_event, label_size, device)

    elif args.model == 'StageNet':
        model = StageNet(Tokenizers_visit_event, label_size, device)

    elif args.model == 'TRANS':
        mdataset = MMDataset(task_dataset, Tokenizers_visit_event, dim=args.dim, device=device, task=args.task,
                             trans_dim=4)
        trainset, validset, testset = split_dataset(mdataset)
        train_loader, val_loader, test_loader = mm_dataloader(trainset, validset, testset, batch_size=args.batch_size)
        model = TRANS(Tokenizers_visit_event, args.dim, label_size, device, args.task)

    # TODO
    #  Recommend running in drug_rec.py, as there are some issues executing it here.
    elif (args.model == 'SafeDrug' or args.model == 'GAMENet' or args.model == 'micron'
          or args.model == 'MoleRec' or args.model == 'GRASP'):
        drug_rec_from_pyhealth(args.model, args.dataset, args.developer)


    # OUR MODEL!
    elif 'CrossMed' in args.model:
        model = CrossMed(Tokenizers_visit_event, Tokenizers_monitor_event, label_size, device)

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

    # Checkpoint saving path
    folder_path = './logs/{}_{}_{}_{}'.format(args.model, args.task, args.dataset, args.batch_size)
    os.makedirs(folder_path, exist_ok=True)
    ckpt_path = f'{folder_path}/best_model.ckpt'
    png_path = f'{folder_path}/loss.png'
    txt_path = f'{folder_path}/final_result.txt'
    log_txt_path = f'{folder_path}/log.txt'

    if not args.test:
        # Loss logging list
        epoch_list = []
        train_losses = []
        val_losses = []

        print('--------------------Begin Training--------------------')
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        best = float('inf')  # Infinity
        best_model = None
        for epoch in range(args.epochs):
            print(f'\nTraining Epoch {epoch + 1}/{args.epochs}')
            model = model.to(device)

            train_loss = training(train_loader, model, label_tokenizer, optimizer, label_name, device)
            val_loss, metrics, code_level_results, visit_level_results, sensitivity, specificity \
                = evaluating(val_loader, model, label_tokenizer, label_name, device)

            # Format two ndarrays.
            code_level_results = ', '.join(map(lambda x: f"{x:.4f}", code_level_results))
            visit_level_results = ', '.join(map(lambda x: f"{x:.4f}", visit_level_results))

            # print results
            print(f'F1: {metrics["f1"]:.4f}, '
                  f'Jaccard: {metrics["jaccard"]:.4f}, '
                  f'ROC-AUC: {metrics["roc_auc"]:.4f}, '
                  f'PR-AUC: {metrics["pr_auc"]:.4f}, '
                  f'code_level: {code_level_results}, '
                  f'visit_level: {visit_level_results},'
                  f'sensitivity: {sensitivity}, '
                  f'specificity: {specificity}'
                  )

            # Log the results to log.txt
            log_results(epoch, train_loss, val_loss, metrics, log_txt_path)

            if val_loss < best:
                best = val_loss
                best_model = model.state_dict()

            # Log the loss
            epoch_list.append(epoch + 1)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Plot the loss curve at each epoch.
            plot_losses(epoch_list, train_losses, val_losses, png_path)

        # It was originally possible to save at each epoch, but it's too large, so it's saved only once.
        torch.save(best_model, ckpt_path)

    print('--------------------Begin Testing--------------------')
    # loading model/parameters
    best_model = torch.load(ckpt_path)
    model.load_state_dict(best_model)
    model = model.to(device)

    # start testing
    sample_size = 0.8  # it is a common practice to use 80%
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
    main(args)
