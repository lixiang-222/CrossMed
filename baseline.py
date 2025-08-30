from pyhealth.models import SafeDrug, Transformer, MoleRec, GRASP, GAMENet, RETAIN, MICRON, StageNet
from pyhealth.datasets import get_dataloader, split_by_patient

from baselines import multilabel, drug_rec_trainer
from baselines.baselines import KAME

from preprocess.data_load import load_preprocessed_data
import torch.nn as nn
import numpy as np

def drug_rec_from_pyhealth(model_name, dataset, epochs, developer):
    if developer:
        data = load_preprocessed_data(f"data/{dataset}/processed_data/drug_rec/processed_developer_data.pkl")
        epochs = 2
    else:
        data = load_preprocessed_data(f"data/{dataset}/processed_data/drug_rec/processed_data.pkl")
        epochs = epochs

    if model_name == 'SafeDrug':
        model = SafeDrug(
            dataset=data,
            embedding_dim=128,
            hidden_dim=128,
            num_layers=1,
            dropout=0.5,
        )
    elif model_name == 'transformer':
        model = Transformer(
            dataset=data,
            embedding_dim=128,
            feature_keys=["conditions", "procedures", "drugs_hist"],
            label_key="drugs",
            mode="multilabel",
        )
    elif model_name == 'MoleRec':
        model = MoleRec(
            dataset=data,
            embedding_dim=128,
            hidden_dim=128,
            num_layers=1,
            dropout=0.5,
        )
    elif model_name == 'GRASP':
        model = GRASP(
            dataset=data,
            feature_keys=['procedures', 'conditions'],
            label_key='drugs',
            mode='multilabel',
            use_embedding=[True, True],
        )
    elif model_name == 'GAMENet':
        model = GAMENet(
            dataset=data,
        )
    elif model_name == 'retain':
        model = RETAIN(
            dataset=data,
            feature_keys=['procedures', 'conditions'],
            label_key='drugs',
            mode='multilabel'
        )
    elif model_name == 'micron':
        model = MICRON(
            dataset=data,
        )
    elif model_name == 'StageNet':
        model = StageNet(
            dataset=data,
            feature_keys=['procedures', 'conditions'],
            label_key='drugs',
            mode='multilabel'
        )
    elif model_name == 'KAME':
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        from Task import initialize_task
        Tokenizers_visit_event, _, label_tokenizer, _ = initialize_task(data, type('Args', (), {'task': 'drug_rec', 'dataset': dataset}))
        
        # 在数据集info和tokenizers中添加映射
        data.input_info['cond_hist'] = data.input_info['conditions']
        Tokenizers_visit_event['cond_hist'] = Tokenizers_visit_event['conditions']
        
        # 在每个样本中也添加cond_hist键
        for sample in data.samples:
            sample['cond_hist'] = sample['conditions']
        
        from utils import get_parent_tokenizers  
        Tokenizers_visit_event.update(get_parent_tokenizers(data, keys=['cond_hist', 'procedures']))
        
        # 添加conditions_parent映射
        Tokenizers_visit_event['conditions_parent'] = Tokenizers_visit_event['cond_hist_parent']
        Tokenizers_visit_event['conditions'] = Tokenizers_visit_event['cond_hist']
        model = KAME(Tokenizers_visit_event, len(label_tokenizer.vocabulary), device)

        # 在创建model后，先检查可用的键
        print("Available embeddings:", list(model.embeddings.keys()))
        print("Available RNN:", list(model.rnn.keys()))
        print("Available knowledge_map:", list(model.knowledge_map.keys()))
        # 手动添加conditions的嵌入和RNN层
        model.embeddings['conditions'] = model.embeddings['cond_hist'] 
        model.rnn['conditions'] = model.rnn['cond_hist']
        model.knowledge_map['conditions'] = model.knowledge_map['cond_hist']  # 
        model.knowledge_map['drugs_hist'] = model.knowledge_map['cond_hist']  # 
        # 在这里添加FC层调整
        actual_input_dim = 1024  # 根据调试信息
        model.fc = nn.Linear(actual_input_dim, len(label_tokenizer.vocabulary)).to(device)

    else:
        return

    train_ds, val_ds, test_ds = split_by_patient(data, [0.75, 0.1, 0.15])
    train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=16, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=16, shuffle=False)

    trainer = drug_rec_trainer.Trainer(model,
                                       metrics=['f1_samples', 'jaccard_samples', 'pr_auc_samples', 'roc_auc_samples',
                                                'visit_level', 'code_level', "sensitivity", "specificity"])
    results = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor='loss',
        monitor_criterion='min',
        test_dataloader=test_loader,
    )

    y_true, y_prob, loss = trainer.inference(test_loader)
    output = multilabel.multilabel_metrics_fn(y_true, y_prob,
                                              metrics=["f1_samples", "jaccard_samples", "pr_auc_samples",
                                                       "roc_auc_samples", "visit_level", "code_level", "sensitivity",
                                                       "specificity"])
    
    print(f'{model_name}--{dataset}--epochs{epochs}')

    results = np.array(results)
    mean, std = results.mean(axis=0), results.std(axis=0)
    metric_list = ['f1_samples', 'jaccard_samples', 'roc_auc_samples', 'pr_auc_samples']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    print(f'code_level: {output["code_level"]} ')
    print(f'visit_level: {output["visit_level"]}')
    print(f'sensitivity: {output["sensitivity"]} ')
    print(f'specificity: {output["specificity"]}')

if __name__ == "__main__":
    drug_rec_from_pyhealth('transformer', 'mimic3', 10, True)
    
    # 'retain', 'transformer', 'SafeDrug', 'GAMENet', 'micron', 'MoleRec', 'GRASP'
