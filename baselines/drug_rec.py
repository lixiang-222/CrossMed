from pyhealth.models import SafeDrug, Transformer, MoleRec, GRASP, GAMENet, RETAIN, MICRON
from pyhealth.datasets import get_dataloader, split_by_patient

from baselines import multilabel, drug_rec_trainer
from preprocess.data_load import load_preprocessed_data


def drug_rec_from_pyhealth(model_name, dataset, developer):
    if developer:
        data = load_preprocessed_data(f"data/{dataset}/processed_data/drug_rec/processed_developer_data.pkl")
        epochs = 2
    else:
        data = load_preprocessed_data(f"data/{dataset}/processed_data/drug_rec/processed_data.pkl")
        epochs = 30

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
    else:
        return

    train_ds, val_ds, test_ds = split_by_patient(data, [0.75, 0.1, 0.15])
    train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=16, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=16, shuffle=False)

    trainer = drug_rec_trainer.Trainer(model,
                                       metrics=['f1_samples', 'jaccard_samples', 'pr_auc_samples', 'roc_auc_samples',
                                                'visit_level', 'code_level', "sensitivity", "specificity"])
    trainer.train(
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

    print(f'{model_name},{dataset},{developer},{output}')


if __name__ == "__main__":
    drug_rec_from_pyhealth('retain', 'mimic3', False)
    # 'retain', 'transformer', 'SafeDrug', 'GAMENet', 'micron', 'MoleRec', 'GRASP'
