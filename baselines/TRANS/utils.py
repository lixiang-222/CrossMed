import random
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.medcode import InnerMap
from pyhealth.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Subset


def mm_dataloader(trainset, validset, testset, batch_size=64):
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader


def custom_collate_fn(batch):
    sequence_data_list = [item[0] for item in batch]
    graph_data_list = [item[1] for item in batch]

    sequence_data_batch = {key: [d[key] for d in sequence_data_list if d[key] != []] for key in sequence_data_list[0]}

    graph_data_batch = graph_data_list

    return sequence_data_batch, graph_data_batch


def convert_to_relative_time(datetime_strings):
    datetimes = parse_datetimes(datetime_strings)
    base_time = min(datetimes)
    return [timedelta_to_str(dt - base_time) for dt in datetimes]


def parse_datetimes(datetime_strings):
    print(datetime_strings)
    return [datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S") for dt_str in datetime_strings]


def timedelta_to_str(tdelta):
    days = tdelta.days
    seconds = tdelta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return days * 1440 + hours * 60 + minutes
