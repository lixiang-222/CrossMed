from itertools import groupby

from OverWrite_data import Patient, Visit
from torch.utils.data import Dataset
# from data.GraphConstruction import *
from tqdm import *
from pyhealth.medcode import CrossMap

mapping = CrossMap("ICD10CM", "CCSCM")
mapping3 = CrossMap("ICD9CM", "CCSCM")


def diag_prediction_mimic3_fn(patient: Patient):
    visits = sorted(patient, key=lambda v: v.HADM_time)
    samples = []
    for visit in visits:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # inj_itemid = visit.get_code_list(table="INPUTEVENTS_MV")

        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # 这里有个映射，不过我们之前映射过了
        # cond_ccs = []
        # for con in conditions:
        #     if mapping3.map(con):
        #         cond_ccs.append(mapping3.map(con)[0])

        # 获取 labevents
        labevents = visit.get_lab_events_with_flags()
        labevents_sorted = sorted(labevents, key=lambda x: x[2])
        itemid_groups = []
        flag_groups = []
        for key, group in groupby(labevents_sorted, key=lambda x: x[2].date()):
            group_list = list(group)
            itemid_group = [event[0] for event in group_list]
            flag_group = [event[1] for event in group_list]
            itemid_groups.append(itemid_group)
            flag_groups.append(flag_group)

        # 获取 inputevents
        inputevents = visit.get_input_events_with_amounts()
        inputevents_sorted = sorted(inputevents, key=lambda x: x[2])
        originalamount_groups = []
        inj_itemid_groups = []
        for key, group in groupby(inputevents_sorted, key=lambda x: x[2].date()):
            group_list = list(group)
            # amount_group = [event[0] for event in group_list]
            originalamount_group = [event[1] for event in group_list]
            inj_itemid_group = [event[3] for event in group_list]
            # amount_groups.append(amount_group)
            originalamount_groups.append(originalamount_group)
            inj_itemid_groups.append(inj_itemid_group)
        # # 获取 microbiology
        # microbiology_events = visit.get_microbiology_events()
        # interpretation_groups = []
        # ab_itemid_groups = []
        # for event in microbiology_events:
        #     interpretation_groups.append(event.interpretation)
        #     ab_itemid_groups.append(event.ab_itemid)

        age = str((visit.HADM_time - patient.birth_datetime).days // 365)

        # exclude: visits without condition, procedure, drug code, or lab events
        if len(conditions) * len(procedures) * len(drugs) * len(labevents_sorted) * len(inputevents_sorted) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "cond_hist": conditions,
                "lab_item": itemid_groups,
                "lab_flag": flag_groups,
                "inj_item": inj_itemid_groups,
                "inj_amt": originalamount_groups,
                "weight": visit.weight,
                "HADM_time": str(visit.HADM_time),
                "gender": patient.gender,
                "age": age
            }
        )

    if len(samples) < 1:
        return []

    # add history
    samples[0]["cond_hist"] = [samples[0]["cond_hist"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["lab_item"] = [samples[0]["lab_item"]]
    samples[0]["lab_flag"] = [samples[0]["lab_flag"]]
    samples[0]["inj_item"] = [samples[0]["inj_item"]]
    samples[0]["inj_amt"] = [samples[0]["inj_amt"]]
    samples[0]["weight"] = [samples[0]["weight"]]
    samples[0]["age"] = [samples[0]["age"]]
    samples[0]["gender"] = [samples[0]["gender"]]
    samples[0]["HADM_time"] = [samples[0]["HADM_time"]]

    for i in range(1, len(samples)):
        samples[i]["cond_hist"] = samples[i - 1]["cond_hist"] + [samples[i]["cond_hist"]]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]['drugs'] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
        samples[i]["lab_item"] = samples[i - 1]["lab_item"] + [samples[i]["lab_item"]]
        samples[i]["lab_flag"] = samples[i - 1]["lab_flag"] + [samples[i]["lab_flag"]]
        samples[i]["inj_item"] = samples[i - 1]["inj_item"] + [samples[i]["inj_item"]]
        samples[i]["inj_amt"] = samples[i - 1]["inj_amt"] + [
            samples[i]["inj_amt"]]
        samples[i]["weight"] = samples[i - 1]["weight"] + [samples[i]["weight"]]
        samples[i]["age"] = samples[i - 1]["age"] + [samples[i]["age"]]
        samples[i]["gender"] = samples[i - 1]["gender"] + [samples[i]["gender"]]
        samples[i]["HADM_time"] = samples[i - 1]["HADM_time"] + [samples[i]["HADM_time"]]

    # 掩盖最近一次的诊断（疾病）
    for i in range(len(samples)):
        samples[i]["cond_hist"][i] = []

    return samples


def diag_prediction_mimic4_fn(patient: Patient):
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # cond_ccs = []
        # for con in conditions:
        #     if mapping.map(con):
        #         cond_ccs.append(mapping.map(con)[0])

        # 获取 labevents
        labevents = visit.get_lab_events_with_flags()
        labevents_sorted = sorted(labevents, key=lambda x: x[2])
        itemid_groups = []
        flag_groups = []
        for key, group in groupby(labevents_sorted, key=lambda x: x[2].date()):
            group_list = list(group)
            itemid_group = [event[0] for event in group_list]
            flag_group = [event[1] for event in group_list]
            itemid_groups.append(itemid_group)
            flag_groups.append(flag_group)

        # 获取 inputevents
        inputevents = visit.get_input_events_with_amounts()
        inputevents_sorted = sorted(inputevents, key=lambda x: x[2])
        originalamount_groups = []
        inj_itemid_groups = []
        for key, group in groupby(inputevents_sorted, key=lambda x: x[2].date()):
            group_list = list(group)
            # amount_group = [event[0] for event in group_list]
            originalamount_group = [event[1] for event in group_list]
            inj_itemid_group = [event[3] for event in group_list]
            # amount_groups.append(amount_group)
            originalamount_groups.append(originalamount_group)
            inj_itemid_groups.append(inj_itemid_group)

        # # 获取 microbiology
        # microbiology_events = visit.get_microbiology_events()
        # interpretation_groups = []
        # ab_itemid_groups = []
        # for event in microbiology_events:
        #     interpretation_groups.append(event.interpretation)
        #     ab_itemid_groups.append(event.ab_itemid)

        age = str((visit.HADM_time - patient.birth_datetime).days // 365)

        # exclude: visits without condition, procedure, drug code, or lab events
        if len(conditions) * len(procedures) * len(drugs) * len(labevents_sorted) * len(inputevents_sorted) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "cond_hist": conditions,
                "lab_item": itemid_groups,
                "lab_flag": flag_groups,
                "inj_item": inj_itemid_groups,
                # 医生的开药量
                "inj_amt": originalamount_groups,
                "weight": visit.weight,
                "HADM_time": str(visit.HADM_time),
                "gender": patient.gender,
                "age": age
            }
        )

    if len(samples) < 1:
        return []

    # add history
    samples[0]["cond_hist"] = [samples[0]["cond_hist"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs"] = [samples[0]["drugs"]]
    samples[0]["lab_item"] = [samples[0]["lab_item"]]
    samples[0]["lab_flag"] = [samples[0]["lab_flag"]]
    samples[0]["inj_item"] = [samples[0]["inj_item"]]
    samples[0]["inj_amt"] = [samples[0]["inj_amt"]]
    samples[0]["weight"] = [samples[0]["weight"]]
    samples[0]["age"] = [samples[0]["age"]]
    samples[0]["gender"] = [samples[0]["gender"]]
    samples[0]["HADM_time"] = [samples[0]["HADM_time"]]

    for i in range(1, len(samples)):
        samples[i]["cond_hist"] = samples[i - 1]["cond_hist"] + [samples[i]["cond_hist"]]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]['drugs'] = samples[i - 1]["drugs"] + [samples[i]["drugs"]]
        samples[i]["lab_item"] = samples[i - 1]["lab_item"] + [samples[i]["lab_item"]]
        samples[i]["lab_flag"] = samples[i - 1]["lab_flag"] + [samples[i]["lab_flag"]]
        samples[i]["inj_item"] = samples[i - 1]["inj_item"] + [samples[i]["inj_item"]]
        samples[i]["inj_amt"] = samples[i - 1]["inj_amt"] + [
            samples[i]["inj_amt"]]
        samples[i]["weight"] = samples[i - 1]["weight"] + [samples[i]["weight"]]
        samples[i]["age"] = samples[i - 1]["age"] + [samples[i]["age"]]
        samples[i]["gender"] = samples[i - 1]["gender"] + [samples[i]["gender"]]
        samples[i]["HADM_time"] = samples[i - 1]["HADM_time"] + [samples[i]["HADM_time"]]

    # 掩盖最近一次的诊断（疾病）
    for i in range(len(samples)):
        samples[i]["cond_hist"][i] = []

    return samples

# class MMDataset(Dataset):
#     def __init__(self, dataset, tokenizer, dim, device, trans_dim=0, di=False):
#         self.sequence_dataset = dataset.samples
#         self.tokenizer = tokenizer
#         self.trans_dim = trans_dim
#         self.di = di
#         self.dim = dim
#         self.device = device
#         self.graph_data = PatientGraph(self.tokenizer, self.sequence_dataset, dim=self.dim, device = self.device, trans_dim=self.trans_dim, di=self.di).all_data
#
#     def __len__(self):
#         return len(self.sequence_dataset)
#
#     def __getitem__(self, idx):
#         sequence_data = self.sequence_dataset[idx]
#         graph_data = self.graph_data[idx]
#         return sequence_data, graph_data
