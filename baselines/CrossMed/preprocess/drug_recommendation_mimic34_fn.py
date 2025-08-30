from itertools import groupby

from OverWrite_data import Patient, Visit
# 此文件所做以下修改
#   1.16行~37行加入了 labevents，并且将itemid和按照时间戳划分
#   2.把只有有一次就诊记录的病人加了进来
#   3.把itemid_groups = []、flag_groups = []加入了样本中
#   4.把 inputevents、microbiology_events


# TODO:我只做了3数据集的药物推荐任务
def drug_recommendation_mimic3_fn(patient: Patient):
    ####
    visits = sorted(patient, key=lambda v: v.HADM_time)
    samples = []
    for visit in visits:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # inj_itemid = visit.get_code_list(table="INPUTEVENTS_MV")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]

        # 获取 labevents
        labevents = visit.get_mimic3_lab_events_with_flags()
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
        inputevents = visit.get_mimic3_input_events_with_amounts()
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
        if len(conditions) * len(procedures) * len(drugs) * len(labevents_sorted)*len(inputevents_sorted) == 0:
            continue
        # 添加个人信息 体重、种族、性别，但感觉这样不利于生成Token，不如注释掉的分别加入
        identity_info = [visit.weight, patient.ethnicity, patient.gender]
        birth_datetime = str(patient.birth_datetime)
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "lab_item": itemid_groups,
                "lab_flag": flag_groups,
                # "inject_amount": amount_groups,
                "inj_amt": originalamount_groups,
                "inj_item": inj_itemid_groups,
                # "interpretation_flag": interpretation_groups,
                # "ab_itemid": ab_itemid_groups,
                "weight": visit.weight,  # 添加体重信息
                "age": age,
                "gender": patient.gender,  # 添加性别信息
                "HADM_time":str(visit.HADM_time),
                "patient.birth_datetime":birth_datetime
            }
        )
    if len(samples) < 1:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["lab_item"] = [samples[0]["lab_item"]]
    samples[0]["lab_flag"] = [samples[0]["lab_flag"]]
    samples[0]["inj_item"] = [samples[0]["inj_item"]]
    # samples[0]["inject_amount"] = [samples[0]["inject_amount"]]
    samples[0]["inj_amt"] = [samples[0]["inj_amt"]]
    # samples[0]["interpretation_flag"] = [samples[0]["interpretation_flag"]]
    # samples[0]["ab_itemid"] = [samples[0]["ab_itemid"]]
    samples[0]["weight"] = [samples[0]["weight"]]
    samples[0]["age"] = [samples[0]["age"]]
    samples[0]["gender"] = [samples[0]["gender"]]
    samples[0]["HADM_time"] = [samples[0]["HADM_time"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]['drugs_hist'] = samples[i - 1]["drugs_hist"] + [samples[i]["drugs_hist"]]
        samples[i]["lab_item"] = samples[i - 1]["lab_item"] + [samples[i]["lab_item"]]
        samples[i]["lab_flag"] = samples[i - 1]["lab_flag"] + [samples[i]["lab_flag"]]
        # samples[i]["inject_amount"] = samples[i - 1]["inject_amount"] + [samples[i]["inject_amount"]]
        samples[i]["inj_item"] = samples[i - 1]["inj_item"] + [samples[i]["inj_item"]]
        samples[i]["inj_amt"] = samples[i - 1]["inj_amt"] + [samples[i]["inj_amt"]]
        # samples[i]["interpretation_flag"] = samples[i - 1]["interpretation_flag"] + [samples[i]["interpretation_flag"]]
        # samples[i]["ab_itemid"] = samples[i - 1]["ab_itemid"] + [samples[i]["ab_itemid"]]
        samples[i]["weight"] = samples[i - 1]["weight"] + [samples[i]["weight"]]
        samples[i]["age"] = samples[i - 1]["age"] + [samples[i]["age"]]
        samples[i]["gender"] = samples[i - 1]["gender"] + [samples[i]["gender"]]
        samples[i]["HADM_time"] = samples[i - 1]["HADM_time"] + [samples[i]["HADM_time"]]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []

    return samples

def drug_recommendation_mimic4_fn(patient: Patient):
    visits = sorted(patient, key=lambda v: v.HADM_time)
    samples = []
    for visit in visits:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # 获取 labevents
        labevents = visit.get_mimic4_lab_events_with_flags()
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
        inputevents = visit.get_mimic4_input_events_with_amounts()
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
        # 添加个人信息 体重、种族、性别，但感觉这样不利于生成Token，不如注释掉的分别加入
        # identity_info = [visit.weight, patient.ethnicity, patient.gender]
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
                "lab_item": itemid_groups,
                "lab_flag": flag_groups,
                "inj_item": inj_itemid_groups,
                # "inject_amount": amount_groups,
                ## 医生的开药量
                "inj_amt": originalamount_groups,
                # "interpretation_flag": interpretation_groups,
                # "ab_itemid": ab_itemid_groups,
                "weight": visit.weight,  # 添加体重信息
                "age": age,
                "gender": patient.gender,  # 添加性别信息
                "HADM_time": str(visit.HADM_time)
            }
        )
    if len(samples) < 1:
        return []
        # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]
    samples[0]["lab_item"] = [samples[0]["lab_item"]]
    samples[0]["lab_flag"] = [samples[0]["lab_flag"]]
    # samples[0]["inject_amount"] = [samples[0]["inject_amount"]]
    samples[0]["inj_item"] = [samples[0]["inj_item"]]
    samples[0]["inj_amt"] = [samples[0]["inj_amt"]]
    # samples[0]["interpretation_flag"] = [samples[0]["interpretation_flag"]]
    # samples[0]["ab_itemid"] = [samples[0]["ab_itemid"]]
    samples[0]["weight"] = [samples[0]["weight"]]
    samples[0]["age"] = [samples[0]["age"]]
    samples[0]["gender"] = [samples[0]["gender"]]
    samples[0]["HADM_time"] = [samples[0]["HADM_time"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [samples[i]["conditions"]]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [samples[i]["procedures"]]
        samples[i]['drugs_hist'] = samples[i - 1]["drugs_hist"] + [samples[i]["drugs_hist"]]
        samples[i]["lab_item"] = samples[i - 1]["lab_item"] + [samples[i]["lab_item"]]
        samples[i]["lab_flag"] = samples[i - 1]["lab_flag"] + [samples[i]["lab_flag"]]
        # samples[i]["inject_amount"] = samples[i - 1]["inject_amount"] + [samples[i]["inject_amount"]]
        samples[i]["inj_item"] = samples[i - 1]["inj_item"] + [samples[i]["inj_item"]]
        samples[i]["inj_amt"] = samples[i - 1]["inj_amt"] + [samples[i]["inj_amt"]]
        # samples[i]["interpretation_flag"] = samples[i - 1]["interpretation_flag"] + [samples[i]["interpretation_flag"]]
        # samples[i]["ab_itemid"] = samples[i - 1]["ab_itemid"] + [samples[i]["ab_itemid"]]
        samples[i]["weight"] = samples[i - 1]["weight"] + [samples[i]["weight"]]
        samples[i]["age"] = samples[i - 1]["age"] + [samples[i]["age"]]
        samples[i]["gender"] = samples[i - 1]["gender"] + [samples[i]["gender"]]
        samples[i]["HADM_time"] = samples[i - 1]["HADM_time"] + [samples[i]["HADM_time"]]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []
    sp = samples
    return samples

def drug_recommendation_eicu_fn(patient: Patient):

    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples


def list_nested_levels(l):
    if not isinstance(l, list):
        return [0]
    if not l:
        return [1]
    return [1 + max(list_nested_levels(i)) for i in l]



def drug_recommendation_omop_fn(patient: Patient):
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples

