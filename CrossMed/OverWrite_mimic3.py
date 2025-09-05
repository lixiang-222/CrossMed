import os
from tqdm import tqdm
import pandas as pd

# from pyhealth.data import Event, Visit, Patient
from OverWrite_data import Event, Visit, Patient
from OverWrite_base_ehr_dataset import BaseEHRDataset
from pyhealth.datasets.utils import strptime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from typing import Dict


# TODO: add other tables
# 本文件修改如下parse_labevents的函数（316行）加入了flag参数（353）
# 加入INPUTEVENTS_MV文件 把此文件的ORINALAMOUNT AMOUNT STARTTIME WEIGHT引入患者表示中

class MIMIC3Dataset(BaseEHRDataset):
    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            # encoding='utf-8',
            on_bad_lines='skip',
            # engine="python",
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admissions_df, on="SUBJECT_ID", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        # group by patient
        df_group = df.groupby("SUBJECT_ID")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["DOB"].values[0]),
                death_datetime=strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("HADM_ID"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    HADM_time=strptime(v_info["ADMITTIME"].values[0]),
                    discharge_time=strptime(v_info["DISCHTIME"].values[0]),
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                    insurance=v_info["INSURANCE"].values[0],
                    language=v_info["LANGUAGE"].values[0],
                    religion=v_info["RELIGION"].values[0],
                    marital_status=v_info["MARITAL_STATUS"].values[0],
                    ethnicity=v_info["ETHNICITY"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.SUBJECT_ID.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "DIAGNOSES_ICD"
        self.code_vocs["conditions"] = "ICD9CM"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
            on_bad_lines='skip',
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "PROCEDURES_ICD"
        self.code_vocs["procedures"] = "ICD9PROC"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
            on_bad_lines='skip',

        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.

        Will be called in `self.parse_tables()`

        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "PRESCRIPTIONS"
        self.code_vocs["drugs"] = "NDC"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
            on_bad_lines='skip',
            low_memory=False,
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "NDC"])
        # sort by start date and end date
        df = df.sort_values(
            ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["STARTDATE"], v_info["NDC"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        table = "LABEVENTS"
        self.code_vocs["labs"] = "MIMIC3_ITEMID"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ITEMID": str, "FLAG": str},
            on_bad_lines='skip',
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ITEMID"])
        # sort by charttime
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for lab (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code, flag in zip(v_info["CHARTTIME"], v_info["ITEMID"], v_info["FLAG"]):
                    event = Event(
                        code=code,
                        flag=flag,
                        table=table,
                        vocabulary="MIMIC3_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_inputevents_mv(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses INPUTEVENTS_MV table.

        Will be called in self.parse_tables()

        Docs:
            - INPUTEVENTS_MV: https://mimic.mit.edu/docs/iii/tables/inputevents_mv/

        Args:
            patients: a dict of Patient objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "INPUTEVENTS_MV"
        self.code_vocs["inputevents"] = "INPUT"
        columns = ["SUBJECT_ID", "HADM_ID", "STARTTIME", "ITEMID", "AMOUNT", "PATIENTWEIGHT", "ORIGINALAMOUNT"]
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            usecols=columns,
            dtype={
                "SUBJECT_ID": str,
                "HADM_ID": str,
                "STARTTIME": str,
                "ITEMID": str,
                "PATIENTWEIGHT": str,
                "ORIGINALAMOUNT": str,
            },
            on_bad_lines='skip',
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values in essential columns
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "STARTTIME", "PATIENTWEIGHT", "ORIGINALAMOUNT"])

        # Convert ORIGINALAMOUNT to float
        df['ORIGINALAMOUNT'] = df['ORIGINALAMOUNT'].astype(float)

        # Normalize and categorize the ORIGINALAMOUNT for each ITEMID
        scaler = MinMaxScaler()
        normalization_results = []

        for itemid, group in tqdm(df.groupby('ITEMID'), desc="Processing ITEMID"):
            if len(group) > 0:
                amounts = group['ORIGINALAMOUNT'].values.reshape(-1, 1)
                normalized = scaler.fit_transform(amounts).flatten()
                binned = np.digitize(normalized, np.linspace(0, 1, 11)) - 1  # 分为10个等级
                normalization_results.append(pd.DataFrame({
                    'index': group.index,
                    'ORIGINALAMOUNT_NORMALIZED': binned
                }))

        # Concatenate normalization results and merge back to the original dataframe
        norm_df = pd.concat(normalization_results).set_index('index')
        df['ORIGINALAMOUNT_NORMALIZED'] = norm_df['ORIGINALAMOUNT_NORMALIZED'].astype(str)

        # sort by start time
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "STARTTIME"], ascending=True)

        # Add patient weight to the corresponding visit
        for p_id, p_info in tqdm(df.groupby("SUBJECT_ID"), desc="Adding patient weight"):
            patient = patients[p_id]
            for v_id, v_info in p_info.groupby("HADM_ID"):
                visit = patient.get_visit_by_id(v_id)
                if visit is not None:
                    # Assuming patient weight is constant during the visit, take the first non-null weight
                    patient_weight = v_info["PATIENTWEIGHT"].dropna().iloc[0]
                    visit.weight = patient_weight  # Store the weight in the visit object

        # Group by patient and visit to add events
        def inputevents_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for _, row in v_info.iterrows():
                    event = Event(
                        code=row["ITEMID"],  # Use ITEMID as the event code
                        table=table,
                        vocabulary=None,  # No specific vocabulary for input events
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(row["STARTTIME"]),
                        additional_info={
                            "amount": row["AMOUNT"],
                            "originalamount": row["ORIGINALAMOUNT_NORMALIZED"],  # Use the normalized amount
                            "inj_itemid": row["ITEMID"]
                        }
                    )
                    events.append(event)
            return events

        # Apply the normalization and processing
        group_df = df.groupby("SUBJECT_ID").apply(
            lambda x: inputevents_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # Summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_microbiologyevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """
        Helper function which parses MICROBIOLOGYEVENTS table.

        Will be called in `self.parse_tables()`

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "MICROBIOLOGYEVENTS"
        self.code_vocs["microbiologyevents"] = "MICROBIOLOGYEVENTS"

        # Read the MICROBIOLOGYEVENTS table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "CHARTDATE": str, "AB_ITEMID": str, "INTERPRETATION": str}
        )

        # Drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]

        # Drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "CHARTDATE", "AB_ITEMID", "INTERPRETATION"])

        # Sort by chartdate
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTDATE"], ascending=True)

        # Group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # Parallel unit for microbiology events (per patient)
        def microbiology_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for _, row in v_info.iterrows():
                    event = Event(
                        table=table,
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(row["CHARTDATE"]),
                        interpretation=row["INTERPRETATION"],
                        ab_itemid=row["AB_ITEMID"]
                    )
                    events.append(event)
            return events

        # Parallel apply
        group_df = group_df.parallel_apply(
            lambda x: microbiology_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # Summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients


if __name__ == "__main__":
    dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/mimiciii-demo/1.4/",
        tables=[
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "PRESCRIPTIONS",
            "LABEVENTS",
        ],
        code_mapping={"NDC": "ATC"},
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()

    # dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimic3/1.4",
    #     tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    #     refresh_cache=False,
    # )
    # print(dataset.stat())
    # print(dataset.available_tables)
    # print(list(dataset.patients.values())[4])
