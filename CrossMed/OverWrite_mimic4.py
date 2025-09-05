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


class MIMIC4Dataset(BaseEHRDataset):

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patients and admissions tables.

        Will be called in `self.parse_tables()`

        Docs:
            - patients:https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "patients.csv"),
            dtype={"subject_id": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patients and admissions tables
        df = pd.merge(patients_df, admissions_df, on="subject_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # group by patient
        df_group = df.groupby("subject_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            # no exact birth datetime in MIMIC-IV
            # use anchor_year and anchor_age to approximate birth datetime
            anchor_year = int(p_info["anchor_year"].values[0])
            anchor_age = int(p_info["anchor_age"].values[0])
            birth_year = anchor_year - anchor_age
            patient = Patient(
                patient_id=p_id,
                # no exact month, day, and time, use Jan 1st, 00:00:00
                birth_datetime=strptime(str(birth_year)),
                # no exact time, use 00:00:00
                death_datetime=strptime(p_info["dod"].values[0]),
                gender=p_info["gender"].values[0],
                ethnicity=p_info["race"].values[0],
                anchor_year_group=p_info["anchor_year_group"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("hadm_id"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    HADM_time=strptime(v_info["admittime"].values[0]),
                    discharge_time=strptime(v_info["dischtime"].values[0]),
                    discharge_status=v_info["hospital_expire_flag"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.subject_id.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients


    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses diagnosis_icd table.

        Will be called in `self.parse_tables()`

        Docs:
            - diagnosis_icd: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in diagnoses_icd
                table, so we set it to None.
        """
        table = "diagnoses_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            # iterate over each patient and visit
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients


    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses procedures_icd table.

        Will be called in `self.parse_tables()`

        Docs:
            - procedures_icd: https://mimic.mit.edu/docs/iv/modules/hosp/procedures_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in procedures_icd
                table, so we set it to None.
        """
        table = "procedures_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients


    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses prescriptions table.

        Will be called in `self.parse_tables()`

        Docs:
            - prescriptions: https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "prescriptions"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"subject_id": str, "hadm_id": str, "ndc": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "ndc"])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "starttime", "stoptime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["starttime"], v_info["ndc"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients


    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses labevents table.

        Will be called in `self.parse_tables()`

        Docs:
            - labevents: https://mimic.mit.edu/docs/iv/modules/hosp/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "labevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "itemid"])
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of labevent (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code, flag in zip(v_info["charttime"], v_info["itemid"], v_info["flag"]):
                    event = Event(
                        code=code,
                        flag=flag,
                        table=table,
                        vocabulary="MIMIC4_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

        # def parse_hcpcsevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses hcpcsevents table.
    
        Will be called in `self.parse_tables()`
    
        Docs:
            - hcpcsevents: https://mimic.mit.edu/docs/iv/modules/hosp/hcpcsevents/
    
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
    
        Returns:
            The updated patients dict.
    
        Note:
            MIMIC-IV does not provide specific timestamps in hcpcsevents
                table, so we set it to None.
        """

        # table = "hcpcsevents"
        # # read table
        # df = pd.read_csv(
        #     os.path.join(self.root, f"{table}.csv"),
        #     dtype={"subject_id": str, "hadm_id": str, "hcpcs_cd": str},
        # )
        # # drop rows with missing values
        # df = df.dropna(subset=["subject_id", "hadm_id", "hcpcs_cd"])
        # # sort by sequence number (i.e., priority)
        # df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # # group by patient and visit
        # group_df = df.groupby("subject_id")
        #
        # # parallel unit of hcpcsevents (per patient)
        # def hcpcsevents_unit(p_id, p_info):
        #     events = []
        #     for v_id, v_info in p_info.groupby("hadm_id"):
        #         for code in v_info["hcpcs_cd"]:
        #             event = Event(
        #                 code=code,
        #                 table=table,
        #                 vocabulary="MIMIC4_HCPCS_CD",
        #                 visit_id=v_id,
        #                 patient_id=p_id,
        #             )
        #             # update patients
        #             events.append(event)
        #     return events
        #
        # # parallel apply
        # group_df = group_df.parallel_apply(
        #     lambda x: hcpcsevents_unit(x.subject_id.unique()[0], x)
        # )
        #
        # # summarize the results
        # patients = self._add_events_to_patient_dict(patients, group_df)
        #
        # return patients
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
        table = "inputevents_mv"
        self.code_vocs["inputevents"] = "input"
        columns = ["subject_id", "hadm_id", "starttime", "itemid", "amount", "patientweight", "originalamount"]
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            usecols=columns,
            dtype={
                "subject_id": str,
                "hadm_id": str,
                "starttime": str,
                "itemid": str,
                "patientweight": str,
                "originalamount": str,
            },
            on_bad_lines='skip',
        )
        # drop records of the other patients
        df = df[df["subject_id"].isin(patients.keys())]
        # drop rows with missing values in essential columns
        df = df.dropna(subset=["subject_id", "hadm_id", "starttime", "itemid","patientweight", "originalamount"])

        # Convert ORIGINALAMOUNT to float
        df['originalamount'] = df['originalamount'].astype(float)

        # Normalize and categorize the ORIGINALAMOUNT for each ITEMID
        scaler = MinMaxScaler()
        normalization_results = []

        for itemid, group in tqdm(df.groupby('itemid'), desc="Processing itemid"):
            if len(group) > 0:
                amounts = group['originalamount'].values.reshape(-1, 1)
                normalized = scaler.fit_transform(amounts).flatten()
                binned = np.digitize(normalized, np.linspace(0, 1, 11)) - 1  # 分为10个等级
                normalization_results.append(pd.DataFrame({
                    'index': group.index,
                    'originalamount_normalized': binned
                }))

        # Concatenate normalization results and merge back to the original dataframe
        norm_df = pd.concat(normalization_results).set_index('index')
        df['originalamount_normalized'] = norm_df['originalamount_normalized'].astype(str)

        # sort by start time
        df = df.sort_values(["subject_id", "hadm_id", "starttime"], ascending=True)

        # Add patient weight to the corresponding visit
        for p_id, p_info in tqdm(df.groupby("subject_id"), desc="Adding patient weight"):
            patient = patients[p_id]
            for v_id, v_info in p_info.groupby("hadm_id"):
                visit = patient.get_visit_by_id(v_id)
                if visit is not None:
                    # Assuming patient weight is constant during the visit, take the first non-null weight
                    patient_weight = v_info["patientweight"].dropna().iloc[0]
                    visit.weight = patient_weight  # Store the weight in the visit object

        # Group by patient and visit to add events
        def inputevents_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for _, row in v_info.iterrows():
                    event = Event(
                        code=row["itemid"],  # Use ITEMID as the event code
                        table=table,
                        vocabulary=None,  # No specific vocabulary for input events
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(row["starttime"]),
                        additional_info={
                            "amount": row["amount"],
                            "originalamount": row["originalamount_normalized"],  # Use the normalized amount
                            "inj_itemid": row["itemid"]
                        }
                    )
                    events.append(event)
            return events

        # Apply the normalization and processing
        group_df = df.groupby("subject_id").apply(
            lambda x: inputevents_unit(x.subject_id.unique()[0], x)
        )

        # Summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients
if __name__ == "__main__":
    dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents", "inputevents"],
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    dataset.stat()
    dataset.info()
