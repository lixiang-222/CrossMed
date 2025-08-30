from collections import OrderedDict
from datetime import datetime
from typing import Optional, List
# 此文所做修改
# 1.在Event加入了一个参数flag（58行）
# 2.在Visit加入了一个参数 events 限免184 283关于events的相关操作
# 另外在254行写了一个get_lab_events_with_flags方法，目的是返回(event.code, event.flag, event.timestamp)三元组

class Event:
    """Contains information about a single event.

    An event can be anything from a diagnosis to a prescription or a lab test
    that happened in a visit of a patient at a specific time.

    Args:
        code: code of the event. E.g., "428.0" for congestive heart failure.
        table: name of the table where the event is recorded. This corresponds
            to the raw csv file name in the dataset. E.g., "DIAGNOSES_ICD".
        vocabulary: vocabulary of the code. E.g., "ICD9CM" for ICD-9 diagnosis codes.
        visit_id: unique identifier of the visit.
        patient_id: unique identifier of the patient.
        timestamp: timestamp of the event. Default is None.
        **attr: optional attributes to add to the event as key=value pairs.

    Attributes:
        attr_dict: Dict, dictionary of visit attributes. Each key is an attribute
            name and each value is the attribute's value.

    Examples:
        >>> from pyhealth.data import Event
        >>> event = Event(
        ...     code="00069153041",
        ...     table="PRESCRIPTIONS",
        ...     vocabulary="NDC",
        ...     visit_id="v001",
        ...     patient_id="p001",
        ...     dosage="250mg",
        ... )
        >>> event
        Event with NDC code 00069153041 from table PRESCRIPTIONS
        >>> event.attr_dict
        {'dosage': '250mg'}
    """

    def __init__(
        self,
        code: str = None,
        flag: str = None,
        table: str = None,
        vocabulary: str = None,
        visit_id: str = None,
        patient_id: str = None,
        timestamp: Optional[datetime] = None,
        additional_info: Optional[dict] = None,
        interpretation=None, ab_itemid=None,
        **attr,
    ):
        assert timestamp is None or isinstance(
            timestamp, datetime
        ), "timestamp must be a datetime object"
        self.code = code
        self.flag = str(flag)
        self.table = table
        self.vocabulary = vocabulary
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.timestamp = timestamp
        self.attr_dict = dict()
        self.attr_dict.update(attr)
        self.additional_info = additional_info or {}
        self.interpretation = interpretation
        self.ab_itemid = ab_itemid

    def __repr__(self):
        return f"Event with {self.vocabulary} code {self.code} from table {self.table}"

    def __str__(self):
        lines = list()
        lines.append(f"Event from patient {self.patient_id} visit {self.visit_id}:")
        lines.append(f"\t- Code: {self.code}")
        lines.append(f"\t- Table: {self.table}")
        lines.append(f"\t- Vocabulary: {self.vocabulary}")
        lines.append(f"\t- Timestamp: {self.timestamp}")
        for k, v in self.attr_dict.items():
            lines.append(f"\t- {k}: {v}")
        return "\n".join(lines)


class Visit:
    """Contains information about a single visit.

    A visit is a period of time in which a patient is admitted to a hospital or
    a specific department. Each visit is associated with a patient and contains
    a list of different events.

    Args:
        visit_id: unique identifier of the visit.
        patient_id: unique identifier of the patient.
        HADM_time: timestamp of visit's encounter. Default is None.
        discharge_time: timestamp of visit's discharge. Default is None.
        discharge_status: patient's status upon discharge. Default is None.
        **attr: optional attributes to add to the visit as key=value pairs.

    Attributes:
        attr_dict: Dict, dictionary of visit attributes. Each key is an attribute
            name and each value is the attribute's value.
        event_list_dict: Dict[str, List[Event]], dictionary of event lists.
            Each key is a table name and each value is a list of events from that
            table ordered by timestamp.

    Examples:
        >>> from pyhealth.data import Event, Visit
        >>> event = Event(
        ...     code="00069153041",
        ...     table="PRESCRIPTIONS",
        ...     vocabulary="NDC",
        ...     visit_id="v001",
        ...     patient_id="p001",
        ...     dosage="250mg",
        ... )
        >>> visit = Visit(
        ...     visit_id="v001",
        ...     patient_id="p001",
        ... )
        >>> visit.add_event(event)
        >>> visit
        Visit v001 from patient p001 with 1 events from tables ['PRESCRIPTIONS']
        >>> visit.available_tables
        ['PRESCRIPTIONS']
        >>> visit.num_events
        1
        >>> visit.get_event_list('PRESCRIPTIONS')
        [Event with NDC code 00069153041 from table PRESCRIPTIONS]
        >>> visit.get_code_list('PRESCRIPTIONS')
        ['00069153041']
        >>> patient.available_tables
        ['PRESCRIPTIONS']
        >>> patient.get_visit_by_index(0)
        Visit v001 from patient p001 with 1 events from tables ['PRESCRIPTIONS']
        >>> patient.get_visit_by_index(0).get_code_list(table="PRESCRIPTIONS")
        ['00069153041']
    """

    def __init__(
        self,
        visit_id: str,
        patient_id: str,
        HADM_time: Optional[datetime] = None,
        discharge_time: Optional[datetime] = None,
        discharge_status=None,
        **attr,
    ):
        assert HADM_time is None or isinstance(
            HADM_time, datetime
        ), "HADM_time must be a datetime object"
        assert discharge_time is None or isinstance(
            discharge_time, datetime
        ), "discharge_time must be a datetime object"
        self.visit_id = visit_id
        self.patient_id = patient_id
        self.HADM_time = HADM_time
        self.discharge_time = discharge_time
        self.discharge_status = discharge_status
        self.attr_dict = dict()
        self.attr_dict.update(attr)
        self.event_list_dict = dict()
        self.events = []
        self.weight = None

    def add_event(self, event: Event) -> None:
        """Adds an event to the visit.

        If the event's table is not in the visit's event list dictionary, it is
        added as a new key. The event is then added to the list of events of
        that table.

        Args:
            event: event to add.

        Note:
            As for now, there is no check on the order of the events. The new event
                is simply appended to end of the list.
        """
        assert event.visit_id == self.visit_id, "visit_id unmatched"
        assert event.patient_id == self.patient_id, "patient_id unmatched"
        table = event.table
        if table not in self.event_list_dict:
            self.event_list_dict[table] = list()
        self.event_list_dict[table].append(event)
        self.events.append(event)
        # 检查事件是否来自 INPUTEVENTS_MV 表，并且包含体重信息
        if table == "inputevents" and hasattr(event, "weight"):
            self.weight = event.weight
        if table == "INPUTEVENTS_MV" and hasattr(event, "weight"):
            self.weight = event.weight
    # def add_event_lab(self, event: Event_lab) -> None:
    #     """Adds an event to the visit.
    #
    #     If the event's table is not in the visit's event list dictionary, it is
    #     added as a new key. The event is then added to the list of events of
    #     that table.
    #
    #     Args:
    #         event: event to add.
    #
    #     Note:
    #         As for now, there is no check on the order of the events. The new event
    #             is simply appended to end of the list.
    #     """
    #     assert event.visit_id == self.visit_id, "visit_id unmatched"
    #     assert event.patient_id == self.patient_id, "patient_id unmatched"
    #     table = event.table
    #     if table not in self.event_list_dict:
    #         self.event_list_dict[table] = list()
    #     self.event_list_dict[table].append(event)
    #     self.events.append(event)

    def get_event_list(self, table: str) -> List[Event]:
        """Returns a list of events from a specific table.

        If the table is not in the visit's event list dictionary, an empty list
        is returned.

        Args:
            table: name of the table.

        Returns:
           List of events from the specified table.

        Note:
            As for now, there is no check on the order of the events. The list of
                events is simply returned as is.
        """
        if table in self.event_list_dict:
            return self.event_list_dict[table]
        else:
            return list()

    def get_code_list(
        self, table: str, remove_duplicate: Optional[bool] = True
    ) -> List[str]:
        """Returns a list of codes from a specific table.

        If the table is not in the visit's event list dictionary, an empty list
        is returned.

        Args:
            table: name of the table.
            remove_duplicate: whether to remove duplicate codes
                (but keep the relative order). Default is True.

        Returns:
            List of codes from the specified table.

        Note:
            As for now, there is no check on the order of the codes. The list of
                codes is simply returned as is.
        """
        event_list = self.get_event_list(table)
        code_list = [event.code for event in event_list]
        if remove_duplicate:
            # remove duplicate codes but keep the order
            code_list = list(dict.fromkeys(code_list))
        return code_list

    def get_mimic3_lab_events_with_flags(self):
        return [(event.code, event.flag, event.timestamp) for event in self.events if event.table == "LABEVENTS" ]
    def get_mimic4_lab_events_with_flags(self):
        return [(event.code, event.flag, event.timestamp) for event in self.events if event.table == "labevents" ]
    def set_event_list(self, table: str, event_list: List[Event]) -> None:
        """Sets the list of events from a specific table.

        This function will overwrite any existing list of events from
        the specified table.

        Args:
            table: name of the table.
            event_list: list of events to set.

        Note:
            As for now, there is no check on the order of the events. The list of
                events is simply set as is.
        """
        self.event_list_dict[table] = event_list
        self.events.extend(event_list)

    def get_mimic3_input_events_with_amounts(self):
        """
        Returns a list of tuples containing the input events with amounts.
        Each tuple contains (amount, originalamount, timestamp).

        Returns:
            List[Tuple[float, float, datetime]]: A list of input events with amounts.
        """
        table = "INPUTEVENTS_MV"
        if table not in self.event_list_dict:
                return []

        events = []
        for event in self.event_list_dict[table]:
            amount = event.additional_info.get("amount", None)
            originalamount = event.additional_info.get("originalamount", None)
            timestamp = event.timestamp
            inj_itemid = event.additional_info.get("inj_itemid", None)

            if amount is not None and originalamount is not None and timestamp is not None and inj_itemid is not None:
                events.append((amount, originalamount, timestamp, inj_itemid ))

        return events
    def get_mimic4_input_events_with_amounts(self):
        """
        Returns a list of tuples containing the input events with amounts.
        Each tuple contains (amount, originalamount, timestamp).

        Returns:
            List[Tuple[float, float, datetime]]: A list of input events with amounts.
        """
        table = "inputevents_mv"
        if table not in self.event_list_dict:
                return []

        events = []
        for event in self.event_list_dict[table]:
            amount = event.additional_info.get("amount", None)
            originalamount = event.additional_info.get("originalamount", None)
            timestamp = event.timestamp
            inj_itemid = event.additional_info.get("inj_itemid", None)

            if amount is not None and originalamount is not None and timestamp is not None and inj_itemid is not None:
                events.append((amount, originalamount, timestamp, inj_itemid ))

        return events
    @property
    def available_tables(self) -> List[str]:
        """Returns a list of available tables for the visit.

        Returns:
            List of available tables.
        """
        return list(self.event_list_dict.keys())

    @property
    def num_events(self) -> int:
        """Returns the total number of events in the visit.

        Returns:
            Total number of events.
        """
        return sum([len(event_list) for event_list in self.event_list_dict.values()])

    def __repr__(self):
        return (
            f"Visit {self.visit_id} "
            f"from patient {self.patient_id} "
            f"with {self.num_events} events "
            f"from tables {self.available_tables}"
        )

    def __str__(self):
        lines = list()
        lines.append(
            f"Visit {self.visit_id} from patient {self.patient_id} "
            f"with {self.num_events} events:"
        )
        lines.append(f"\t- Encounter time: {self.HADM_time}")
        lines.append(f"\t- Discharge time: {self.discharge_time}")
        lines.append(f"\t- Discharge status: {self.discharge_status}")
        lines.append(f"\t- Available tables: {self.available_tables}")
        for k, v in self.attr_dict.items():
            lines.append(f"\t- {k}: {v}")
        for table, event_list in self.event_list_dict.items():
            for event in event_list:
                event_str = str(event).replace("\n", "\n\t")
                lines.append(f"\t- {event_str}")
        return "\n".join(lines)


class Patient:
    """Contains information about a single patient.

    A patient is a person who is admitted at least once to a hospital or
    a specific department. Each patient is associated with a list of visits.

    Args:
        patient_id: unique identifier of the patient.
        birth_datetime: timestamp of patient's birth. Default is None.
        death_datetime: timestamp of patient's death. Default is None.
        gender: gender of the patient. Default is None.
        ethnicity: ethnicity of the patient. Default is None.
        **attr: optional attributes to add to the patient as key=value pairs.

    Attributes:
        attr_dict: Dict, dictionary of patient attributes. Each key is an attribute
            name and each value is the attribute's value.
        visits: OrderedDict[str, Visit], an ordered dictionary of visits. Each key
            is a visit_id and each value is a visit.
        index_to_visit_id: Dict[int, str], dictionary that maps the index of a visit
            in the visits list to the corresponding visit_id.

    Examples:
            >>> from pyhealth.data import Event, Visit, Patient
            >>> event = Event(
            ...     code="00069153041",
            ...     table="PRESCRIPTIONS",
            ...     vocabulary="NDC",
            ...     visit_id="v001",
            ...     patient_id="p001",
            ...     dosage="250mg",
            ... )
            >>> visit = Visit(
            ...     visit_id="v001",
            ...     patient_id="p001",
            ... )
            >>> visit.add_event(event)
            >>> patient = Patient(
            ...     patient_id="p001",
            ... )
            >>> patient.add_visit(visit)
            >>> patient
            Patient p001 with 1 visits
    """

    def __init__(
        self,
        patient_id: str,
        birth_datetime: Optional[datetime] = None,
        death_datetime: Optional[datetime] = None,
        gender=None,
        ethnicity=None,
        **attr,
    ):
        self.patient_id = patient_id
        self.birth_datetime = birth_datetime
        self.death_datetime = death_datetime
        self.gender = gender
        self.ethnicity = ethnicity
        self.attr_dict = dict()
        self.attr_dict.update(attr)
        self.visits = OrderedDict()
        self.index_to_visit_id = dict()

    def add_visit(self, visit: Visit) -> None:
        """Adds a visit to the patient.

        If the visit's visit_id is already in the patient's visits dictionary,
        it will be overwritten by the new visit.

        Args:
            visit: visit to add.

        Note:
            As for now, there is no check on the order of the visits. The new visit
                is simply added to the end of the ordered dictionary of visits.
        """
        assert visit.patient_id == self.patient_id, "patient_id unmatched"
        self.visits[visit.visit_id] = visit
        # incrementing index
        self.index_to_visit_id[len(self.visits) - 1] = visit.visit_id

    def add_event(self, event: Event) -> None:
        """Adds an event to the patient.

        If the event's visit_id is not in the patient's visits dictionary, this
        function will raise KeyError.

        Args:
            event: event to add.

        Note:
            As for now, there is no check on the order of the events. The new event
                is simply appended to the end of the list of events of the
                corresponding visit.
        """
        assert event.patient_id == self.patient_id, "patient_id unmatched"
        visit_id = event.visit_id
        if visit_id not in self.visits:
            raise KeyError(
                f"Visit with id {visit_id} not found in patient {self.patient_id}"
            )
        self.get_visit_by_id(visit_id).add_event(event)

    def get_visit_by_id(self, visit_id: str) -> Visit:
        """Returns a visit by visit_id.

        Args:
            visit_id: unique identifier of the visit.

        Returns:
            Visit with the given visit_id.
        """
        return self.visits[visit_id]

    def get_visit_by_index(self, index: int) -> Visit:
        """Returns a visit by its index.

        Args:
            index: int, index of the visit to return.

        Returns:
            Visit with the given index.
        """
        if index not in self.index_to_visit_id:
            raise IndexError(
                f"Visit with  index {index} not found in patient {self.patient_id}"
            )
        visit_id = self.index_to_visit_id[index]
        return self.get_visit_by_id(visit_id)


    @property
    def available_tables(self) -> List[str]:
        """Returns a list of available tables for the patient.

        Returns:
            List of available tables.
        """
        tables = []
        for visit in self:
            tables.extend(visit.available_tables)
        return list(set(tables))

    def __len__(self):
        """Returns the number of visits in the patient."""
        return len(self.visits)

    def __getitem__(self, index) -> Visit:
        """Returns a visit by its index."""
        return self.get_visit_by_index(index)

    def __repr__(self):
        return f"Patient {self.patient_id} with {len(self)} visits"

    def __str__(self):
        lines = list()
        # patient info
        lines.append(f"Patient {self.patient_id} with {len(self)} visits:")
        lines.append(f"\t- Birth datetime: {self.birth_datetime}")
        lines.append(f"\t- Death datetime: {self.death_datetime}")
        lines.append(f"\t- Gender: {self.gender}")
        lines.append(f"\t- Ethnicity: {self.ethnicity}")
        for k, v in self.attr_dict.items():
            lines.append(f"\t- {k}: {v}")
        # visit info
        for visit in self:
            visit_str = str(visit).replace("\n", "\n\t")
            lines.append(f"\t- {visit_str}")
        return "\n".join(lines)