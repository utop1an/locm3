from .ocm import  TypedObject, Event, SingletonEvent, StatePointers
from .locm2 import LOCM2
from .po_baseline import POBASELINE
from typing import Dict, List, Tuple, Set, NamedTuple
from traces import Hypothesis, HIndex, HItem, FSM, PartialOrderedTrace, IndexedEvent
from pddl import ActionSignature, LearnedModel, LearnedLiftedFluent, LearnedAction
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
from utils.helpers import pprint_table
import time

class POLOCM2BASELINE(POBASELINE, LOCM2):

    def __init__(self,solver_path="default", cores= 1,state_param:bool=True, viz=False, timeout:int = 600, debug: Dict[str, bool]=None):
        super().__init__(state_param=state_param, viz=viz, timeout=timeout, debug=debug)
        self.solver_path = solver_path
        self.cores = cores

    def extract_model(self, po_trace_list: List[PartialOrderedTrace], type_dict: Dict[str, int] = None) -> LearnedModel:
        start = time.time()
        sorts, sort_to_type_dict = self._get_sorts(po_trace_list, type_dict)
        obj_consecutive_transitions_list, TM_list = self.solve_po(po_trace_list, sorts)
        po_time = time.time() - start
        transition_sets_per_sort_list = self.split_transitions(TM_list, obj_consecutive_transitions_list, sorts)
        locm2_time = time.time() - start - po_time
        TS, OS, event_state_pointers = self.get_TS_OS(obj_consecutive_transitions_list, transition_sets_per_sort_list, TM_list, sorts)
        if self.state_param:
            bindings = self.get_state_bindings(TS, event_state_pointers, OS, sorts, TM_list, debug=self.debug)
        else:
            bindings = None
        model = self.get_model(OS, event_state_pointers, sorts, bindings, None, statics=[], debug=False)
        locm_time = time.time() - start - po_time - locm2_time
        return model, TM_list, (po_time, locm2_time, locm_time)
        

    
    
    


    