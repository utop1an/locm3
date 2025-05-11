from .ocm import OCM, TypedObject, Event, SingletonEvent, StatePointers
from typing import Dict, List, Tuple, Set, NamedTuple
from traces import Hypothesis, HIndex, HItem, FSM, PartialOrderedTrace
from pddl import ActionSignature, LearnedModel, LearnedLiftedFluent, LearnedAction
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from utils.helpers import pprint_table, check_well_formed, check_valid

class POLOCM2(LOCM2):

    def __init__(self, solver_path="default", cores= 1,state_param:bool=True, viz=False, timeout:int = 600, debug: Dict[str, bool]=None):
        super().__init__(state_param=state_param, viz=viz, timeout=timeout, debug=debug)
        self.solver_path = solver_path
        self.cores = cores

    def extract_model(self, trace_list: List[PartialOrderedTrace], type_dict: Dict[str, int] = None) -> LearnedModel:

        sorts = self._get_sorts(trace_list, type_dict)
        obj_traces_list, TM_list = self.solve_po()

        transition_sets_per_sort_list = super.split_transitions(TM_list, obj_traces_list, sorts)
        TS, OS, ap_state_pointers = super.get_TS_OS(obj_traces_list, transition_sets_per_sort_list, TM_list, sorts)
        if self.state_param:
            bindings = super.get_state_bindings(TS, ap_state_pointers, OS, sorts, TM_list, debug=self.debug)
        else:
            bindings = None
        model = super.get_model(OS, ap_state_pointers, sorts, bindings, None, statics=[], debug=False)
        return model
        

    def solve_po(self, trace_list, sorts):
        _ = self.get_matrices(trace_list, sorts)
        _ = self.build_PO_constraints()
        _ = self.build_FO_constraints()
        _ = self.build_TM_constraints()
        _ = self.bip_solve_PO()
        obj_traces_list, TM_list = self.get_obj_traces()

    def get_matrices(self, trace_list, sorts):
        pass

    def build_PO_constraints(self):
        pass

    def build_FO_constraints(self):
        pass

    def build_TM_constraints(self):
        pass

    def bip_solve_PO(self):
        pass

    def get_obj_traces(self):
        pass



    
            