from .locm import LOCM
from .po_optimisation import POOPTIMISATION
from typing import Dict, List
from traces import  PartialOrderedTrace
from pddl import LearnedModel
import time

class POLOCM(POOPTIMISATION, LOCM):

    def __init__(self, solver_path="default", cores= 1,state_param:bool=True, viz=False, timeout:int = 600, debug: Dict[str, bool]=None):
        super().__init__(state_param=state_param, viz=viz, timeout=timeout, debug=debug)
        self.solver_path = solver_path
        self.cores = cores


    def extract_model(self, PO_trace_list: List[PartialOrderedTrace], type_dict: Dict[str, int] = None) -> LearnedModel:
        start = time.time()
        sorts, sort_to_type_dict = self._get_sorts(PO_trace_list, type_dict)
        obj_traces_list, TM_list = self.solve_po(PO_trace_list, sorts)
        po_time = time.time() - start
        TS, OS, ap_state_pointers = self.get_TS_OS(obj_traces_list, sorts)
        if self.state_param:
            bindings = self.get_state_bindings(TS, ap_state_pointers, OS, sorts, debug=self.debug)
        else:
            bindings = None
        model = self.get_model(OS, ap_state_pointers, sorts, bindings, None, statics=[], debug=False)
        locm_time = time.time() - start - po_time
        return model, TM_list, (po_time, 0 , locm_time)
        

    