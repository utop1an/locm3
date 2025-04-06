from .ocm import OCM
from typing import Dict

class LOCM(OCM):
    def __init__(self):
        super().__init__()

    def extract_model(self, tracelist, sorts=None):
        
        sorts = self._get_sorts(tracelist, sorts)
        
        obj_trace_list, AM_list = self.trace_to_transition_matrix(tracelist, sorts)

    def trace_to_transition_matrix(self, tracelist, sorts: Dict):
        
        obj_trace_list = self._extract_trace(tracelist, sorts)
        return obj_trace_list