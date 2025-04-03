from .locm import LOCM
from collections import defaultdict
from typing import Set, Dict, List
from traces import *

class LOCM3(LOCM):

    def __init__(self,timeout:int=600, debug: Dict[str, bool]=None):
        super().__init__(timeout, debug)
        pass

    def extract_model(self, tracelist: List[Trace], sorts: Dict = None):
        sorts = self._get_sorts(tracelist)
        obj_trace_list, AM_list = self.trace_to_transition_matrix(self, tracelist, sorts)
        ce_B, ce_A = self.get_ceB_ceA(AM_list)

        pass
    

    
    def get_ceB_ceA(self, AM_list):
        ceB_list = []
        ceA_list = []
        for AM in AM_list:
            ceB: Set[Event] = set()
            ceA: Dict[Event, Set[Event]] = defaultdict(set)
            
            for i, row in AM.iterrows():
                for j, value in row.items():
                    if value == 1:
                        ceB.add(i)
                        ceA[i].add(j)
            ceB_list.append(ceB)
            ceA_list.append(ceA)
        return ceB_list, ceA_list
    
    def get_me_list(self, AM_list, ceB_list, ceA_list):
        for i in range(len(AM_list)):
            AM = AM_list[i]
            ceB = ceB_list[i]
            ceA = ceA_list[i]

            me = self.get_me(AM, ceB, ceA)
            if me is not None:
                return me
    
    def get_me(self, AM, ceB, ceA):
        me = FSM()

    
    def pre_works():
        pass

    
     


    
