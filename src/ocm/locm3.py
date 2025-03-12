from .ocm import OCM
from collections import defaultdict
from typing import Set, Dict
from traces import *

class LOCM3(OCM):

    def __init__(self):
        pass

    def extract_action_model(self):
        obj_trace_list, AM_list = self.step1()
        ce_B, ce_A = self.ceB_ceA(AM_list)



        pass

    def ceB_ceA(AM_list):
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

    
    def pre_works():
        pass

    
     


    
