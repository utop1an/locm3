from abc import ABC, abstractmethod
from typing import List, Set, Dict
from collections import defaultdict
from networkx import nx
from traces import *
from utlis import *

ObjectTrace= Dict[PlanningObject, List[Event]]
ZERO = PlanningObject("zero", Type("zero"))

class OCM(ABC):

    def __init__(self,timeout:int = 600, debug: Dict[str, bool]=None):
        self.timeout = timeout
        self.debug = debug

    @abstractmethod
    def extract_model(self):
        pass

    @abstractmethod
    def trace_to_transition_matrix(self):
        pass
    
    @staticmethod
    def _get_sorts(tracelist, sorts) -> Dict:
        if sorts is not None:
           return sorts
        pass

    @staticmethod
    def _pointer_to_set(states: List[Set], pointer1, pointer2 = None):
     pass

    



   