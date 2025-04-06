from abc import ABC, abstractmethod
from typing import List, Set, Dict
from collections import defaultdict
from networkx import nx
from traces import *
from utlis import *

# Predefined types
Sorts= Dict[str, int]
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
    def _get_sorts(trace_list, types) -> Sorts:
        """Get the sorts of objects from the traces. If sorts is not given, use it.

        :param trace_list: List of traces to extract sorts from.
        :param sorts: Dictionary of sorts to use. If None, extract from traces.

        :return: Dictionary of sorts.
        """
        if types is not None:
           return types
        s = defaultdict(set)
        for obs_trace in trace_list:
            for obs in obs_trace:
                action = obs.action
                if action is None:
                    continue
                for i,obj in enumerate(action.obj_params):
                    s[action.name, i].add(obj.name)
        
        unique_sorts = list({frozenset(se) for se in s.values()})
        sorts_copy = {i: sort for i, sort in enumerate(unique_sorts)}
        # now do pairwise intersections of all values. If intersection, combine them; then return the final sets.
        while True:
            intersection_count = 0
            for i in list(sorts_copy.keys()):
                for j in list(sorts_copy.keys()):
                    if i >= j:
                        continue
                    s1 = sorts_copy.get(i, None)
                    if s1 is None:
                        continue
                    s2 = sorts_copy.get(j, None)
                    if s2 is None:
                        continue
                    if s1.intersection(s2):
                        intersection_count+=1
                        sorts_copy[i] = s1.union(s2)
                        del sorts_copy[j]
            if intersection_count == 0:
                break
        # add zero class
        obj_sorts = {}
        for i, sort in enumerate(sorts_copy.values()):
            for obj in sort:
                # NOTE: object sorts are 1-indexed so the zero-object can be sort 0
                obj_sorts[obj] = i + 1
        obj_sorts['zero'] = 0
 
        return obj_sorts

    @staticmethod
    def _pointer_to_set(states: List[Set], pointer1, pointer2 = None):
     pass

    



   