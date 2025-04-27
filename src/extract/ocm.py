from abc import ABC, abstractmethod
from typing import List, Set, Dict
from collections import defaultdict
import networkx as nx
from traces import Trace, Event, SingletonEvent, StatePointers
from pddl import TypedObject, Type
from utils import *



class OCM(ABC):

    # Predefined types
    SortDict= Dict[str, int] # {obj_name: sort}
    ObjectTrace= Dict[TypedObject, List[Event]] 
    ZEROOBJ = TypedObject("zero","zero")

    def __init__(self,state_param:bool=True,viz=False, timeout:int = 600, debug: Dict[str, bool]=None):
        self.state_param = state_param
        self.timeout = timeout
        self.debug = debug
        self.viz = viz

    @abstractmethod
    def extract_model(self):
        pass

    @abstractmethod
    def trace_to_obj_trace(self):
        pass
    
    @staticmethod
    def _get_sorts(trace_list, type_dict) -> tuple:
        """Get the sorts of objects from the traces. If sorts is not given, use it.

        :param trace_list: List of traces to extract sorts from.
        :param sorts: Dictionary of sorts to use. If None, extract from traces.

        :return: Dictionary of sorts, and a dictionary of sort to type mapping.
        """
        sort_to_type_dict  = {}
        if type_dict is not None:
           raise NotImplementedError("Predefined type is not implemented yet.")
        s = defaultdict(set)
        for obs_trace in trace_list:
            for step in obs_trace.steps:
                action = step.action
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
 
        return obj_sorts, sort_to_type_dict
    
    @staticmethod
    def _sorts_to_types(sorts: SortDict, sort_to_type_dict ):
        if sort_to_type_dict  is None:
            types= [TypedObject(obj, f"s{sort}") for obj, sort in sorts.items()]
            return types
        
        raise NotImplementedError("Predefined type is not implemented yet.")
        types = {}
        return types

    @staticmethod
    def _pointer_to_set(states: List[Set], pointer1, pointer2 = None):
        """
        Get the state(index) of the given pointer in the states.
        """
        state1, state2 = None, None
        for i, state_set in enumerate(states):
            if pointer1 in state_set:
                state1 = i
            if pointer2 is None or pointer2 in state_set:
                state2 = i
            if state1 is not None and state2 is not None:
                break

        assert state1 is not None, f"Pointer ({pointer1}) not in states: {states}"
        assert state2 is not None, f"Pointer ({pointer2}) not in states: {states}"
        return state1, state2
    
    @staticmethod
    def _debug_state_machines(OS, ap_state_pointers, state_params):
        import os

        import networkx as nx

        for sort in OS:
            G = nx.DiGraph()
            for n in range(len(OS[sort])):
                lbl = f"state{n}"
                if (
                    state_params is not None
                    and sort in state_params
                    and n in state_params[sort]
                ):
                    lbl += str(
                        [
                            state_params[sort][n][v]
                            for v in sorted(state_params[sort][n].keys())
                        ]
                    )
                G.add_node(n, label=lbl, shape="oval")
            for ap, apstate in ap_state_pointers[sort].items():
                start_idx, end_idx = OCM._pointer_to_set(
                    OS[sort], apstate.start, apstate.end
                )
                # check if edge is already in graph
                if G.has_edge(start_idx, end_idx):
                    # append to the edge label
                    G.edges[start_idx, end_idx][
                        "label"
                    ] += f"\n{ap.action.name}.{ap.pos}"
                else:
                    G.add_edge(start_idx, end_idx, label=f"{ap.action.name}.{ap.pos}")
            # write to dot file
            nx.drawing.nx_pydot.write_dot(G, f"LOCM-step7-sort{sort}.dot")
            os.system(
                f"dot -Tpng LOCM-step7-sort{sort}.dot -o LOCM-step7-sort{sort}.png"
            )
            os.system(f"rm LOCM-step7-sort{sort}.dot")
    

    



   