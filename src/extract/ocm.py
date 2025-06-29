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
        if debug:
            debug = defaultdict(lambda: False, debug)
        else:
            debug = defaultdict(lambda: False)
        self.debug = debug
        self.viz = viz

    @abstractmethod
    def extract_model(self):
        pass


    

    def trace_to_obj_trace(self, trace_list, sorts, debug=False):
        """
        Convert a list of traces to a list of object traces.
        Each object trace is a dictionary mapping each object to a list of events.
        
        """
        # create the zero-object for zero analysis (step 2)
        zero_obj = OCM.ZEROOBJ
        graphs = []
        for sort in range(len(set(sorts.values()))):
            graphs.append(nx.DiGraph())
        
        # collect action sequences for each object
        obj_traces_list: List[OCM.ObjectTrace] = []
        
        for trace in trace_list:
            obj_traces: OCM.ObjectTrace = defaultdict(list)

            for step in trace.steps:
                action = step.action
                if action is not None:
                    # add the step for the zero-object
                    zero_event = SingletonEvent(action, pos=0, sort=0)
                    obj_traces[zero_obj].append(zero_event)
                    graphs[0].add_node(zero_event)
                    # for each combination of action name A and argument pos P
                    added_objs = []
                    for j, obj in enumerate(action.obj_params):
                        # create transition A.P
                        assert obj not in added_objs, "LOCMv1 cannot handle duplicate objects in the same action."

                        event = SingletonEvent(action, pos=j + 1, sort=sorts[obj.name])
                        obj_traces[obj].append(event)
                        graphs[sorts[obj.name]].add_node(event)
                        added_objs.append(obj)
            obj_traces_list.append(obj_traces)

        TM_list = []
        for obj_trace in obj_traces_list:
            for obj, seq in obj_trace.items():
                sort = sorts[obj.name]
                for i in range(0, len(seq) - 1):
                    graphs[sort].add_edge(seq[i],seq[i+1],weight=1)
                        
        for sort, G in enumerate(graphs):
            TM = nx.to_pandas_adjacency(G, nodelist=G.nodes(), dtype=int)
            TM_list.append(TM)
            if self.debug['trace_to_obj_trace']:
                print(f"Transition matrix for sort {sort}:")
                pprint_table(TM)

        grouped_obj_traces = defaultdict(list)
        for obj_traces in obj_traces_list:
            for obj, seq in obj_traces.items():
                grouped_obj_traces[obj].append(seq)

        return grouped_obj_traces, TM_list

    
    def get_TM_list(self, trace_list, identify_type=False):
        sorts, _ = self._get_sorts(trace_list, identify_type)
        _, TM_list = self.trace_to_obj_trace(trace_list, sorts)
        return TM_list
    
    @staticmethod
    def _get_sorts(trace_list, identify_type=False) -> tuple:
        """Get the sorts of objects from the traces. If sorts is not given, use it.

        :param trace_list: List of traces to extract sorts from.
        :param sorts: Dictionary of sorts to use. If None, extract from traces.

        :return: Dictionary of sorts, and a dictionary of sort to type mapping.
        """
        sort_to_type_dict  = {}
        
        
        if identify_type:
            sorts = []
            obj_sorts = {}
            for obs_trace in trace_list:
                for step in obs_trace.steps:
                    action = step.action
                    if action is None:
                        continue
                    for i,obj in enumerate(action.obj_params):
                        t = obj.type_name
                        if t not in sorts:
                            sorts.append(t)
                        obj_sorts[obj.name] = sorts.index(t) + 1  # 1-indexed sort
            obj_sorts['zero'] = 0  # zero-object sort
            return obj_sorts, sort_to_type_dict
        
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
    

    



   