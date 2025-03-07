from abc import ABC, abstractmethod
from typing import List, Set, Dict
from collections import defaultdict
from networkx import nx
from traces import *
from utlis import *

ObjectTrace= Dict[PlanningObject, List[Event]]
ZERO = PlanningObject("zero", Type("zero"))

class OCM(ABC):

    def __init__(self):
        pass
    

    @staticmethod
    def _pointer_to_set(states: List[Set], pointer1, pointer2 = None):
        pass

    def step1(trace_list: List[Trace], sorts: bool, ocm_arity = 1, debug: bool = False):
        graphs = []
        for sort in range(len(set(sorts.values()))):
            graphs.append(nx.DiGraph())

        obj_trace_list = []
        for trace in trace_list:
            obj_trace: ObjectTrace = defaultdict(list)
            for step in trace.steps:
                action = step.action
                assert action, "Invalid input, missing Action"
                zero_event = Event(action, {0})
                obj_trace[ZERO].append(zero_event)
                for i, obj in enumerate(action.obj_params):
                    sort = sorts[obj.name]
                    event = Event(action, {i+1}, sort)
                    obj_trace[obj].append(event)
                    graphs[sort].add_node(event)
            obj_trace_list.append(obj_trace)

        AM_list = []
        for obj_trace in obj_trace_list:
            for obj, seq in obj_trace.items():
                sort = sorts[obj.name]
                for i in range(0, len(seq) -1):
                    if (not graphs[sort].has_edge(seq[i], seq[i+1])):
                        graphs[sort].add_edge(seq[i], seq[i+1], weight = 1)
        for i, G in enumerate(graphs):
            AM = nx.to_pandas_adjacency(G, nodelist=G.nodes(), dtype=int)
            AM_list.append(AM)
            if debug:
                print(f"AM for sort {i}")
                pprint_table(AM)
        return obj_trace_list

   
        
            
        

    @abstractmethod
    def extract_action_model(self):
        pass

   