from .ocm import OCM
from collections import defaultdict
from typing import Set, Dict, List, Tuple
from pddl.actions import LearnedAction, LearnedLiftedFluent, TypedObject
from pddl.model import LearnedModel
from traces import OSM, Trace, Event
import pandas as pd
import networkx as nx
from utils import pprint_table
from itertools import combinations, permutations

def generate_permutations(pos: List[int], arity: int):
    result = []
    k = min(len(pos), arity)
    for r in range(1, k+1):
        for comb in combinations(pos, r):
            for perm in permutations(comb):
                result.append(tuple(perm))
    return result

def generate_combinations(pos: List[int], arity: int):
    result = []
    k = min(len(pos), arity)
    for r in range(1, k+1):
        for comb in combinations(pos, r):
            result.append(tuple(comb))
    return result

class LOCM3(OCM):

    ObjectTrace= Dict[Tuple[TypedObject], List[Event]] 

    def __init__(self, event_arity=1,timeout:int=600, viz=False, debug: Dict[str, bool]=None):
        super().__init__(timeout=timeout, debug=debug, viz=viz)
        self.event_arity = event_arity


    

    def extract_model(self, trace_list: List[Trace], types: Dict = None):
        sorts, _ = self._get_sorts(trace_list, types)
        obj_trace_list, TM_list = self.trace_to_obj_trace(trace_list, sorts)
        OSM_list =  self.TM_to_OSM(TM_list)
        model = self.get_model(OSM_list, sorts)

        return model
    
    def trace_to_obj_trace(self, trace_list, sorts, debug=False):
        """
        Convert a list of traces to a list of object traces.
        Each object trace is a dictionary mapping each object to a list of events.
        
        """
        # create the zero-object for zero analysis (step 2)
        zero_obj = OCM.ZEROOBJ
        graphs = defaultdict(lambda: nx.DiGraph())
        
        
        # collect action sequences for each object
        obj_traces_list: List[OCM.ObjectTrace] = []

        for trace in trace_list:
            obj_traces: OCM.ObjectTrace = defaultdict(list)

            for step in trace.steps:
                action = step.action
                assert action is not None, "Action cannot be None"
                zero_event = Event(action, pos=(0,), sort=(0,))
                obj_traces[(zero_obj,)].append(zero_event)
                graphs[(0,)].add_node(zero_event)

                all_pos_tuples = generate_permutations([i+1 for i in range(len(action.obj_params))], self.event_arity)
                # add the step for the zero-object
                for pos_tuple in all_pos_tuples:
                    objs = tuple(action.obj_params[pos-1] for pos in pos_tuple)
                    sort_tuple = tuple(sorts[obj.name] for obj in objs)
                    event = Event(action, pos=pos_tuple, sort=sort_tuple)
                    
                    obj_traces[objs].append(event)
                    graphs[sort_tuple].add_node(event)
                    
            obj_traces_list.append(obj_traces)

        TM_dict = defaultdict(pd.DataFrame)
        for obj_trace in obj_traces_list:
            for objs, seq in obj_trace.items():
                sort_tuple = tuple(sorts[obj.name] for obj in objs)
                for i in range(0, len(seq) - 1):
                    graphs[sort_tuple].add_edge(seq[i],seq[i+1],weight=1)
                        
        for sort_tuple, G in graphs.items():
            TM = nx.to_pandas_adjacency(G, nodelist=G.nodes(), dtype=int)
            TM_dict[sort_tuple] = TM
            if self.debug['trace_to_obj_trace']:
                print(f"Transition matrix for sort {sort_tuple}:")
                pprint_table(TM)

        return  obj_traces_list, TM_dict

    def TM_to_OSM(self, TM_dict: List[pd.DataFrame]):
        OSM_dict = defaultdict(list)
        for sort_tuple, TM in TM_dict.items():
            OSM_by_event = []
            for event, row in TM.iterrows():
                ceA, self_included = self.get_ceA(TM, event)
                # TODO: check if an even has no consecutive events after it and ban such an OSM?
                
                if not ceA:
                    continue
                
                osm = OSM(sort_tuple, event, ceA, self_included)
                OSM_by_event.append(osm)
            OSM_dict[sort_tuple] = OSM_by_event

        if self.viz:
            self._debug_state_machine(OSM_dict)
        return OSM_dict
    
    def get_ceA(self, TM: pd.DataFrame, event: Event):
        """
        Get the set of events that are consecutively after the given evenit.
        Except the event itself.
        
        Parameters
        ----------
        TM : pd.DataFrame
            Transition matrix.
        event : Event
            Event to get the set of events after it.

        Returns
        -------
        ceA : Set
            Set of events that are consecutively after the given event.
        self_included : bool
            True if the event itself is included in the set of events.
        """
        ceA = set()
        self_included = False
        for i, row in TM.iterrows():
            for j, val in row.items():
                if val == 1:
                    if j == event:
                        self_included = True
                    else:
                        ceA.add(j)
        return ceA, self_included
    

    def _debug_state_machine(self, OSM_dict: Dict[Tuple[int],List[OSM]]):
        """
        Debug the state machine by drawing the OSMs.
        """
        for sort_tuple, OSM_by_event in OSM_dict.items():
            for osm in OSM_by_event:
                osm.draw()
    # osm has one state
    def get_model(self, OSM_dict: Dict[Tuple[int],List[OSM]], sorts: Dict, sort_to_type_dict: Dict[int, str] = None) -> LearnedModel:
        actions = {}
        fluents = defaultdict(tuple)

        def get_type(sort):
            if sort_to_type_dict:
                return sort_to_type_dict[sort]
            else:
                return f"s{sort}"

        for sort_tuple, OSM_by_event in OSM_dict.items():
            for osm in OSM_by_event:
                event = osm.event
                if event.action.name not in actions:
                    
                    action = LearnedAction(event.action.name, ['s0'] + [get_type(sorts[obj.name]) for obj in event.action.obj_params])
                    actions[event.action.name] = action
                else:
                    action = actions[event.action.name]

                if event not in fluents:

                    f0 = LearnedLiftedFluent(
                        f"m-{event.action.name}.{event.pos}-s0",
                        [s for s in event.sort],
                        [pos for pos in event.pos],
                    )

                    f1 = LearnedLiftedFluent(
                        f"m-{event.action.name}.{event.pos}-s1",
                        [s for s in event.sort],
                        [pos for pos in event.pos],
                    )
                    
                    fluents[event] = (f0, f1)
                else:
                    f0, f1 = fluents[event]

                action.update_precond(f0)
                action.update_add(f1)
                action.update_delete(f0)

                for event_after in osm.del_events:
                    if event_after.action.name not in actions:
                        action_after = LearnedAction(event_after.action.name, ['s0'] + [get_type(sorts[obj.name]) for obj in event_after.action.obj_params])
                        actions[event_after.action.name] = action_after
                    else:
                        action_after = actions[event_after.action.name]

                    if event_after not in fluents:

                        _f0 = LearnedLiftedFluent(
                            f"m-{event_after.action.name}.{event_after.pos}-s0",
                            [s for s in event_after.sort],
                            [pos for pos in event_after.pos],
                        )

                        _f1 = LearnedLiftedFluent(
                            f"m-{event_after.action.name}.{event_after.pos}-s1",
                            [s for s in event_after.sort],
                            [pos for pos in event_after.pos],
                        )
                        
                        fluents[event_after] = (_f0, _f1)
                    else:
                        _f0, _f1 = fluents[event_after]
                    
                    action_after.update_precond(_f1)
                    action_after.update_add(_f0)
                    action_after.update_delete(_f1)
        
        fluents = set(f for pair in fluents.values() for f in pair)
   
        actions = set(actions.values())
      
        types = OCM._sorts_to_types(sorts, None)
        model = LearnedModel(fluents, actions, types)
        return model
    
    # osm has one state
    # def get_model(self, OSM_dict: Dict[Tuple[int],List[OSM]], sorts: Dict):
    #     actions = {}
    #     fluents = set()

    #     for sort_tuple, OSM_by_event in OSM_dict.items():
    #         for osm in OSM_by_event:
    #             event = osm.event
    #             if event.action.name not in actions:
    #                 action = LearnedAction(event.action, [sorts[obj] for obj in event.action.obj_params])
    #             else:
    #                 action = actions[event.action.name]

    #             fluent = LearnedLiftedFluent(
    #                 f"m{event.action.name}.{event.position}.1",
    #                 [sorts[obj] for i,obj in enumerate(event.action.obj_params) if i in event.position ],
    #                 [pos for pos in event.position],
    #             )
    #             # f0 = LearnedLiftedFluent(
    #             #     f"m{event.action.name}.{event.position}.0",
    #             #     [sorts[obj] for i,obj in enumerate(event.action.obj_params) if i in event.position ],
    #             #     [pos for pos in event.position],
    #             # )
    #             fluents.add(fluent)
    #             # fluents.add(f0)

                
    #             action.update_precond(f0)
    #             action.update_add(f1)
    #             action.update_delete(f0)

    #             for event_after in osm.del_events:
    #                 if event_after.action.name not in actions:
    #                     action_after = LearnedAction(event_after.action, [sorts[obj] for obj in event_after.action.obj_params])
    #                 else:
    #                     action_after = actions[event_after.action.name]
                    
    #                 action_after.update_precond(f1)
    #                 action_after.update_add(f0)
    #                 action_after.update_delete(f1)
        

    #     actions = set(actions.values())
    #     types = OCM._sorts_to_types(sorts, None)
    #     model = LearnedModel(fluents, actions, types)
    #     return model

                


    

    
     


    
