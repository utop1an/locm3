from .ocm import OCM, TypedObject, Event, SingletonEvent, StatePointers
from typing import Dict, List, Tuple, Set, NamedTuple
from traces import Hypothesis, HIndex, HItem, FSM
from pddl import TypedObject
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from utils import pprint_table, check_well_formed, check_valid

class LOCM2(OCM):

    TransitionSet = Dict[FSM, Dict[TypedObject, List[Event]]]
    ObjectStates = Dict[FSM, List[Set[int]]]
    EventStatePointers = Dict[FSM, Dict[Event, StatePointers]]
    Binding = NamedTuple("Binding", [("hypothesis", Hypothesis), ("param", int)])
    Bindings = Dict[FSM, Dict[int, List[Binding]]]

    def __init__(self, state_param:bool=True, timeout:int=600, debug: Dict[str, bool]=None):
        super().__init__(state_param, timeout, debug)

    def extract_action_model(self, trace_list, types=None):
        
        sorts, sort_to_type_dict = self._get_sorts(trace_list, types)
        obj_tracelist, TM_list = self.trace_to_obj_trace(trace_list, sorts)
        transitions_per_sort_list = self.split_transitions(TM_list)

        return super().extract_action_model()
    

    def trace_to_obj_trace(self, trace_list, sorts, debug=False)-> Tuple:
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
                    obj_traces[zero_obj].append()
                    graphs[0].add_node(zero_event)
                    # for each combination of action name A and argument pos P
                    for j, obj in enumerate(action.obj_params):
                        # create transition A.P
                        event = SingletonEvent(action, pos=j + 1, sort=sorts[obj.name])
                        obj_traces[obj].append(event)
                        graphs[sorts[obj.name]].add_node(event)
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
            if debug:
                print(f"Transition matrix for sort {sort}:")
                pprint_table(TM)

        return obj_traces_list, TM_list
    
    def split_transitions(self, TM_list, debug=False)-> List[pd.DataFrame]:
        TM_list_with_holes = self.find_holes(TM_list, debug)
        holes_per_sort_list = self.extract_holes(TM_list_with_holes, debug)
        transitions_per_sort_list = self.get_sort_transitions(TM_list_with_holes)
        consecutive_transitions_per_sort_list = self.get_consecutive_transitions(TM_list_with_holes)
        transition_sets_per_sort_list = self.get_transition_sets(
            TM_list, 
            holes_per_sort_list, 
            transitions_per_sort_list, 
            consecutive_transitions_per_sort_list,
            debug
        )
        return transition_sets_per_sort_list
    
    def find_holes(TM_list, debug=False)-> List[pd.DataFrame]:
        TM_list_with_holes = []
        for sort, TM in enumerate(TM_list):
            df = TM.copy()
            if sort == 0:
                TM_list_with_holes.append(df)
                continue

            df1 = df.to_numpy()
            n = df1.shape[0]

            # Iterate over pairs of rows using upper triangular indices
            for i in range(n - 1):
                row1 = df1[i, :]
                for j in range(i + 1, n):
                    row2 = df1[j, :]

                    # Check if there is any common value in both rows (vectorized)
                    common_values_flag = np.any((row1 > 0) & (row2 > 0))

                    if common_values_flag:
                        # Check for holes: one row has a value, the other has zero (vectorized)
                        df1[i, :] = np.where((row1 == 0) & (row2 > 0), -1, df1[i, :])
                        df1[j, :] = np.where((row1 > 0) & (row2 == 0), -1, df1[j, :])

            # Convert NumPy array back to DataFrame
            df1 = pd.DataFrame(df1, index=df.index, columns=df.columns)

            if debug:
                print(f"Sort.{sort} TM with holes:")
                pprint_table(df1)

            TM_list_with_holes.append(df1)

        return TM_list_with_holes
    
    def extract_holes(self, TM_list, debug=False)-> List[Set[Tuple[SingletonEvent, SingletonEvent]]]:
        holes_per_sort_list  = []
        for sort, TM in enumerate(TM_list):
            if sort == 0:
                holes_per_sort_list.append(set())
                continue
            holes = set()
            for i in range(TM.shape[0]):
                for j in range(TM.shape[1]):
                    if TM.iloc[i, j] == -1:
                        holes.add((TM.index[i], TM.columns[j]))
            holes_per_sort_list.append(holes)
            if debug:
                print("#holes in Sort.{}: {}".format(sort, len(holes)))
        return holes_per_sort_list
    
    def get_sort_transitions(TM_list_with_holes):
        transitions_per_sort_list = []
        for TM_with_holes in TM_list_with_holes:
            transitions_per_sort_list.append(TM_list_with_holes.column.values)
        return transitions_per_sort_list
    
    def get_consecutive_transitions(TM_list_with_holes):
        consecutive_transitions_per_sort_list = []
        for TM in TM_list_with_holes:
            consecutive_transitions = set()
            for i in range(TM.shape[0]):
                for j in range(TM.shape[1]):
                    if TM.iloc[i, j] > 0:
                        consecutive_transitions.add((TM.index[i], TM.columns[j]))
            consecutive_transitions_per_sort_list.append(consecutive_transitions)
        return consecutive_transitions_per_sort_list
    
    def get_transition_sets(
            TM_list, 
            holes_per_sort_list, 
            transitions_per_sort_list, 
            consecutive_transitions_per_sort_list,
            debug=False):
        transition_sets_per_sort_list = []
        for sort, holes in enumerate(holes_per_sort_list):
            
            # Initialize a set for transition sets for the current sort
            transition_set_set = set()
            valid_pairs = set(consecutive_transitions_per_sort_list[sort])

            transitions = transitions_per_sort_list[sort]  # transitions is a list

            if holes:  # If there are any holes for the sort
                for hole in holes:
                    # hole is already a frozenset

                    # Check if the hole is already covered
                    is_hole_already_covered_flag = any(
                        hole.issubset(s_prime) for s_prime in transition_set_set
                    )

                    if not is_hole_already_covered_flag:
                        h = hole  # h is a frozenset
                        remaining_transitions = set(transitions) - h  # Convert transitions to set for set operations

                        found_valid_set = False
                        for size in range(len(h) + 1, len(transitions)):
                            k = size - len(h)
                            if k == len(remaining_transitions):
                                break

                            # Generate combinations efficiently using generators
                            for comb in combinations(remaining_transitions, k):
                                s = h.union(comb)  # h is frozenset, comb is tuple
                                s_frozen = frozenset(s)
                                

                                # Extract the subset DataFrame
                                subset_df = TM_list[sort].loc[list(s), list(s)]

                                # Check for well-formedness
                                if check_well_formed(subset_df):
                                    # Check for validity against data
                                    
                                    if check_valid(subset_df, valid_pairs):
                                        transition_set_set.add(s_frozen)
                                        found_valid_set = True
                                        break  # Exit combinations loop

                            if found_valid_set:
                                break  # Exit size loop

            # Step 7: Remove redundant sets
            # Since sets are unordered, we need a way to compare and remove subsets efficiently
            non_redundant_sets = []
            for s in transition_set_set:
                if not any(s < other_set for other_set in transition_set_set if s != other_set):
                    non_redundant_sets.append(s)
           
            # Step 8: Include all-transitions machine, even if it is not well-formed
            non_redundant_sets.append(set(transitions))

            if debug:
                print("#### Final transition set list for sort index", sort)
                for ts in non_redundant_sets:
                    print(set(ts))
            transition_sets_per_sort_list.append(non_redundant_sets)

        return transition_sets_per_sort_list
    
    def get_TS_OS(self, obj_traces_list, transition_sets_per_sort_list, debug=False):
        TS: LOCM2.TransitionSet = defaultdict(dict)
        OS: LOCM2.ObjectStates = defaultdict(list)
        event_state_pointers: LOCM2.EventStatePointers = defaultdict(dict)

        zero_obj = OCM.ZEROOBJ
        zero_event = SingletonEvent("zero", pos=0, sort=0)

        for sort, transition_sets in enumerate(transition_sets_per_sort_list):
            for index, transition_set in enumerate(transition_sets):
                fsm = FSM(sort, index)
            