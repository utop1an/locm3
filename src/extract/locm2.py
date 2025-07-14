from .ocm import OCM, TypedObject, SingletonEvent, StatePointers
from typing import Dict, List, Tuple, Set, NamedTuple
from traces import Hypothesis, HIndex, HItem, FSM
from pddl import ActionSignature, LearnedModel, LearnedLiftedFluent, LearnedAction
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from utils import pprint_table, check_well_formed, check_valid, default_dict_factory

class LOCM2(OCM):

    TransitionSet = Dict[FSM, Dict[TypedObject, List[List[SingletonEvent]]]]
    ObjectStates = Dict[FSM, List[Set[int]]]
    EventStatePointers = Dict[FSM, Dict[SingletonEvent, StatePointers]]
    Hypothese = Dict[FSM, Dict[int, Set["Hypothesis"]]]
    Binding = NamedTuple("Binding", [("hypothesis", Hypothesis), ("param", int)])
    Bindings = Dict[FSM, Dict[int, List[Binding]]]



    def __init__(self, event_arity=1, state_param:bool=True, viz=False, timeout:int=600, debug: Dict[str, bool]=None):
        super().__init__(event_arity=event_arity, state_param=state_param, viz=viz, timeout=timeout, debug=debug)

    def extract_model(self, trace_list, types=None):
        
        sorts, sort_to_type_dict = self._get_sorts(trace_list, types)
        obj_traces, TM_list = self.trace_to_obj_trace(trace_list, sorts)
        transition_sets_per_sort_list = self.split_transitions(TM_list, obj_traces, sorts)
        TS, OS, ap_state_pointers = self.get_TS_OS(obj_traces, transition_sets_per_sort_list, TM_list, sorts)
        if self.state_param:
            bindings = self.get_state_bindings(TS, ap_state_pointers, OS, sorts,TM_list, debug=self.debug)
        else:
            bindings = None
        model = self.get_model(OS, ap_state_pointers, sorts, bindings, None, statics=[], debug=False)
        return model


    
    def split_transitions(self, TM_list, obj_traces_list, sorts)-> List[pd.DataFrame]:
        TM_list_with_holes = self.find_holes(TM_list)
        holes_per_sort_list = self.extract_holes(TM_list_with_holes)
        transitions_per_sort_list = self.get_sort_transitions(TM_list_with_holes)
        example_sequences_per_sort = self.get_example_sequences(obj_traces_list, sorts)
        transition_sets_per_sort_list = self.get_transition_sets(
            TM_list, 
            holes_per_sort_list, 
            transitions_per_sort_list, 
            example_sequences_per_sort,
        )
        return transition_sets_per_sort_list
    
    def find_holes(self, TM_list, debug=False)-> List[pd.DataFrame]:
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

            if self.debug['find_holes']: 
                print(f"Sort.{sort} TM with holes:")
                pprint_table(df1)

            TM_list_with_holes.append(df1)

        return TM_list_with_holes
    
    def extract_holes(self, TM_list_with_holes)-> List[Set[Tuple[SingletonEvent, SingletonEvent]]]:
        holes_per_sort_list  = []
        for sort, TM_with_holes in enumerate(TM_list_with_holes):
            if sort == 0:
                holes_per_sort_list.append(set())
                continue
            holes = set()
            for i in range(TM_with_holes.shape[0]):
                for j in range(TM_with_holes.shape[1]):
                    if TM_with_holes.iloc[i, j] == -1:
                        holes.add(frozenset({TM_with_holes.index[i], TM_with_holes.columns[j]}))
            holes_per_sort_list.append(holes)
            if self.debug['extract_holes']:
                print("#holes in Sort.{}: {}".format(sort, len(holes)))
                if (len(holes) > 0):
                    print(holes)
        return holes_per_sort_list
    
    def get_sort_transitions(self, TM_list_with_holes):
        transitions_per_sort_list = []
        for TM_with_holes in TM_list_with_holes:
            transitions_per_sort_list.append(TM_with_holes.columns.values)
        return transitions_per_sort_list
    
    def get_example_sequences(self,obj_traces, sorts):
        example_sequences_per_sort = defaultdict(list)
        
        for obj,obj_traces in obj_traces.items():
            for trace in obj_traces:
                if obj == OCM.ZEROOBJ:
                    continue
                example_sequences_per_sort[sorts[obj.name]].append(trace)
        return example_sequences_per_sort
    
    def get_transition_sets(
            self,
            TM_list, 
            holes_per_sort_list, 
            transitions_per_sort_list, 
            example_sequences_per_sort,
            debug=False):
        transition_sets_per_sort_list = []
        for sort, holes in enumerate(holes_per_sort_list):
                       # Initialize a set for transition sets for the current sort
            transition_set_set = set()
            

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
                                    
                                    if check_valid(subset_df, example_sequences_per_sort[sort]):
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

            if self.debug['get_transition_sets']:
                print("#### Final transition set list for sort index", sort)
                for ts in non_redundant_sets:
                    print(set(ts))
            transition_sets_per_sort_list.append(non_redundant_sets)

        return transition_sets_per_sort_list
    
    def get_TS_OS(self, obj_traces, transition_sets_per_sort_list, TM_list, sorts: OCM.SortDict, debug=False):
        TS: LOCM2.TransitionSet = defaultdict(lambda: defaultdict(list))
        OS: LOCM2.ObjectStates = defaultdict(list)
        event_state_pointers: LOCM2.EventStatePointers = defaultdict(dict)

        zero_obj = OCM.ZEROOBJ
        zero_fsm = FSM(0,0)

        for obj, traces in obj_traces.items():
            for trace in traces:
                if obj != zero_obj:
                    for sort_, transition_sets in enumerate(transition_sets_per_sort_list):
                        for fsm_no, transitions in enumerate(transition_sets):
                        
                            sort = sorts[obj.name]
                            if (sort == sort_):
                                subseq = [x for x in trace if x in transitions]
                                
                                fsm = FSM(sort, fsm_no)
                        
                                TS[fsm][obj].append(subseq)
                else:
                    TS[zero_fsm][zero_obj].append(trace)


         # initialize ap_state_pointers and OS      
        for sort, transition_sets in enumerate(transition_sets_per_sort_list):
            
            for fsm_no, transitions in enumerate(transition_sets):
                fsm = FSM(sort, fsm_no)

                state_n = 1  # count current (new) state id
                # add the sequence to the transition set
                prev_states: StatePointers = None  # type: ignore
                # iterate over each transition A.P in the sequence
                for ap in transitions:
                    # if the transition has not been seen copy TMfor the current sort
                    if ap not in event_state_pointers[fsm]:
                        event_state_pointers[fsm][ap] = StatePointers(state_n, state_n + 1)

                        # add the start and end states to the state set as unique states
                        OS[fsm].append({state_n})
                        OS[fsm].append({state_n + 1})

                        state_n += 2
        
        for fsm, ap_states in event_state_pointers.items():
            for ap, state in ap_states.items():
                
                ts = transition_sets_per_sort_list[fsm.sort][fsm.index]
                fsm_ts = TM_list[fsm.sort].loc[list(ts), list(ts)]
                prev_aps = fsm_ts[ap]
                for prev_ap, val in prev_aps.items():
                    if val > 0:
                        current_start, _ = OCM._pointer_to_set(OS[fsm], state.start, state.end)
                        prev_state = event_state_pointers[fsm][prev_ap]
                        _, prev_end = OCM._pointer_to_set(OS[fsm], prev_state.start, prev_state.end)
                        if (current_start != prev_end):
                            if OS[fsm][prev_end]:
                                OS[fsm][current_start] = OS[fsm][current_start].union(OS[fsm][prev_end])
                                OS[fsm].pop(prev_end)

        if len(OS[zero_fsm]) == 1:
            event_state_pointers[zero_fsm] = {}
            OS[zero_fsm] = []
        
        return TS, OS, event_state_pointers
    

    def get_state_bindings(self,TS, ap_state_pointers, OS, sorts, TM_list, debug=False):
        hs = self._step3(TS, ap_state_pointers, OS, sorts,TM_list, debug)
        bindings = self._step4(hs, debug)
        # remove parameter flaws
        bindings = self._step5(hs, bindings, ap_state_pointers, OS, debug)

        return bindings
    
    def _step3(
        self,
        TS: TransitionSet,
        ap_state_pointers: EventStatePointers,
        OS: ObjectStates,
        sorts: OCM.SortDict,
        TM_list,
        debug: bool = False,
    ) -> Hypothesis:
        """Step 3: Induction of parameterised FSMs"""

        zero_obj = LOCM2.ZEROOBJ
        
        # indexed by B.k and C.l for 3.2 matching hypotheses against transitions
        HS: Dict[HIndex, Set[HItem]] = defaultdict(set)

        # 3.1: Form hypotheses from state machines
        
        for fsm, fsm_ts in TS.items():
            
            G = fsm.sort
            # for each O ∈ O_u (not including the zero-object)
            for obj, seqs in fsm_ts.items():
               
                if obj == zero_obj:
                    continue
                # for each pair of transitions B.k and C.l consecutive for O
                for seq in seqs:
                    for B, C in zip(seq, seq[1:]):
                        # skip if B or C only have one parameter, since there is no k' or l' to match on
                        if len(B.action.obj_params) == 1 or len(C.action.obj_params) == 1:
                            continue
                        if TM_list[G].loc[B,C]==0:
                            continue

                        k = B.pos
                        l = C.pos

                        # check each pair B.k' and C.l'
                        for i, Bk_ in enumerate(B.action.obj_params):
                            k_ = i + 1
                            if k_ == k:
                                continue
                            G_ = sorts[Bk_.name]

                            for j, Cl_ in enumerate(C.action.obj_params):
                                l_ = j + 1
                                if l_ == l:
                                    continue

                                # check that B.k' and C.l' are of the same sort
                                if sorts[Cl_.name] == G_:
                                    # check that end(B.P) = start(C.P)
                                    # NOTE: just a sanity check, should never fail
                                    S, S2 = LOCM2._pointer_to_set(
                                        OS[fsm],
                                        ap_state_pointers[fsm][B].end,
                                        ap_state_pointers[fsm][C].start,
                                    )
                                    assert (
                                        S == S2
                                    ), f"end(B.P) != start(C.P)\nB.P: {B}\nC.P: {C}"
                                    # save the hypothesis in the hypothesis set
                                    HS[HIndex(B, k, C, l)].add(
                                        HItem(S, k_, l_, G, G_, supported=False, fsm=fsm)
                                    )

        # 3.2: Test hypotheses against sequence
       
        for fsm, fsm_ts in TS.items():
            # for each O ∈ O_u (not including the zero-object)
            for obj, seqs in fsm_ts.items():
                if obj == zero_obj:
                    continue
                for seq in seqs:
                # for each pair of transitions Ap.m and Aq.n consecutive for O
                    for Ap, Aq in zip(seq, seq[1:]):
                        m = Ap.pos
                        n = Aq.pos
                        # Check if we have a hypothesis matching Ap=B, m=k, Aq=C, n=l
                        BkCl = HIndex(Ap, m, Aq, n)
                        if BkCl in HS:
                            # check each matching hypothesis
                            for H in HS[BkCl].copy():
                                # if Op,k' = Oq,l' then mark the hypothesis as supported
                                if (
                                    Ap.action.obj_params[H.k_ - 1]
                                    == Aq.action.obj_params[H.l_ - 1]
                                ):
                                    H.supported = True
                                else:  # otherwise remove the hypothesis
                                    HS[BkCl].remove(H)
        
        # Remove any unsupported hypotheses (but yet undisputed)
        for hind, hs in HS.copy().items():
            for h in hs.copy():
                if not h.supported:
                    HS[hind].remove(h)
            if len(hs) == 0:
                del HS[hind]
        
        # Converts HS {HSIndex: HSItem} to a mapping of hypothesis for states of a sort {sort: {state: Hypothesis}}
        return Hypothesis.from_dict_fsm(HS)
    

    def _step4(self,HS: Hypothesis, debug: bool = False) -> Bindings:
        """Step 4: Creation and merging of state parameters"""
         # bindings = {fsm: {state: [(hypothesis, state param)]}}
        bindings: LOCM2.Bindings = defaultdict(dict)
        for fsm, hs_fsm in HS.items():
            for state, hs_fsm_state in hs_fsm.items():
                # state_bindings = {hypothesis (h): state param (v)}
                state_bindings: Dict[Hypothesis, int] = {}

                # state_params = [set(v)]; params in the same set are the same
                state_params: List[Set[int]] = []

                # state_param_pointers = {v: P}; maps state param to the state_params set index
                # i.e. map hypothesis state param v -> actual state param P
                state_param_pointers: Dict[int, int] = {}

                # for each hypothesis h,
                hs_fsm_state = list(hs_fsm_state)
                for v, h in enumerate(hs_fsm_state):
                    # add the <h, v> binding pair
                    state_bindings[h] = v
                    # add a param v as a unique state parameter
                    state_params.append({v})
                    state_param_pointers[v] = v

                # for each (unordered) pair of hypotheses h1, h2
                for i, h1 in enumerate(hs_fsm_state):
                    for h2 in hs_fsm_state[i + 1 :]:
                        # check if hypothesis parameters (v1 & v2) need to be unified
                        if (
                            (h1.B.action == h2.B.action and h1.k == h2.k and h1.k_ == h2.k_)
                            and # See https://github.com/AI-Planning/macq/discussions/200 
                            (h1.C.action == h2.C.action and h1.l == h2.l and h1.l_ == h2.l_)  # fmt: skip
                        ):
                            v1 = state_bindings[h1]
                            v2 = state_bindings[h2]

                            # get the parameter sets P1, P2 that v1, v2 belong to
                            P1, P2 = OCM._pointer_to_set(state_params, v1, v2)

                            if P1 != P2:
                                # merge P1 and P2
                                state_params[P1] = state_params[P1].union(
                                    state_params[P2]
                                )
                                state_params.pop(P2)
                                state_param_pointers[v2] = P1

                                # fix state_param_pointers after v2
                                for ind in range(v2 + 1, len(state_param_pointers)):
                                    state_param_pointers[ind] -= 1
                # add state bindings for the sort to the output bindings
                # replacing hypothesis params with actual state params
                bindings[fsm][state] = [
                    LOCM2.Binding(h, OCM._pointer_to_set(state_params, v)[0])
                    for h, v in state_bindings.items()
                ]
  
        
        return dict(bindings)
    
    def _step5(
        self,
        HS: Hypothesis,
        bindings: Bindings,
        ap_state_pointers: EventStatePointers,
        OS: ObjectStates,
        debug: bool = False,
    ) -> Bindings:
        """Step 5: Removing parameter flaws"""

        # check each bindings[G][S] -> (h, P)
        for fsm, hs_fsm in HS.items():
            for state, hs in hs_fsm.items():

                # track all the h.Bs that occur in bindings[G][S]
                pointers = OS[fsm][state]
                
                inaps = set(ap for ap, (start, end) in ap_state_pointers[fsm].items() if end in pointers)
                outaps = set(ap for ap, (start, end) in ap_state_pointers[fsm].items() if start in pointers)        
                
                # track the set of h.B that set parameter P
                sets_P = defaultdict(set)
              
                for h, P in bindings[fsm][state]:
                    sets_P[P].add(h)
                    
                # for each P, check if there is a transition h.B that never sets parameter P
                # i.e. if sets_P[P] != all_hB
                for P, setby in sets_P.items():
                    flag = True
                    for ap in inaps:
                        candidate_hs = {h for h in hs if h.B == ap}
                    
                        if len(candidate_hs) == 0:
                            flag = False
                            break
                        if len(candidate_hs.intersection(setby))==0:
                            flag = False
                            break
                    if flag:
                        for ap in outaps:
                            candidate_hs = {h for h in hs if h.C == ap}
                            if len(candidate_hs) == 0:
                                flag = False
                                break
                            if len(candidate_hs.intersection(setby))==0:
                                flag = False
                                break
                    if not flag:  # P is a flawed parameter
                        # remove all bindings referencing P
                        for h, P_ in bindings[fsm][state].copy():
                            if P_ == P:
                                bindings[fsm][state].remove(LOCM2.Binding(h, P_))
                        if len(bindings[fsm][state]) == 0:
                            del bindings[fsm][state]


        for k, v in bindings.copy().items():
            if not v:
                del bindings[k]

        return bindings
    
    def get_model(self, OS, ap_state_pointers, sorts, bindings, sort_to_type_dict=None, statics=[],  debug=False):
        zero_fsm = FSM(0,0)
        # delete zero-object if it's state machine was discarded
        if not OS[zero_fsm]:
            
            del OS[zero_fsm]
            del ap_state_pointers[zero_fsm]
            shift = 1
        else:
            shift = 0


        # all_aps = {action_name: [AP]}
        all_aps: Dict[str, Set[SingletonEvent]] = defaultdict(set)
        for aps in ap_state_pointers.values():
            for ap in aps:
                all_aps[ap.action.name].add(ap)

        def get_type(sort):
            if sort_to_type_dict:
                return sort_to_type_dict[sort]
            else:
                return f"s{sort}"

        state_params = defaultdict(dict)
        if bindings:
            state_params_to_hyps = defaultdict(dict)
            for fsm in bindings:
                state_params[fsm] = defaultdict(dict)
                state_params_to_hyps[fsm] = defaultdict(dict)
                for state in bindings[fsm]:
                    keys = {b.param for b in bindings[fsm][state]}
                    typ = None
                    for key in keys:
                        hyps = [
                            b.hypothesis for b in bindings[fsm][state] if b.param == key
                        ]
                        # assert that all are the same G_
                        assert len(set([h.G_ for h in hyps])) == 1
                        state_params[fsm][state][key] = hyps[0].G_
                        state_params_to_hyps[fsm][state][key] = hyps

        if self.viz:
            LOCM2._debug_state_machines(OS, ap_state_pointers, state_params)

        fluents = defaultdict(dict)
        actions = {}
        for fsm in ap_state_pointers:
            sort = fsm.sort
            type_str = get_type(sort)
            for ap in ap_state_pointers[fsm]:
                if ap.action.name not in actions:
                    actions[ap.action.name] = LearnedAction(
                        ap.action.name,
                        [None for _ in range(len(all_aps[ap.action.name]))],  # type: ignore
                    )
                a = actions[ap.action.name]
                a.param_types[ap.pos-shift] = type_str

                start_pointer, end_pointer = ap_state_pointers[fsm][ap]
                start_state, end_state = LOCM2._pointer_to_set(
                    OS[fsm], start_pointer, end_pointer
                )

                start_fluent_name = f"s{sort}f{fsm.index}_state{start_state}"
              
                if start_fluent_name not in fluents[ap]:
                    start_fluent = LearnedLiftedFluent(
                        start_fluent_name,
                        param_types=[type_str],
                        param_act_idx=[ap.pos-shift],
                    )
                    fluents[ap][start_fluent_name] = start_fluent

                start_fluent = fluents[ap][start_fluent_name]

                if (
                    bindings and 
                    fsm in state_params_to_hyps
                    and start_state in state_params_to_hyps[fsm]
                ):
                    for param in state_params_to_hyps[fsm][start_state]:
                        psort = None
                        pind = None
                        for hyp in state_params_to_hyps[fsm][start_state][param]:
                            if hyp.C == ap:
                                if (psort is not None and psort != hyp.G_) or \
                                   (pind is not None and pind != hyp.l_):
                                    print(f"\n\tError: The following set of hypotheses for sort {sort} fsm {fsm.index} and state {start_state} are not consistent (ap = {ap}):")
                                    for hyp in state_params_to_hyps[fsm][start_state][param]:
                                        if hyp.C == ap:
                                            print(f"\t\t{hyp}")
                                    print("\n\t This domain cannot be handled by LOCMv1. Please see https://github.com/AI-Planning/macq/discussions/200 for more info.\n\n")
                                    exit(1)
                                assert psort is None or psort == hyp.G_
                                assert pind is None or pind == hyp.l_
                                psort = hyp.G_
                                pind = hyp.l_
                        if psort is not None:
                            start_fluent.param_types.append(get_type(psort))
                            start_fluent.param_act_idx.append(pind - shift)

                a.update_precond(start_fluent)

                if end_state != start_state:
                    end_fluent_name = f"s{sort}f{fsm.index}_state{end_state}"
                    if end_fluent_name not in fluents[ap]:
                        end_fluent = LearnedLiftedFluent(
                            end_fluent_name,
                            param_types=[type_str],
                            param_act_idx=[ap.pos-shift],
                        )
                        fluents[ap][end_fluent_name] = end_fluent

                    end_fluent = fluents[ap][end_fluent_name]


                    if (
                        bindings and 
                        fsm in state_params_to_hyps
                        and end_state in state_params_to_hyps[fsm]
                    ):
                        for param in state_params_to_hyps[fsm][end_state]:
                            psort = None
                            pind = None
                            for hyp in state_params_to_hyps[fsm][end_state][param]:
                                if hyp.B == ap:
                                    if (psort is not None and psort != hyp.G_) or \
                                       (pind is not None and pind != hyp.k_):
                                         print(f"\n\tError: The following set of hypotheses for sort {sort} fsm {fsm.index} and state {end_state} are not consistent (ap = {ap}):")
                                         for hyp in state_params_to_hyps[fsm][end_state][param]:
                                              if hyp.B == ap:
                                                    print(f"\t\t{hyp}")
                                         print("\n\t This domain cannot be handled by LOCMv1. Please see https://github.com/AI-Planning/macq/discussions/200 for more info.\n\n")
                                         exit(1)
                                    assert psort is None or psort == hyp.G_
                                    assert pind is None or pind == hyp.k_
                                    psort = hyp.G_
                                    pind = hyp.k_
                            if psort is not None:
                                end_fluent.param_types.append(get_type(psort))
                                end_fluent.param_act_idx.append(pind - shift)

                    a.update_delete(start_fluent)
                    a.update_add(end_fluent)

        # Step 6: Extraction of static preconditions
        for action in actions.values():
            if action.name in statics:
                for static in statics[action.name]:
                    action.update_precond(static)

        fluents = set(fluent
            for action_fluents in fluents.values()
            for fluent in action_fluents.values())
        actions = set(actions.values())
        types = OCM._sorts_to_types(sorts, None)
        model = LearnedModel(fluents, actions, types)
        return model
            