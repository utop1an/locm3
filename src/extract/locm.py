from .ocm import OCM, TypedObject, SingletonEvent, StatePointers
from pddl import ActionSignature, LearnedModel, LearnedLiftedFluent, LearnedAction
from typing import Dict, List, Set, NamedTuple, Tuple
from collections import defaultdict
from traces import Hypothesis, HIndex, HItem
import pandas as pd

class LOCM(OCM):

    TransitionSet = Dict[Tuple, Dict[TypedObject, List[List[SingletonEvent]]]]
    ObjectStates = Dict[Tuple, List[Set[int]]]
    EventStatePointers = Dict[Tuple, Dict[SingletonEvent, StatePointers]]
    Binding = NamedTuple("Binding", [("hypothesis", Hypothesis), ("param", int)])
    Bindings = Dict[int, Dict[int, List[Binding]]]  # {sort: {state: [Binding]}}

    def __init__(self, event_arity=1, state_param:bool=True, viz=False, timeout:int=600, debug: Dict[str, bool]=None):
        super().__init__(event_arity=event_arity, state_param=state_param,viz=viz, timeout=timeout, debug=debug)
        

    def extract_model(self, tracelist, types=None)-> LearnedModel:
        
        sorts, sort_to_type_dict = self._get_sorts(tracelist, types)
        
        obj_trace_list, TM_list = self.trace_to_obj_trace(tracelist, sorts)
        TS, OS, ap_state_pointers = self.get_TS_OS(obj_trace_list, sorts)
      
        if self.state_param:
            bindings = self.get_state_bindings(TS, ap_state_pointers, OS, sorts, debug=self.debug)
        else:
            bindings = None
    
        model = self.get_model(OS, ap_state_pointers, sorts, bindings, None, statics=[], debug=False)
      
        return model


    def get_TS_OS(self, obj_traces, sorts) -> tuple[TransitionSet, ObjectStates, Dict[int, Dict[SingletonEvent, StatePointers]]]:
        """
        Convert a list of object traces to a list of transition matrices.
        
        """
        TS: LOCM.TransitionSet = defaultdict(lambda: defaultdict(list))
        OS: LOCM.ObjectStates = defaultdict(list)
        ap_state_pointers = defaultdict(dict)

        zero_obj = OCM.ZEROOBJ
        for obj_tuple, seqs in obj_traces.items():
         # iterate over each object and its action sequence
            for seq in seqs:
                sort_tuple = self.obj_tuple_to_sort_tuple(obj_tuple, sorts)
                TS[sort_tuple][obj_tuple].append(seq)  # add the sequence to the transition set
                # max of the states already in OS[sort], plus 1
                state_n = (
                    max([max(s) for s in OS[sort_tuple]] + [0]) + 1
                )  # count current (new) state id
                prev_states: StatePointers = None  # type: ignore
                # iterate over each transition A.P in the sequence
                for ap in seq:
                    # if the transition has not been seen before for the current sort
                    if ap not in ap_state_pointers[sort_tuple]:
                        ap_state_pointers[sort_tuple][ap] = StatePointers(state_n, state_n + 1)

                        # add the start and end states to the state set as unique states
                        OS[sort_tuple].append({state_n})
                        OS[sort_tuple].append({state_n + 1})

                        state_n += 2

                    ap_states = ap_state_pointers[sort_tuple][ap]

                    if prev_states is not None:
                        # get the state ids (indecies) of the state sets containing
                        # start(A.P) and the end state of the previous transition
                        start_state, prev_end_state = LOCM._pointer_to_set(
                            OS[sort_tuple], ap_states.start, prev_states.end
                        )

                        # if not the same state set, merge the two
                        if start_state != prev_end_state:
                            OS[sort_tuple][start_state] = OS[sort_tuple][start_state].union(
                                OS[sort_tuple][prev_end_state]
                            )
                            OS[sort_tuple].pop(prev_end_state)
                        assert len(set.union(*OS[sort_tuple])) == sum([len(s) for s in OS[sort_tuple]])

                    prev_states = ap_states

            # remove the zero-object sort if it only has one state
        if len(OS[(0,)]) == 1:
            ap_state_pointers[(0,)] = {}
            OS[(0,)] = []
 

        return TS, OS, ap_state_pointers

    

    def get_state_bindings(self,TS, ap_state_pointers, OS, sorts, debug=False):
        hs = self._step3(TS, ap_state_pointers, OS, sorts, debug)
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
        debug: bool = False,
    ) -> Hypothesis:
        """Step 3: Induction of parameterised FSMs"""

        zero_obj = OCM.ZEROOBJ

        # indexed by B.k and C.l for 3.2 matching hypotheses against transitions
        HS: Dict[HIndex, Set[HItem]] = defaultdict(set)

        # 3.1: Form hypotheses from state machines
        for G_tuple, sort_ts in TS.items():
            
            # for each O ∈ O_u (not including the zero-object)
            for obj_tuple, seqs in sort_ts.items():
                if obj_tuple == (zero_obj,):
                    continue
                for seq in seqs:
                # for each pair of transitions B.k and C.l consecutive for O
                    for B, C in zip(seq, seq[1:]):
                        # skip if B or C only have one parameter, since there is no k' or l' to match on
                        if len(B.action.obj_params) == 1 or len(C.action.obj_params) == 1:
                            continue

                        k_tuple = B.pos
                        l_tuple = C.pos

                        # check each pair B.k' and C.l'
                        for i, Bk_ in enumerate(B.action.obj_params):
                            k_ = i + 1
                            if k_ in k_tuple:
                                continue
                            G_ = sorts[Bk_.name]

                            for j, Cl_ in enumerate(C.action.obj_params):
                                l_ = j + 1
                                if l_ in l_tuple:
                                    continue

                                # check that B.k' and C.l' are of the same sort
                                if sorts[Cl_.name] == G_:
                                    # check that end(B.P) = start(C.P)
                                    # NOTE: just a sanity check, should never fail
                                    S, S2 = LOCM._pointer_to_set(
                                        OS[G_tuple],
                                        ap_state_pointers[G_tuple][B].end,
                                        ap_state_pointers[G_tuple][C].start,
                                    )
                                    assert (
                                        S == S2
                                    ), f"end(B.P) != start(C.P)\nB.P: {B}\nC.P: {C}"

                                    # save the hypothesis in the hypothesis set
                                    HS[HIndex(B, k_tuple, C, l_tuple)].add(
                                        HItem(S, k_, l_, G_tuple, G_, supported=False)
                                    )

        # 3.2: Test hypotheses against sequence
        for G_tuple, sort_ts in TS.items():
            # for each O ∈ O_u (not including the zero-object)
            for obj_tuple, seq in sort_ts.items():
                if obj_tuple == (zero_obj,):
                    continue
                for seq in seqs:
                    # for each pair of transitions Ap.m and Aq.n consecutive for O
                    for Ap, Aq in zip(seq, seq[1:]):
                        m_tuple = Ap.pos
                        n_tuple = Aq.pos
                        # Check if we have a hypothesis matching Ap=B, m=k, Aq=C, n=l
                        BkCl = HIndex(Ap, m_tuple, Aq, n_tuple)
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
        return Hypothesis.from_dict(HS)
    

    def _step4(self,HS: Hypothesis, debug: bool = False) -> Bindings:
        """Step 4: Creation and merging of state parameters"""
        # bindings = {sort: {state: [(hypothesis, state param)]}}
        bindings: LOCM.Bindings = defaultdict(dict)
        for sort_tuple, hs_sort in HS.items():
            for state, hs_sort_state in hs_sort.items():
                # state_bindings = {hypothesis (h): state param (v)}
                state_bindings: Dict[Hypothesis, int] = {}

                # state_params = [set(v)]; params in the same set are the same
                state_params: List[Set[int]] = []

                # state_param_pointers = {v: P}; maps state param to the state_params set index
                # i.e. map hypothesis state param v -> actual state param P
                state_param_pointers: Dict[int, int] = {}

                # for each hypothesis h,
                hs_sort_state = list(hs_sort_state)
                for v, h in enumerate(hs_sort_state):
                    # add the <h, v> binding pair
                    state_bindings[h] = v
                    # add a param v as a unique state parameter
                    state_params.append({v})
                    state_param_pointers[v] = v

                # for each (unordered) pair of hypotheses h1, h2
                for i, h1 in enumerate(hs_sort_state):
                    for h2 in hs_sort_state[i + 1 :]:
                        # check if hypothesis parameters (v1 & v2) need to be unified
                        if (
                            (h1.B.action == h2.B.action and h1.k == h2.k and h1.k_ == h2.k_)
                            or   # See https://github.com/AI-Planning/macq/discussions/200
                            (h1.C.action == h2.C.action and h1.l == h2.l and h1.l_ == h2.l_)  # fmt: skip
                        ):
                            v1 = state_bindings[h1]
                            v2 = state_bindings[h2]

                            # get the parameter sets P1, P2 that v1, v2 belong to
                            P1, P2 = LOCM._pointer_to_set(state_params, v1, v2)

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
                bindings[sort_tuple][state] = [
                    LOCM.Binding(h, LOCM._pointer_to_set(state_params, v)[0])
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
                                bindings[fsm][state].remove(LOCM.Binding(h, P_))
                        if len(bindings[fsm][state]) == 0:
                            del bindings[fsm][state]


        for k, v in bindings.copy().items():
            if not v:
                del bindings[k]

        return bindings
    
    def get_model(self, OS, ap_state_pointers, sorts, bindings, sort_to_type_dict=None, statics=[],  debug=False):
        
        # delete zero-object if it's state machine was discarded
        if not OS[(0,)]:
            
            del OS[(0,)]
            del ap_state_pointers[(0,)]
            shift = 1
        else:
            shift = 0
        
        def get_type(sort):
            if sort_to_type_dict:
                return sort_to_type_dict[sort]
            else:
                return f"s{sort}"

        # all_aps = {action_name: [AP]}
        actions = {}
        added_action_names = set()
        all_aps: Dict[str, Set[SingletonEvent]] = defaultdict(set)
        for aps in ap_state_pointers.values():
            for ap in aps:
                if ap.action.name not in added_action_names:
                    added_action_names.add(ap.action.name)
                    if shift == 1:
                        param_types = [get_type(sorts[obj.name]) for obj in ap.action.obj_params]
                    else:
                        param_types= ['s0'] + [get_type(sorts[obj.name]) for obj in ap.action.obj_params]
                    actions[ap.action.name] = LearnedAction(
                        ap.action.name,
                        param_types,  # type: ignore
                    )
        
        

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
            LOCM._debug_state_machines(OS, ap_state_pointers, state_params)

        fluents = defaultdict(dict)
        
        for sort_tuple in ap_state_pointers:
            type_str_list = [get_type(s) for s in sort_tuple]
            
            for ap in ap_state_pointers[sort_tuple]:
                pos_list = [p-shift for p in ap.pos]
                a = actions[ap.action.name]
              

                start_pointer, end_pointer = ap_state_pointers[sort_tuple][ap]
                start_state, end_state = LOCM._pointer_to_set(
                    OS[sort_tuple], start_pointer, end_pointer
                )

                start_fluent_name = f"sort{sort_tuple}_state{start_state}"
              
                if start_fluent_name not in fluents[ap]:
                    start_fluent = LearnedLiftedFluent(
                        start_fluent_name,
                        param_types=type_str_list,
                        param_act_idx=pos_list,
                    )
                    fluents[ap][start_fluent_name] = start_fluent

                start_fluent = fluents[ap][start_fluent_name]

                if (
                    bindings and 
                    sort_tuple in state_params_to_hyps
                    and start_state in state_params_to_hyps[sort_tuple]
                ):
                    for param in state_params_to_hyps[sort_tuple][start_state]:
                        psort = None
                        pind = None
                        for hyp in state_params_to_hyps[sort_tuple][start_state][param]:
                            if hyp.C == ap:
                                if (psort is not None and psort != hyp.G_) or \
                                   (pind is not None and pind != hyp.l_):
                                    print(f"\n\tError: The following set of hypotheses for sort {sort_tuple} and state {start_state} are not consistent (ap = {ap}):")
                                    for hyp in state_params_to_hyps[sort_tuple][start_state][param]:
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
                    end_fluent_name = f"sort{sort_tuple}_state{end_state}"
                    if end_fluent_name not in fluents[ap]:
                        end_fluent = LearnedLiftedFluent(
                            end_fluent_name,
                            param_types=type_str_list,
                            param_act_idx=pos_list,
                        )
                        fluents[ap][end_fluent_name] = end_fluent

                    end_fluent = fluents[ap][end_fluent_name]


                    if (
                        bindings and 
                        sort_tuple in state_params_to_hyps
                        and end_state in state_params_to_hyps[sort_tuple]
                    ):
                        for param in state_params_to_hyps[sort_tuple][end_state]:
                            psort = None
                            pind = None
                            for hyp in state_params_to_hyps[sort_tuple][end_state][param]:
                                if hyp.B == ap:
                                    if (psort is not None and psort != hyp.G_) or \
                                       (pind is not None and pind != hyp.k_):
                                         print(f"\n\tError: The following set of hypotheses for sort {sort_tuple} and state {end_state} are not consistent (ap = {ap}):")
                                         for hyp in state_params_to_hyps[sort_tuple][end_state][param]:
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