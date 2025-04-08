from .ocm import OCM, TypedObject, Event, SingletonEvent, StatePointers
from pddl import ActionSignature, LearnedModel, LearnedLiftedFluent, LearnedAction
from typing import Dict, List, Set
from collections import defaultdict
import pandas as pd

class LOCM(OCM):

    TransitionSet = set[Event] | Dict[int, Dict[TypedObject, List[Event]]]
    ObjectStates = Dict[int, List[Set[int]]]

    def __init__(self):
        super().__init__()

    def extract_model(self, tracelist, sorts=None):
        
        sorts = self._get_sorts(tracelist, sorts)
        
        obj_trace_list = self.trace_to_obj_trace(tracelist, sorts)
        TS, OS, ap_state_pointers = self.get_OS_TS(obj_trace_list)
        bindings = self.get_state_bindings()



    def trace_to_obj_trace(self, trace_list, sorts: Dict):
        """
        Convert a list of traces to a list of object traces.
        Each object trace is a dictionary mapping each object to a list of events.
        
        """
        # create the zero-object for zero analysis (step 2)
        zero_obj = OCM.ZEROOBJ
        
        # collect action sequences for each object
        obj_traces_list: List[OCM.ObjectTrace] = []
        for trace in trace_list:
            obj_traces: OCM.ObjectTrace = defaultdict(list)
            for obs in trace:
                action = obs.action
                if action is not None:
                    # add the step for the zero-object
                    obj_traces[zero_obj].append(SingletonEvent(action, pos=0, sort=0))
                    # for each combination of action name A and argument pos P
                    for j, obj in enumerate(action.obj_params):
                        # create transition A.P
                        ap = SingletonEvent(action, pos=j + 1, sort=sorts[obj.name])
                        obj_traces[obj].append(ap)
        obj_traces_list.append(obj_traces)
    
    def get_OS_TS(self, obj_traces_list: List[OCM.ObjectTrace], sorts) -> List[pd.DataFrame]:
        """
        Convert a list of object traces to a list of transition matrices.
        
        """
        TS: LOCM.TransitionSet = defaultdict(dict)
        OS: LOCM.ObjectStates = defaultdict(list)
        ap_state_pointers = defaultdict(dict)

        zero_obj = OCM.ZEROOBJ

        for obj_traces in obj_traces_list:
         # iterate over each object and its action sequence
            for obj, seq in obj_traces.items():
                sort = sorts[obj.name] if obj != zero_obj else 0
                TS[sort][obj] = seq  # add the sequence to the transition set
                # max of the states already in OS[sort], plus 1
                state_n = (
                    max([max(s) for s in OS[sort]] + [0]) + 1
                )  # count current (new) state id
                prev_states: StatePointers = None  # type: ignore
                # iterate over each transition A.P in the sequence
                for ap in seq:
                    # if the transition has not been seen before for the current sort
                    if ap not in ap_state_pointers[sort]:
                        ap_state_pointers[sort][ap] = StatePointers(state_n, state_n + 1)

                        # add the start and end states to the state set as unique states
                        OS[sort].append({state_n})
                        OS[sort].append({state_n + 1})

                        state_n += 2

                    ap_states = ap_state_pointers[sort][ap]

                    if prev_states is not None:
                        # get the state ids (indecies) of the state sets containing
                        # start(A.P) and the end state of the previous transition
                        start_state, prev_end_state = LOCM._pointer_to_set(
                            OS[sort], ap_states.start, prev_states.end
                        )

                        # if not the same state set, merge the two
                        if start_state != prev_end_state:
                            OS[sort][start_state] = OS[sort][start_state].union(
                                OS[sort][prev_end_state]
                            )
                            OS[sort].pop(prev_end_state)
                        assert len(set.union(*OS[sort])) == sum([len(s) for s in OS[sort]])

                    prev_states = ap_states

            # remove the zero-object sort if it only has one state
            if len(OS[0]) == 1:
                ap_state_pointers[0] = {}
                OS[0] = []

        return TS, OS, ap_state_pointers
    

    def get_state_bindings():


        bindings = defaultdict(dict)


        return bindings
    
    def get_model(self, OS, ap_state_pointers, sorts, bindings, statics, debug=False, viz= False):
        
        # delete zero-object if it's state machine was discarded
        if not OS[0]:
         
            del OS[0]
            del ap_state_pointers[0]


        # all_aps = {action_name: [AP]}
        all_aps: Dict[str, List[Event]] = defaultdict(list)
        for aps in ap_state_pointers.values():
            for ap in aps:
                all_aps[ap.action.name].append(ap)

        state_params = defaultdict(dict)
        state_params_to_hyps = defaultdict(dict)
        for sort in bindings:
            state_params[sort] = defaultdict(dict)
            state_params_to_hyps[sort] = defaultdict(dict)
            for state in bindings[sort]:
                keys = {b.param for b in bindings[sort][state]}
                typ = None
                for key in keys:
                    hyps = [
                        b.hypothesis for b in bindings[sort][state] if b.param == key
                    ]
                    # assert that all are the same G_
                    assert len(set([h.G_ for h in hyps])) == 1
                    state_params[sort][state][key] = hyps[0].G_
                    state_params_to_hyps[sort][state][key] = hyps

        if viz:
            LOCM._debug_state_machines(OS, ap_state_pointers, state_params)

        fluents = defaultdict(dict)
        actions = {}
        for sort in ap_state_pointers:
            sort_str = f"sort{sort}"
            for ap in ap_state_pointers[sort]:
                if ap.action.name not in actions:
                    actions[ap.action.name] = LearnedAction(
                        ap.action.name,
                        [None for _ in range(len(all_aps[ap.action.name]))],  # type: ignore
                    )
                a = actions[ap.action.name]
                a.param_sorts[ap.pos - 1] = sort_str

                start_pointer, end_pointer = ap_state_pointers[sort][ap]
                start_state, end_state = LOCM._pointer_to_set(
                    OS[sort], start_pointer, end_pointer
                )

                start_fluent_name = f"sort{sort}_state{start_state}"
                if start_fluent_name not in fluents[ap.action.name]:
                    start_fluent = LearnedLiftedFluent(
                        start_fluent_name,
                        param_sorts=[sort_str],
                        param_act_inds=[ap.pos - 1],
                    )
                    fluents[ap.action.name][start_fluent_name] = start_fluent

                start_fluent = fluents[ap.action.name][start_fluent_name]

                if (
                    sort in state_params_to_hyps
                    and start_state in state_params_to_hyps[sort]
                ):
                    for param in state_params_to_hyps[sort][start_state]:
                        psort = None
                        pind = None
                        for hyp in state_params_to_hyps[sort][start_state][param]:
                            if hyp.C == ap:
                                if (psort is not None and psort != hyp.G_) or \
                                   (pind is not None and pind != hyp.l_):
                                    print(f"\n\tError: The following set of hypotheses for sort {sort} and state {start_state} are not consistent (ap = {ap}):")
                                    for hyp in state_params_to_hyps[sort][start_state][param]:
                                        if hyp.C == ap:
                                            print(f"\t\t{hyp}")
                                    print("\n\t This domain cannot be handled by LOCMv1. Please see https://github.com/AI-Planning/macq/discussions/200 for more info.\n\n")
                                    exit(1)
                                assert psort is None or psort == hyp.G_
                                assert pind is None or pind == hyp.l_
                                psort = hyp.G_
                                pind = hyp.l_
                        if psort is not None:
                            start_fluent.param_sorts.append(f"sort{psort}")
                            start_fluent.param_act_inds.append(pind - 1)

                a.update_precond(start_fluent)

                if end_state != start_state:
                    end_fluent_name = f"sort{sort}_state{end_state}"
                    if end_fluent_name not in fluents[ap.action.name]:
                        end_fluent = LearnedLiftedFluent(
                            end_fluent_name,
                            param_sorts=[sort_str],
                            param_act_inds=[ap.pos - 1],
                        )
                        fluents[ap.action.name][end_fluent_name] = end_fluent

                    end_fluent = fluents[ap.action.name][end_fluent_name]

                    if (
                        sort in state_params_to_hyps
                        and end_state in state_params_to_hyps[sort]
                    ):
                        for param in state_params_to_hyps[sort][end_state]:
                            psort = None
                            pind = None
                            for hyp in state_params_to_hyps[sort][end_state][param]:
                                if hyp.B == ap:
                                    if (psort is not None and psort != hyp.G_) or \
                                       (pind is not None and pind != hyp.k_):
                                         print(f"\n\tError: The following set of hypotheses for sort {sort} and state {end_state} are not consistent (ap = {ap}):")
                                         for hyp in state_params_to_hyps[sort][end_state][param]:
                                              if hyp.B == ap:
                                                    print(f"\t\t{hyp}")
                                         print("\n\t This domain cannot be handled by LOCMv1. Please see https://github.com/AI-Planning/macq/discussions/200 for more info.\n\n")
                                         exit(1)
                                    assert psort is None or psort == hyp.G_
                                    assert pind is None or pind == hyp.k_
                                    psort = hyp.G_
                                    pind = hyp.k_
                            if psort is not None:
                                end_fluent.param_sorts.append(f"sort{psort}")
                                end_fluent.param_act_inds.append(pind - 1)

                    a.update_delete(start_fluent)
                    a.update_add(end_fluent)

        # Step 6: Extraction of static preconditions
        for action in actions.values():
            if action.name in statics:
                for static in statics[action.name]:
                    action.update_precond(static)

        model = LearnedModel(fluents, actions, self.sorts)
        return model