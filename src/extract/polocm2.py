from .ocm import TypedObject, Event, SingletonEvent, StatePointers
from .locm2 import LOCM2
from typing import Dict, List, Tuple, Set, NamedTuple
from traces import Hypothesis, HIndex, HItem, FSM, PartialOrderedTrace, IndexedEvent, SingletonEvent
from pddl import ActionSignature, LearnedModel, LearnedLiftedFluent, LearnedAction
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import pulp as pl
from itertools import combinations
from utils.helpers import pprint_table, check_well_formed, check_valid, complete_PO, complete_FO

class POLOCM2(LOCM2):

    def __init__(self, solver_path="default", cores= 1,state_param:bool=True, viz=False, timeout:int = 600, debug: Dict[str, bool]=None):
        super().__init__(state_param=state_param, viz=viz, timeout=timeout, debug=debug)
        self.solver_path = solver_path
        self.cores = cores

    def extract_model(self, PO_trace_list: List[PartialOrderedTrace], type_dict: Dict[str, int] = None) -> LearnedModel:

        sorts, sort_to_type_dict = self._get_sorts(PO_trace_list, type_dict)
        obj_traces_list, TM_list = self.solve_po(PO_trace_list, sorts)
   
        transition_sets_per_sort_list = self.split_transitions(TM_list, obj_traces_list, sorts)
        TS, OS, ap_state_pointers = self.get_TS_OS(obj_traces_list, transition_sets_per_sort_list, TM_list, sorts)
        if self.state_param:
            bindings = self.get_state_bindings(TS, ap_state_pointers, OS, sorts, TM_list, debug=self.debug)
        else:
            bindings = None
        model = self.get_model(OS, ap_state_pointers, sorts, bindings, None, statics=[], debug=False)
        return model
        

    def solve_po(self, PO_trace_list, sorts):
        if self.solver_path == 'default':
            solver = pl.PULP_CBC_CMD(msg=False, timeLimit=600)
        else:
            print(f"launching cplex with {self.cores} cores")
            solver = pl.CPLEX_CMD(path=self.solver_path, msg=False, timeLimit=600, threads=self.cores, maxMemory=8192)
        
        trace_PO_matrix_overall, \
        obj_trace_PO_matrix_overall, \
        obj_trace_FO_matrix_overall, \
        sort_aps = self.get_matrices(PO_trace_list, sorts)
        prob, PO_vars_overall = self.build_PO_constraints(
            trace_PO_matrix_overall, 
            obj_trace_PO_matrix_overall
            )
        
        prob, FO_vars_overall = self.build_FO_constraints(
            prob,
            PO_vars_overall,
            obj_trace_PO_matrix_overall,
            obj_trace_FO_matrix_overall
            )
        prob, sort_transition_matrix, sort_AP_vars = self.build_TM_constraints(
            prob,
            sort_aps,
            sorts,
            FO_vars_overall,
            obj_trace_FO_matrix_overall)
        solution = self.bip_solve_PO(
            prob,
            solver,
            sort_AP_vars
            )
        obj_traces_list, TM_list = self.get_obj_traces(
            obj_trace_PO_matrix_overall,
            PO_vars_overall,
            obj_trace_FO_matrix_overall,
            FO_vars_overall,
            sort_transition_matrix,
            sort_AP_vars,
            solution
            )
        return obj_traces_list, TM_list

    def get_matrices(self, PO_trace_list, sorts):
         # TODO: add zero obj
        zero_obj = LOCM2.ZEROOBJ
        
        sort_aps:Dict[int, Set[SingletonEvent]] = defaultdict(set)
        # obj_traces for all obs_traces in obs_tracelist, indexed by trace_no
        obj_PO_trace_overall = []
        trace_PO_matrix_overall = []
        for PO_trace in PO_trace_list:
            indexed_actions = [IndexedEvent(step.action, step.index, None, None) for step in PO_trace]
            traces_PO_matrix = pd.DataFrame(columns=indexed_actions, index=indexed_actions)
            # collect action sequences for each object
            obj_PO_traces: Dict[TypedObject, List[SingletonEvent|IndexedEvent]] = defaultdict(list)
            for i, PO_step in enumerate(PO_trace):
                
                action = PO_step.action
                current_iap = indexed_actions[i]
                if action is not None:
                    zero_iap = IndexedEvent(action,PO_step.index ,0, 0)
                    obj_PO_traces[zero_obj].append(zero_iap)
                    sort_aps[0].add(zero_iap.to_event())
                    # for each combination of action name A and argument pos P
                    added_objs = []
                    for k, obj in enumerate(action.obj_params):
                        # create transition A.P
                        assert obj not in added_objs, "LOCMv1 cannot handle duplicate objects in the same action."
                        sort = sorts[obj.name]
                        iap = IndexedEvent(action, PO_step.index, pos=k + 1, sort=sort)
                        obj_PO_traces[obj].append(iap)
                        sort_aps[sort].add(iap.to_event())
                        added_objs.append(obj)
                      
                    for successor_ind in PO_step.successors:
                        successor = PO_trace[successor_ind]
                        assert successor != None
                        successor_iap = IndexedEvent(successor.action, successor_ind, None, None)
                        j = indexed_actions.index(successor_iap)
                        assert i!=j, "Invalid successor!"
                        if i < j:
                            traces_PO_matrix.at[current_iap, successor_iap]=1
                            traces_PO_matrix.at[successor_iap, current_iap]=0
                        else:
                            traces_PO_matrix.at[current_iap, successor_iap]=1
                            traces_PO_matrix.at[successor_iap, current_iap]=0
            complete_PO(traces_PO_matrix)
            trace_PO_matrix_overall.append(traces_PO_matrix) 
            obj_PO_trace_overall.append(obj_PO_traces)
        
    
        # Constructing PO matrix of actions for each object in each trace
        obj_trace_PO_matrix_overall = []
        # Initialize FO matrix
        obj_trace_FO_matrix_overall = []
        for trace_no,obj_PO_trace in enumerate(obj_PO_trace_overall):
            PO_matrices: Dict[TypedObject, pd.DataFrame] = defaultdict()
            FO_matrices: Dict[TypedObject, pd.DataFrame] = defaultdict()
            for obj, iaps in obj_PO_trace.items():
                obj_trace_PO_matrix = pd.DataFrame(columns=iaps, index=iaps)    
                obj_trace_FO_matrix = pd.DataFrame(columns=iaps, index=iaps)                
                for row_header, row in obj_trace_PO_matrix.iterrows():
                    for col_header, val in row.items():
                        origin = trace_PO_matrix_overall[trace_no].at[row_header.to_indexed_action(),
                                                                      col_header.to_indexed_action()]
                        obj_trace_PO_matrix.at[row_header, col_header] = origin
                
                complete_PO(obj_trace_PO_matrix)
                PO_matrices[obj] = obj_trace_PO_matrix
                complete_FO(obj_trace_FO_matrix, obj_trace_PO_matrix)
                FO_matrices[obj] = obj_trace_FO_matrix  
            obj_trace_PO_matrix_overall.append(PO_matrices)
            obj_trace_FO_matrix_overall.append(FO_matrices)
        

        if self.debug['get_matrices']:
          
            print("## Obj PO traces:\n")
            for i, obj_PO_traces in enumerate(obj_PO_trace_overall):
                print(f"### Trace **{i}**\n")
                pprint_table(obj_PO_traces.items())

            print("## Traces PO Matrix")
            for i, matrix in enumerate(trace_PO_matrix_overall):
                print(f"### Trace **{i}**\n")
                pprint_table(matrix)
            
            print("## Object Traces PO Matrix")
            for i, matrices in enumerate(obj_trace_PO_matrix_overall):
                print(f"### Trace **{i}**\n")
                for obj, matrix in matrices.items():
                    print(f"#### Object ***{obj.name}***\n")
                    pprint_table(matrix)
            
            print("## Object Traces FO Matrix")
            for i, matrices in enumerate(obj_trace_FO_matrix_overall):
                print(f"### Trace {i}\n")
                for obj, matrix in matrices.items():
                    print(f"#### Object {obj.name}\n")
                    pprint_table(matrix)
        return trace_PO_matrix_overall, obj_trace_PO_matrix_overall, obj_trace_FO_matrix_overall, sort_aps

    def build_PO_constraints(self,trace_PO_matrix_overall, obj_trace_PO_matrix_overall,):
        prob = pl.LpProblem("polocm", sense=pl.LpMinimize)
        PO_vars_overall= []
            
        for trace_no, matrix in enumerate(trace_PO_matrix_overall):
            PO_vars = {}
            
            cols = matrix.columns.tolist()
            for i in range(len(matrix)):
                for j in range(i+1,len(matrix)):
                    if pd.isna(matrix.iloc[i,j]) or matrix.iloc[i,j] == np.nan:
                        var_name = f"t_{trace_no}_{repr(cols[i])}_{repr(cols[j])}"
                        var = pl.LpVariable(var_name, cat=pl.LpBinary, upBound=1, lowBound=0) 
                        PO_vars[(cols[i], cols[j])] = var 
                        matrix.iloc[i,j] = (cols[i], cols[j])

                        transpose_var_name = f"t_{trace_no}_{repr(cols[j])}_{repr(cols[i])}"
                        transpose_var = pl.LpVariable(transpose_var_name, cat=pl.LpBinary, upBound=1, lowBound=0) 
                        PO_vars[(cols[j], cols[i])] = transpose_var 
                        matrix.iloc[j,i] = (cols[j], cols[i])

                        prob += var == 1 - transpose_var # (i,j) == not (j,i)
            
            PO_vars_overall.append(PO_vars)
            for i in range(len(matrix)):
                for j in range(i+1,len(matrix)):
                    current = PO_vars.get(matrix.iloc[i,j], matrix.iloc[i,j])
                    for x in range(len(matrix)):
                        if (x==i or x ==j):
                            continue
                        _next = PO_vars.get(matrix.iloc[j,x], matrix.iloc[j,x])
                        target = PO_vars.get(matrix.iloc[i, x],matrix.iloc[i, x])
                        if (isinstance(current, pl.LpVariable) or isinstance(_next, pl.LpVariable)):
                            prob += target >= current + _next -1
                            prob += target <= current + _next
        
            
        
        for trace_no, matrices in enumerate(obj_trace_PO_matrix_overall):
            
            PO_vars = PO_vars_overall[trace_no]
            for obj, matrix in matrices.items():
                
                for row_header, row in matrix.iterrows():
                    for col_header, val in row.items():
                        key = (row_header.to_indexed_action(), col_header.to_indexed_action())
                        if key in PO_vars.keys():
                            matrix.at[row_header, col_header] = key
                   

        # adding constraints on transitivity
        # for trace_no, matrices in enumerate(obj_trace_PO_matrix_overall):
        #     PO_vars = PO_vars_overall[trace_no]
        #     for obj, matrix in matrices.items():
        #         for i in range(len(matrix)):
        #             for j in range(i+1, len(matrix)):
        #                 current = PO_vars.get(matrix.iloc[i,j], matrix.iloc[i,j])
        #                 for x in range(len(matrix)):
        #                     if (x==i or x ==j):
        #                         continue
                            
        #                     _next = PO_vars.get(matrix.iloc[j,x], matrix.iloc[j,x])
        #                     target = PO_vars.get(matrix.iloc[i, x],matrix.iloc[i, x])
        #                     if (isinstance(current, pl.LpVariable) or isinstance(_next, pl.LpVariable)):
        #                         prob += target >= current + _next -1 # a>b, b>c, then a>c
        #                         prob += target <= current + _next  # a<b. b<c, then a<c

                                    
        if self.debug['build_PO_constraints']:
            print("# Step 2")
            print("## PO vars")
            for trace_no, PO_vars in enumerate(PO_vars_overall):
                print(f"### Trace {trace_no}")
                print(PO_vars)

            print("## PO matrix with vars")
            for trace_no, matrices in enumerate(obj_trace_PO_matrix_overall):
                print(f"### Trace {trace_no}")
                for obj, matrix in matrices.items():
                    print(f"#### Obj {obj.name}")
                    pprint_table(matrix)

        return prob, PO_vars_overall

    def build_FO_constraints(
            self, 
            prob,
            PO_vars_overall,
            obj_trace_PO_matrix_overall,
            obj_trace_FO_matrix_overall
        ):
        FO_vars_overall = []
        for trace_no, matrices in enumerate(obj_trace_PO_matrix_overall):
            FO_vars: Dict[TypedObject, Dict[tuple, pl.LpVariable]] = defaultdict(dict)
            for obj, PO_matrix in matrices.items():
                headers = PO_matrix.columns.tolist()
                FO_matrix = obj_trace_FO_matrix_overall[trace_no][obj]
                for i in range(len(PO_matrix)):
                    for j in range(len(PO_matrix)):
                        if i==j:
                            continue
                        current_PO = PO_vars_overall[trace_no].get(PO_matrix.iloc[i,j],PO_matrix.iloc[i,j])
                        if (pd.isna(FO_matrix.iloc[i,j])):
                            var_name = f"FO_{trace_no}_{str(obj.name)}{repr(headers[i])}{repr(headers[j])}"
                            var = pl.LpVariable(var_name, cat=pl.LpBinary, upBound=1, lowBound=0) 
                            FO_vars[obj][(headers[i], headers[j])] = var 
                            FO_matrix.iloc[i,j] = (headers[i], headers[j])

                            if isinstance(current_PO, pl.LpVariable):
                                prob += var <= current_PO # if PO == 0, then FO = 0; if PO==1, then FO = 1 or 0
                               
                            if i>j: # if var == 1, then transpose must be 0
                                transpose_var = FO_vars[obj].get(FO_matrix.iloc[j,i], FO_matrix.iloc[j,i])
                                # if(1-transpose_var != current_PO):
                                prob += var <= 1 - transpose_var
                                
                            # candidates = []
                            # for x in range(len(FO_matrix)):
                             
                            #     if (x!=i and x!=j):
                            #         ix = PO_vars_overall[trace_no].get(PO_matrix.iloc[i,x],PO_matrix.iloc[i,x])
                            #         xj = PO_vars_overall[trace_no].get(PO_matrix.iloc[x,j],PO_matrix.iloc[x,j])
                            #         if (isinstance(ix, pl.LpVariable) or isinstance(xj, pl.LpVariable)):
                            #             # aux = NAND(ix, xj)
                            #             aux = pl.LpVariable(f"x_{str(trace_no)}_{str(obj.name)}_{i}_{j}_{x}", cat=pl.LpBinary)
                            #             prob += aux <= 2-ix-xj
                            #             prob += aux >= ix-xj
                            #             prob += aux >= xj-ix
                            #             prob += aux <= ix+xj
                                      
                            #             # var <= aux
                            #             prob += var <= aux
                            #             candidates.append(aux)
                                        
                            # if (len(candidates)>0):
                            #     # var = 1 if all aux = 1
                            #     prob += var >= pl.lpSum(candidates) - len(candidates) + current_PO
                               
            FO_vars_overall.append(FO_vars)
        for trace_no, matrices in enumerate(obj_trace_FO_matrix_overall):
            for obj, FO_matrix in matrices.items():
                flatten = []
                FO_vars = FO_vars_overall[trace_no][obj]
                for m in range(len(FO_matrix)):
                    row = [FO_vars.get(FO_matrix.iloc[m,n],FO_matrix.iloc[m,n]) for n in range(len(FO_matrix)) if m!=n]
                    rowsum= pl.lpSum(row) # every row sum <= 1
                    if(not rowsum.isNumericalConstant()):
                        prob += rowsum <=1
                    
                    col = [FO_vars.get(FO_matrix.iloc[n,m] ,FO_matrix.iloc[n,m] ) for n in range(len(FO_matrix)) if m!=n]
                    colsum = pl.lpSum(col) # every col sum <= 1
                    if (not colsum.isNumericalConstant()):
                        prob+= colsum <=1
                    
                    flatten = flatten + row            
                prob += pl.lpSum(flatten) == len(FO_matrix)-1
        
        if self.debug['build_FO_constraints']:
            print("# Step 3")
            print("## FO vars")
            for trace_no, FO_vars in enumerate(FO_vars_overall):
                print(f"### Trace {trace_no}")
                pprint_table(FO_vars.items())

            print("## FO matrix with vars")
            for trace_no, matrices in enumerate(obj_trace_FO_matrix_overall):
                print(f"### Trace {trace_no}")
                for obj, matrix in matrices.items():
                    print(f"#### Obj {obj.name}")
                    pprint_table(matrix)

        return prob, FO_vars_overall

    def build_TM_constraints(
            self,  
            prob,
            sort_aps,
            sorts,
            FO_vars_overall,
            obj_trace_FO_matrix_overall
        ):
        _sort_transition_matrix: Dict[int, pd.DataFrame] = defaultdict()
        sort_transition_matrix: Dict[int, pd.DataFrame] = defaultdict()
        for sort, aps in sort_aps.items():
            cols = list(aps)
            transition_matrix = pd.DataFrame(columns=cols, index=cols)
            _sort_transition_matrix[sort] = transition_matrix.copy()
            sort_transition_matrix[sort] = transition_matrix.copy()
    
        for trace_no, matrices in enumerate(obj_trace_FO_matrix_overall):
            for obj, matrix in matrices.items():
                 cols = matrix.columns.tolist()
                 for i in range(len(matrix)):
                    for j in range(len(matrix) ):
                        if (i==j):
                            continue
                        from_ap = cols[i].to_event()
                        to_ap = cols[j].to_event()
                        if (pd.isna(_sort_transition_matrix[sorts[obj.name]].at[from_ap, to_ap])):
                            _sort_transition_matrix[sorts[obj.name]].at[from_ap, to_ap] = set()
                        elif (_sort_transition_matrix[sorts[obj.name]].at[from_ap, to_ap] == 1):
                            continue
                        if (isinstance(matrix.iloc[i,j], tuple)):
                            _sort_transition_matrix[sorts[obj.name]].at[from_ap, to_ap].add(FO_vars_overall[trace_no][obj].get(matrix.iloc[i,j],matrix.iloc[i,j]))
                        elif(matrix.iloc[i,j] == 1):
                            _sort_transition_matrix[sorts[obj.name]].at[from_ap, to_ap] = 1
 
        sort_AP_vars: Dict[int, Dict[Tuple[SingletonEvent,SingletonEvent], pl.LpVariable]] = defaultdict(dict)
     
        for sort, matrix in sort_transition_matrix.items():
            
            for row_header, rows in matrix.iterrows():
                for col_header, val in rows.items():
                    candidates = _sort_transition_matrix[sort].at[row_header, col_header]
                    if (isinstance(candidates, set)):
                        if (len(candidates)>0):
                            var_name = "AP_" + repr(row_header) + "_" + repr(col_header)
                            var = pl.LpVariable(var_name, cat=pl.LpBinary, upBound=1, lowBound=0) 
                            sort_AP_vars[sort][(row_header, col_header)] = var
                            matrix.at[row_header, col_header] = (row_header, col_header)

                            for FO_var in candidates:
                                prob += var>= FO_var # exist one
                        
                        else:
                            matrix.at[row_header, col_header] = np.nan
                    else:
                        matrix.at[row_header, col_header] = _sort_transition_matrix[sort].at[row_header, col_header]
      
        if self.debug['build_TM_constraints']:
            print("# Step 4")
            print("## AP vars")
            pprint_table(sort_AP_vars.items())                

            print("## AP matrix with vars")
            for sort, matrix in sort_transition_matrix.items():
                print(f"### Sort {sort}")
                pprint_table(matrix.values)
                    
        return prob, sort_transition_matrix, sort_AP_vars

    def bip_solve_PO(self, prob,
        solver,
        sort_AP_vars,):
        prob += pl.lpSum(var for var_list in sort_AP_vars.values() for var in var_list.values())

        try:
            prob.solve(solver)
        except Exception as e:
            raise Exception("Invalid MLP task")
        solution = {var.name: var.varValue for var in prob.variables()}
        if self.debug['bip_solve_PO']:

            print("Status: ", pl.LpStatus[prob.status])
            print()

            print("Solution:")
            print(solution)
            print()

            print("Objectives:")
            print(pl.value(prob.objective))
            print()
        
        return solution

    def get_obj_traces(self, obj_trace_PO_matrix_overall,
        PO_vars_overall,
        obj_trace_FO_matrix_overall,
        FO_vars_overall,
        sort_transition_matrix,
        AP_vars_overall,
        solution,):

        sol_PO_matrix = obj_trace_PO_matrix_overall.copy()
        obj_traces_list = []
        for trace_no, matrices, in enumerate(obj_trace_PO_matrix_overall):
            obj_traces: Dict[TypedObject, List[SingletonEvent]] = defaultdict(list)
            for obj,PO_matrix in matrices.items():
                sol = sol_PO_matrix[trace_no][obj]
                for i in range(len(PO_matrix)):
                    for j in range(len(PO_matrix)):
                        if(isinstance(PO_matrix.iloc[i,j], tuple)):
                            var = PO_vars_overall[trace_no].get(PO_matrix.iloc[i,j])

                            sol_var = solution[var.name]
                            sol.iloc[i,j] = sol_var
                sorted_header = sol.sum(axis=1).sort_values(ascending=False).index.tolist()
                obj_traces[obj] = [iap.to_event() for iap in sorted_header]
            obj_traces_list.append(obj_traces)
        
        sol_AP_matrix = sort_transition_matrix.copy()
        for sort, matrix in sort_transition_matrix.items():
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if(isinstance(matrix.iloc[i,j], tuple)):
                        var = AP_vars_overall[sort].get(matrix.iloc[i,j])
                       
                        sol_var = solution[var.name]
                        sol_AP_matrix[sort].iloc[i,j]= sol_var
        if self.debug['get_obj_traces']:
            sol_FO_matrix = obj_trace_FO_matrix_overall.copy()
            for trace_no, matrices in enumerate(obj_trace_FO_matrix_overall):
                for obj, FO_matrix in matrices.items():
                    sol = sol_FO_matrix[trace_no][obj]
                    for i in range(len(FO_matrix)):
                        for j in range(len(FO_matrix)):
                            if(isinstance(FO_matrix.iloc[i,j], tuple)):
                                var = FO_vars_overall[trace_no][obj].get(FO_matrix.iloc[i,j])

                                sol_var = solution[var.name]
                                sol.iloc[i,j] = sol_var
            print("## Solution FO matrix")
            for trace_no, matrices in enumerate(sol_FO_matrix):
                print(f"### Trace {trace_no}")
                for obj, matrix in matrices.items():
                    print(f"#### Obj {obj.name}")
                    pprint_table(matrix)

            print("## Solution AP matrix")
            for sort, matrix in sol_AP_matrix.items():
                print(f"### Sort {sort}")
                pprint_table(matrix)

        grouped_obj_traces = defaultdict(list)
        for obj_traces in obj_traces_list:
            for obj, seq in obj_traces.items():
                grouped_obj_traces[obj].append(seq)

        TM_list = []
        for sort, matrix in sorted(sol_AP_matrix.items()):
            TM = matrix.infer_objects(copy=False).fillna(0).astype(int)
            TM_list.append(TM)



        return grouped_obj_traces, TM_list



    
            