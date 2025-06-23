from .ocm import TypedObject, Event, SingletonEvent, StatePointers, OCM
from typing import Dict, List, Tuple, Set, NamedTuple
from traces import Hypothesis, HIndex, HItem, FSM, PartialOrderedTrace, IndexedEvent, SingletonEvent
from pddl import ActionSignature, LearnedModel, LearnedLiftedFluent, LearnedAction
from collections import defaultdict
import pandas as pd
import numpy as np
import networkx as nx
import pulp as pl
from utils.helpers import pprint_table, complete_PO_np, complete_FO_np
from datetime import datetime
from pympler import asizeof

class POOPTIMISATION(OCM):

  
    def extract_model(self, PO_trace_list: List[PartialOrderedTrace], type_dict: Dict[str, int] = None) -> LearnedModel:
        pass
        

    def solve_po(self, PO_trace_list, sorts):
        if self.solver_path == 'default':
            self.solver = pl.PULP_CBC_CMD(msg=False, timeLimit=600)
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Launching CPLEX with {self.cores} cores")
            self.solver = pl.CPLEX_CMD(path=self.solver_path, msg=False, timeLimit=600, threads=self.cores, maxMemory=4096)
        
        try: 
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
        except Exception as e:
            print("Exception in PO optimisation: ", e)
            raise e
        return obj_traces_list, TM_list

    def get_matrices(self, PO_trace_list, sorts):
        # TODO: add zero obj
        zero_obj = OCM.ZEROOBJ
        
        sort_aps:Dict[int, Set[SingletonEvent]] = defaultdict(set)
        # obj_traces for all obs_traces in obs_tracelist, indexed by trace_no
        obj_PO_trace_list = []
        trace_PO_matrix_list = []
        for PO_trace in PO_trace_list:
            indexed_actions = [IndexedEvent(step.action, step.index, None, None) for step in PO_trace]
            trace_PO_matrix = np.full((len(indexed_actions), len(indexed_actions)), np.nan, dtype=object)
            # collect action sequences for each object
            obj_PO_traces: Dict[TypedObject, List[SingletonEvent|IndexedEvent]] = defaultdict(list)
            for i, PO_step in enumerate(PO_trace):
                
                action = PO_step.action
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
                        
                        trace_PO_matrix[i,j]=1
                        trace_PO_matrix[j,i]=0
                       
            complete_PO_np(trace_PO_matrix)
            trace_PO_matrix_list.append((indexed_actions, trace_PO_matrix)) 
            obj_PO_trace_list.append(obj_PO_traces)
        
    
        # Constructing PO matrix of actions for each object in each trace
        obj_PO_matrix_list = []
        # Initialize FO matrix
        obj_trace_FO_matrix_overall = []
        for trace_no,obj_PO_trace in enumerate(obj_PO_trace_list):
            ias = trace_PO_matrix_list[trace_no][0] # indexed actions
            PO_matrices: Dict[TypedObject, Tuple[any, np.array] ] = defaultdict()
            FO_matrices: Dict[TypedObject, Tuple[any, np.array] ] = defaultdict()
            for obj, iaps in obj_PO_trace.items():
                obj_trace_PO_matrix = np.full((len(iaps), len(iaps)),np.nan, dtype=object)
                obj_trace_FO_matrix = np.full((len(iaps), len(iaps)),np.nan, dtype=object)   

                for i in range(obj_trace_PO_matrix.shape[0]):
                    for j in range(obj_trace_PO_matrix.shape[1]):

                        i_action = iaps[i].to_indexed_action()
                        j_action = iaps[j].to_indexed_action()

                        action_i = ias.index(i_action)
                        action_j = ias.index(j_action)
                        origin = trace_PO_matrix_list[trace_no][1][action_i, action_j]

                        obj_trace_PO_matrix[i, j] = origin
                

                complete_PO_np(obj_trace_PO_matrix)
                PO_matrices[obj] = (iaps, obj_trace_PO_matrix)
                complete_FO_np(obj_trace_FO_matrix, obj_trace_PO_matrix)
                FO_matrices[obj] = (iaps, obj_trace_FO_matrix)  
            obj_PO_matrix_list.append(PO_matrices)
            obj_trace_FO_matrix_overall.append(FO_matrices)
        

        if self.debug['get_matrices']:
          
            print("## Obj PO traces:\n")
            for i, obj_PO_traces in enumerate(obj_PO_trace_list):
                print(f"### Trace **{i}**\n")
                print(obj_PO_traces.items())

            print("## Traces PO Matrix")
            for i, (header, matrix) in enumerate(trace_PO_matrix_list):
                print(f"### Trace **{i}**\n")
                print(matrix)
            
            print("## Object Traces PO Matrix")
            for i, matrices in enumerate(obj_PO_matrix_list):
                print(f"### Trace **{i}**\n")
                for obj, (header, matrix) in matrices.items():
                    print(f"#### Object ***{obj.name}***\n")
                    print(matrix)
            
            print("## Object Traces FO Matrix")
            for i, matrices in enumerate(obj_trace_FO_matrix_overall):
                print(f"### Trace {i}\n")
                for obj, (header, matrix) in matrices.items():
                    print(f"#### Object {obj.name}\n")
                    print(matrix)
        return trace_PO_matrix_list, obj_PO_matrix_list, obj_trace_FO_matrix_overall, sort_aps

    def build_PO_constraints(self,trace_PO_matrix_overall, obj_trace_PO_matrix_overall,):
        prob = pl.LpProblem("polocm", sense=pl.LpMinimize)
        PO_vars_overall= []
            
        for trace_no, (ias, matrix) in enumerate(trace_PO_matrix_overall):
            PO_vars = {}
            
           
            for i in range(matrix.shape[0]):
                for j in range(i+1, matrix.shape[1]):
                    if pd.isna(matrix[i,j]) or matrix[i,j] == np.nan:
                        var_name = f"t{trace_no}_{i}_{j}"
                        var = pl.LpVariable(var_name, cat=pl.LpBinary, upBound=1, lowBound=0) 
                        PO_vars[(ias[i], ias[j])] = var 
                        matrix[i,j] = (ias[i], ias[j])

                        transpose_var_name = f"tr{trace_no}_{i}_{j}"
                        transpose_var = pl.LpVariable(transpose_var_name, cat=pl.LpBinary, upBound=1, lowBound=0) 
                        PO_vars[(ias[j], ias[i])] = transpose_var 
                        matrix[j,i] = (ias[j], ias[i])

                        prob += var == 1 - transpose_var # (i,j) == not (j,i)
            
            PO_vars_overall.append(PO_vars)
            for i in range(matrix.shape[0]):
                for j in range(i+1, matrix.shape[1]):
                    current = PO_vars.get(matrix[i,j], matrix[i,j])
                    for x in range(matrix.shape[0]):
                        if (x==i or x ==j):
                            continue
                        _next = PO_vars.get(matrix[j,x], matrix[j,x])
                        target = PO_vars.get(matrix[i, x],matrix[i, x])
                        if (isinstance(current, pl.LpVariable) or isinstance(_next, pl.LpVariable)):
                            prob += target >= current + _next -1
                            prob += target <= current + _next
        
            
        
        for trace_no, matrices in enumerate(obj_trace_PO_matrix_overall):
            
            PO_vars = PO_vars_overall[trace_no]
            for obj, (iaps, matrix) in matrices.items():
                
                for i in range(matrix.shape[0]):
                    for j in range(i+1, matrix.shape[1]):

                        key = (iaps[i].to_indexed_action(), iaps[j].to_indexed_action())
                        if key in PO_vars.keys():
                            matrix[i, j] = key
                   

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
            for obj, (iaps, PO_matrix) in matrices.items():
                
                FO_matrix = obj_trace_FO_matrix_overall[trace_no][obj][1]
                for i in range(PO_matrix.shape[0]):
                    for j in range(PO_matrix.shape[1]):
                        if i==j:
                            continue
                        current_PO = PO_vars_overall[trace_no].get(PO_matrix[i,j],PO_matrix[i,j])
                        if (pd.isna(FO_matrix[i,j])):
                            var_name = f"f{trace_no}_{str(obj.name)}_{i}_{j}"
                            var = pl.LpVariable(var_name, cat=pl.LpBinary, upBound=1, lowBound=0) 
                            FO_vars[obj][(iaps[i], iaps[j])] = var 
                            FO_matrix[i,j] = (iaps[i], iaps[j])

                            if isinstance(current_PO, pl.LpVariable):
                                prob += var <= current_PO # if PO == 0, then FO = 0; if PO==1, then FO = 1 or 0
                               
                            if i>j: # if var == 1, then transpose must be 0
                                transpose_var = FO_vars[obj].get(FO_matrix[j,i], FO_matrix[j,i])
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
            for obj, (iaps, FO_matrix) in matrices.items():
                flatten = []
                FO_vars = FO_vars_overall[trace_no][obj]
                for m in range(FO_matrix.shape[0]):
                    row = [FO_vars.get(FO_matrix[m,n],FO_matrix[m,n]) for n in range(FO_matrix.shape[0]) if m!=n]
                    rowsum= pl.lpSum(row) # every row sum <= 1
                    if(not rowsum.isNumericalConstant()):
                        prob += rowsum <=1
                    
                    col = [FO_vars.get(FO_matrix[n,m] ,FO_matrix[n,m]) for n in range(FO_matrix.shape[0]) if m!=n]
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
        _sort_transition_matrix: Dict[int, Tuple[any, np.array]] = defaultdict(tuple)
        sort_transition_matrix: Dict[int, Tuple[any, np.array]] = defaultdict(tuple)
        for sort, _aps in sort_aps.items():
            aps = list(_aps)
            _sort_transition_matrix[sort] = (aps, np.full((len(aps), len(aps)),np.nan, dtype=object))
            sort_transition_matrix[sort] = (aps, np.full((len(aps), len(aps)),np.nan, dtype=object))
    
        for trace_no, matrices in enumerate(obj_trace_FO_matrix_overall):
            for obj, (iaps, matrix) in matrices.items():
                sort = sorts[obj.name]
                aps = _sort_transition_matrix[sort][0]
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        if (i==j):
                            continue
                        from_ap = iaps[i].to_event()
                        to_ap = iaps[j].to_event()

                        from_ap_i = aps.index(from_ap)
                        to_ap_j = aps.index(to_ap)

                        

                        if (pd.isna(_sort_transition_matrix[sort][1][from_ap_i, to_ap_j])):
                            _sort_transition_matrix[sort][1][from_ap_i, to_ap_j] = set()
                        elif (_sort_transition_matrix[sort][1][from_ap_i, to_ap_j] == 1):
                            continue
                        if (isinstance(matrix[i,j], tuple)):
                            _sort_transition_matrix[sort][1][from_ap_i, to_ap_j].add(FO_vars_overall[trace_no][obj].get(matrix[i,j],matrix[i,j]))
                        elif(matrix[i,j] == 1):
                            _sort_transition_matrix[sort][1][from_ap_i, to_ap_j] = 1
 
        sort_AP_vars: Dict[int, Dict[Tuple[SingletonEvent,SingletonEvent], pl.LpVariable]] = defaultdict(dict)
     
        for sort, (aps, matrix) in sort_transition_matrix.items():
            
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):

                    candidates = _sort_transition_matrix[sort][1][i, j]
                    if (isinstance(candidates, set)):
                        if (len(candidates)>0):
                            var_name = f"e_{sort}_{i}_{j}"
                            var = pl.LpVariable(var_name, cat=pl.LpBinary, upBound=1, lowBound=0) 
                            sort_AP_vars[sort][(aps[i], aps[j])] = var
                            matrix[i, j] = (aps[i], aps[j])

                            for FO_var in candidates:
                                prob += var>= FO_var # exist one
                        
                        else:
                            matrix[i, j] = np.nan
                    else:
                        matrix[i, j] = _sort_transition_matrix[sort][1][i, j]
      
        if self.debug['build_TM_constraints']:
            print("# Step 4")
            print("## AP vars")
            pprint_table(sort_AP_vars.items())                

            print("## AP matrix with vars")
            for sort, (header, matrix) in sort_transition_matrix.items():
                print(f"### Sort {sort}")
                print(f"#### Header: {header}")
                print(matrix)
                    
        return prob, sort_transition_matrix, sort_AP_vars

    def bip_solve_PO(self, prob,
        sort_AP_vars,):
        prob += pl.lpSum(var for var_list in sort_AP_vars.values() for var in var_list.values())

        max_allowed_size_bytes = int(3.2* 1024**3)  # 3.2< 4
        
        pympler_size = asizeof.asizeof(prob.variables()) + asizeof.asizeof(prob.constraints)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Estimated problem size: {pympler_size / (1024 ** 2):.2f} MB")
        if pympler_size > max_allowed_size_bytes:
            raise MemoryError("Problem too large")
        
        try:

            prob.solve(self.solver)
        except Exception as e:
            raise Exception("Invalid MLP task: "+ str(e))
        if pl.LpStatus[prob.status] != 'Optimal':
            raise Exception(f"Solver failed with status: {pl.LpStatus[prob.status]}")
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
        solution):

        
        obj_traces_list = []
        for trace_no, matrices, in enumerate(obj_trace_PO_matrix_overall):
            obj_traces: Dict[TypedObject, List[SingletonEvent]] = defaultdict(list)
            for obj,(iaps, PO_matrix) in matrices.items():
                sol = pd.DataFrame(columns=iaps, index=iaps)
                for i in range(PO_matrix.shape[0]):
                    for j in range(PO_matrix.shape[1]):
                        
                        if(isinstance(PO_matrix[i,j], tuple)):
                            var = PO_vars_overall[trace_no].get(PO_matrix[i,j])
                            sol_var = solution[var.name]
                            sol.iloc[i,j] = sol_var
                sorted_header = sol.sum(axis=1).sort_values(ascending=False).index.tolist()
                obj_traces[obj] = [iap.to_event() for iap in sorted_header]
            obj_traces_list.append(obj_traces)
        
        sol_AP_matrix: Dict[int, pd.DataFrame] = defaultdict()
        for sort, (aps, matrix) in sort_transition_matrix.items():
            sol_AP_matrix[sort] = pd.DataFrame(index=aps, columns=aps)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if(isinstance(matrix[i,j], tuple)):
                        var = AP_vars_overall[sort].get(matrix[i,j])
                       
                        sol_var = solution[var.name]
                        sol_AP_matrix[sort].iloc[i,j]= sol_var
                    else:
                        sol_AP_matrix[sort].iloc[i,j]= matrix[i,j]
        
        if self.debug['get_obj_traces']:
            # sol_FO_matrix = obj_trace_FO_matrix_overall.copy()
            # for trace_no, matrices in enumerate(obj_trace_FO_matrix_overall):
            #     for obj, (iaps, FO_matrix) in matrices.items():
            #         sol = sol_FO_matrix[trace_no][obj]
            #         for i in range(len(FO_matrix)):
            #             for j in range(len(FO_matrix)):
            #                 if(isinstance(FO_matrix[i,j], tuple)):
            #                     var = FO_vars_overall[trace_no][obj].get(FO_matrix[i,j])

            #                     sol_var = solution[var.name]
            #                     sol.iloc[i,j] = sol_var
            # print("## Solution FO matrix")
            # for trace_no, matrices in enumerate(sol_FO_matrix):
            #     print(f"### Trace {trace_no}")
            #     for obj, matrix in matrices.items():
            #         print(f"#### Obj {obj.name}")
            #         pprint_table(matrix)

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



    
            