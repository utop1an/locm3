from .ocm import  TypedObject, SingletonEvent, OCM
from typing import Dict, List, Tuple, Set
from traces import  IndexedEvent
from collections import defaultdict
import pandas as pd
import networkx as nx
from utils.helpers import pprint_table

class POBASELINE(OCM):

    def extract_model():
        pass

    def solve_po(self, po_trace_list, sorts):
        obj_PO_matrix_list, obj_PO_trace_list = self.get_PO_matrix(po_trace_list, sorts)
        obj_consecutive_transitions_list, obj_traces  = self.baseline_solve_po(obj_PO_matrix_list)
        # obj_traces = self.find_obj_traces(obj_consecutive_transitions_list, obj_PO_trace_list)
        TM_list = self.get_learned_TM_list(obj_consecutive_transitions_list, sorts)
        return obj_traces, TM_list

    def get_PO_matrix(self, PO_trace_list, sorts):
        """
        Build PO matrices
        """
        # TODO: add zero obj
        zero_obj = OCM.ZEROOBJ
        # obj_traces for all obs_traces in obs_tracelist, indexed by trace_no
        obj_PO_trace_list = []
        trace_PO_matrix_list = []
        for PO_trace in PO_trace_list:
            indexed_actions = [IndexedEvent(step.action, step.index, None, None) for step in PO_trace]
            traces_PO_matrix = pd.DataFrame(columns=indexed_actions, index=indexed_actions)
            # collect action sequences for each object
            obj_PO_trace: Dict[TypedObject, List[IndexedEvent]] = defaultdict(list)
            for i, PO_step in enumerate(PO_trace):
                
                action = PO_step.action
                current_iap = indexed_actions[i]
                if action is not None:
                    zero_iap = IndexedEvent(action, PO_step.index ,0, 0)
                    obj_PO_trace[zero_obj].append(zero_iap)
                    # for each combination of action name A and argument pos P
                    added_objs = []
                    for k, obj in enumerate(action.obj_params):
                        # create transition A.P
                        assert obj not in added_objs, "LOCMv1 cannot handle duplicate objects in the same action."

                        iap = IndexedEvent(action, PO_step.index, k + 1, sorts[obj.name])
                        obj_PO_trace[obj].append(iap)
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
            trace_PO_matrix_list.append(traces_PO_matrix) 
            obj_PO_trace_list.append(obj_PO_trace)
                            
    
        # Constructing PO matrix of actions for each object in each trace
        obj_PO_matrix_list = []

        for trace_no,obj_PO_trace in enumerate(obj_PO_trace_list):
            obj_PO_matrices: Dict[TypedObject, pd.DataFrame] = defaultdict()
            for obj, iaps in obj_PO_trace.items():
                obj_trace_PO_matrix = pd.DataFrame(columns=iaps, index=iaps)    
                for row_header, row in obj_trace_PO_matrix.iterrows():
                    for col_header, val in row.items():
                        origin = trace_PO_matrix_list[trace_no].at[row_header.to_indexed_action(),
                                                                      col_header.to_indexed_action()]
                        obj_trace_PO_matrix.at[row_header, col_header] = origin
                
                obj_PO_matrices[obj] = obj_trace_PO_matrix
             
            obj_PO_matrix_list.append(obj_PO_matrices)
        

        if self.debug['get_PO_matrix']:
            for i, obj_PO_traces in enumerate(obj_PO_trace_list):
                print(f"### Trace **{i}**\n")
                pprint_table(obj_PO_traces.items())
            
        return obj_PO_matrix_list, obj_PO_trace_list
    
    def baseline_solve_po(self, obj_PO_matrix_list):
        obj_consecutive_transitions_list = []
        for trace_no, matrices, in enumerate(obj_PO_matrix_list):
            obj_consecutive_transitions: Dict[TypedObject, List[Tuple[SingletonEvent, SingletonEvent]]] = defaultdict(list)
            for obj,PO_matrix in matrices.items():
                if (len(PO_matrix) == 1):
                    obj_consecutive_transitions[obj].append((PO_matrix.columns[0],None))
                for i in range(len(PO_matrix)):
                    for j in range(len(PO_matrix)):
                        if (i==j):
                            continue
                        if (PO_matrix.iloc[i,j] == 0):
                            continue
                        if (PO_matrix.iloc[i,j] == 1):
                            flag = False
                            for k in range(len(PO_matrix)):
                                if (k==i or k==j):
                                    continue
                                if (PO_matrix.iloc[i,k] == 1 and PO_matrix.iloc[k,j] == 1):
                                    flag = True
                                    break
                            if not flag:
                                obj_consecutive_transitions[obj].append((PO_matrix.columns[i],PO_matrix.columns[j]))
                        else:
                            obj_consecutive_transitions[obj].append((PO_matrix.columns[i],PO_matrix.columns[j]))
            
            for obj, transitions in obj_consecutive_transitions.items():
                # Remove duplicates while preserving order
                seen = set()
                unique = []
                for transition in transitions:
                    if transition not in seen:
                        seen.add(transition)
                        unique.append(transition)
                obj_consecutive_transitions[obj] = unique
            obj_consecutive_transitions_list.append(obj_consecutive_transitions)
        
        grouped_obj_traces = defaultdict(list)
        for obj_traces in obj_consecutive_transitions_list:
            for obj, seqs in obj_traces.items():
                for pair in seqs:
                    e1, e2 = pair
                    if (e2 is None):
                        seq = [e1.to_event()]
                    else:
                        seq = [e1.to_event(), e2.to_event()]
                    grouped_obj_traces[obj].append(seq)

        return obj_consecutive_transitions_list, grouped_obj_traces

        # return obj_consecutive_transitions_list
    

    def find_obj_traces(self, obj_consecutive_transitions_list, obj_PO_trace_list):
        possible_traces = defaultdict(list)
        for trace_no, obj_consecutive_transitions in enumerate(obj_consecutive_transitions_list):
            obj_PO_trace = obj_PO_trace_list[trace_no]

            for obj, transitions in obj_consecutive_transitions.items():
                events_set = set(obj_PO_trace[obj])
                if len(events_set) == 1:
                    # If only one event, add it as a trace
                    possible_traces[obj].append([events_set.pop().to_event()])
                    continue

                graph = defaultdict(list)
                in_degree = defaultdict(int)

                for e1, e2 in transitions:
                    if e2 is None:
                        graph[e1]  # e1 is a terminal event
                        in_degree[e1] +=0
                    if e1 in events_set and e2 in events_set:
                        graph[e1].append(e2)
                        in_degree[e2] += 1
                
                sources = [e for e in events_set if in_degree[e] == 0]
                if not sources:
                    sources = obj_PO_trace[obj]
              

                def dfs(path, visited):
                    if len(path) == len(events_set):
                        possible_traces[obj].append([e.to_event() for e in path])
                        return
                    last = path[-1]
                    for nxt in graph[last]:
                        if nxt not in visited:
                            visited.add(nxt)
                            path.append(nxt)
                            dfs(path, visited)
                            path.pop()
                            visited.remove(nxt)
                # def dfs(path, visited):
                #     extended = False
                #     last = path[-1]
                #     for nxt in graph[last]:
                #         if nxt not in visited:
                #             visited.add(nxt)
                #             path.append(nxt)
                #             dfs(path, visited)
                #             path.pop()
                #             visited.remove(nxt)
                #             extended = True
                #     if not extended: 
                #         possible_traces[obj].append([e.to_event() for e in path])
                
                for source in sources:
                    dfs([source], {source})
        
        for obj, traces in possible_traces.items():
            # Deduplicate by converting to set of tuples
            unique_traces = list({tuple(trace) for trace in traces})
            possible_traces[obj] = [list(t) for t in unique_traces]

        return possible_traces
        

    def get_learned_TM_list(self, obj_consecutive_transitions_list, sorts):
        
        graphs = []
        for sort in range(len(set(sorts.values()))):
            graphs.append(nx.DiGraph())
        
        for obj_consecutive_transitions in obj_consecutive_transitions_list:
            for obj, transition in obj_consecutive_transitions.items():
                for iap1, iap2 in transition:
                    graphs[sorts[obj.name]].add_node(iap1.to_event())
                    if iap2:
                        graphs[sorts[obj.name]].add_node(iap2.to_event())
        
        # adjacent matrix list for all sorts
        for obj_consecutive_transitions in obj_consecutive_transitions_list:
            for obj, transition in obj_consecutive_transitions.items():
                sort = sorts[obj.name] if obj.name!='zero' else 0
                for iap1, iap2 in transition:
                    if not iap2:
                        continue
                    if (graphs[sort].has_edge(iap1.to_event(), iap2.to_event())):
                        graphs[sort][iap1.to_event()][iap2.to_event()]['weight']+=1
                    else:
                        graphs[sort].add_edge(iap1.to_event(),iap2.to_event(),weight=1)
        
        TM_list = []
        for index, G in enumerate(graphs):
            df = nx.to_pandas_adjacency(G, nodelist=G.nodes(), dtype=int)
            TM_list.append(df)
            if self.debug['get_obj_trace']:
                print("Sort.{} AML:".format(index))
                pprint_table(df)

        return TM_list