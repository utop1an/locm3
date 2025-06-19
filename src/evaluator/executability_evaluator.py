
from collections import defaultdict
import random
from typing import Literal

from pddl import ActionSignature, TypedObject
from pddl_parser import open
from translate.invariant_finder import find_invariants

from traces import Step



random.seed(42)

class ExecutabilityEvaluator:
    def __init__(
            self,
            learned_domain, 
            gt_domain_filename=None, 
            invariant_check=False,
            debug=False
        ) -> None:
        self.learned_domain = learned_domain
        self.gt_domain_filename = gt_domain_filename
        self.invariant_check = invariant_check
        self.debug = debug
        try:
            if (gt_domain_filename):
                self.gt_domain =  open(self.gt_domain_filename)
            if invariant_check:
                self.parse_invariants()
                    
        except Exception as e:
            raise Exception(f"Initialize Evaluator failed: {e}")
        

    def parse_invariants(self):
        self.l_invariants = find_invariants(self.learned_domain, None)
        if self.gt_domain:
            self.gt_invariants = find_invariants(self.gt_domain, None)

    def check_invariants(self,domain_type, true_effs, adds, dels):
        """
        Check if the true effects satisfy the invariants
        """
        if domain_type == 'l':
            invariants = self.l_invariants
        elif domain_type == 'gt':
            invariants = self.gt_invariants
        else:
            raise Exception("Invalid domain")
        
        def get_possible_violations(atom):
            res = set()
            for invariant in invariants:
                for part in invariant.parts:
                    if part.predicate == atom.predicate:
                        res = res.union(set(x for x in invariant.parts if x!=part))
                    break
            return res
        
        def get_violations(part, atom):
            res = set()
            order = part.order
            omitted_pos = part.omitted_pos
            for eff in true_effs:
                if eff.predicate != part.predicate:
                    continue
                flag = True
                for pos in order:
                    if eff.args[pos] != atom.args[pos]:
                        flag = False
                        continue
                if flag:
                    res.add(eff)

            return res
        
        def check_valid(possible_violations, atom):
            for part in possible_violations:
                #... check if the true effs violates
                violations = get_violations(part, atom)
                 # if violates, check if its deleted
                if len(violations) > 0:
                    if not violations.issubset(dels):
                        return False
            return True

        # for each atom in adds, check if it satisfies the invariants against the true effects
        for atom in adds:
            possible_violations = get_possible_violations(atom)
            valid = check_valid(possible_violations, atom)
            if not valid:
                return False
                
        return True
        

    def get_acceptance_rate(self, valid_seq, invalid_suffixes=None):
        if not self.learned_domain:
            raise Exception("Domain not initialized")
        
        valid_acceptance = self.get_first_fail_executability('l',valid_seq, set(), set(),None, True )
     
        if invalid_suffixes:
            invalid_acceptance = self.get_first_fail_executability('l', valid_seq, set(), set(), invalid_suffixes, True)
     
        else:
            invalid_acceptance = 0
        return valid_acceptance, invalid_acceptance

    def get_balanced_executability(self, valid_seqs, invalid_seqs,binary=False, debug=False):
        if not self.learned_domain:
            raise Exception("Domain not initialized")
        
        assert len(valid_seqs) > 0, f"Valid seqs should not be empty"
        assert len(invalid_seqs) > 0, f"Invalid seqs should not be empty"
        
        valid_res = []
        invalid_res = []

        for i in range(len(valid_seqs)):
            exe = self.get_overall_executability('l',valid_seqs[i],set(),set() )
            valid_res.append(exe)
        for j in range(len(invalid_seqs)):
            exe = self.get_first_fail_executability('l',invalid_seqs[j], set(),set(),binary)
            
            if binary:
                invalid_res.append(1-exe)
            else:
                invalid_gt_res = (1-1/len(invalid_seqs[j]))
                assert invalid_gt_res != 0, f"Invalid gt seqs should not be empty"
                invalid_res.append(exe/invalid_gt_res)

        valid_exe = sum(valid_res)/len(valid_res)
        invalid_exe = sum(invalid_res)/len(invalid_res)
        return valid_exe, invalid_exe
        

    
        
    def get_cross_executability(self,prefix_seqs, gt_seqs=[], gen_seq_rate=None, debug=False):
        """
        Check the cross executability of the gt seqs on learned domain and learned seqs on gt domain

        Args:
            prefix_action_sequence (list): list of initial action seq, if given, it will be used to generate the action seqs
            gt_seqs (list): list of gt action sequences
            invalid_gt_seqs (list): list of invalid gt action sequences
            debug (bool, optional): whether to print debug information. Defaults to False.
        """
        if not self.learned_domain:
            raise Exception("Domain not initialized")
        if not self.gt_domain_filename:
            raise Exception("GT Domain not given")
        
        
        
        if prefix_seqs and not gt_seqs:

            l_items= self.generate_action_seqs('l', prefix_seqs)
            length_limits = [l for _,_,_,l in l_items]
            gt_items = self.generate_action_seqs('gt', prefix_seqs, limit= length_limits)

            l_res = []
            gt_res = []
            for i in range(len(l_items)):
                l_seqs, l_init, l_visited, length = l_items[i]
                gt_seqs, gt_init, gt_visited, _ = gt_items[i]

                for j in range(len(l_seqs)):
                    exe_on_learned = self.get_overall_executability('l', gt_seqs[j], l_init, l_visited)
                    exe_on_gt = self.get_overall_executability('gt', l_seqs[j], gt_init, gt_visited)
                
                    l_res.append(exe_on_learned)
                    gt_res.append(exe_on_gt)
            
            return sum(l_res)/len(l_res), sum(gt_res)/len(gt_res)

        if prefix_seqs and gt_seqs:

            l_items = self.generate_action_seqs('l', prefix_seqs, gen_seq_rate = gen_seq_rate)

            l_res = []
            gt_res = []
            for i in range(len(l_items)):
                l_seqs, l_init, l_visited, _ = l_items[i]
                for j in range(len(l_seqs)):
                    exe_on_gt = self.get_overall_executability('gt', l_seqs[j], set(), set())
                    gt_res.append(exe_on_gt)

            for k in range(len(gt_seqs)):
                exe_on_learned = self.get_overall_executability('l', gt_seqs[k], set(), set())
                l_res.append(exe_on_learned)
            
            
            return sum(l_res)/len(l_res), sum(gt_res)/len(gt_res)


    
    def generate_action_seqs(self, domain_type, prefix_seqs, limits=None, gen_seq_rate=None):
        if domain_type == 'l':
            domain = self.learned_domain
        elif domain_type == 'gt':
            domain = self.gt_domain
        else:
            raise Exception("Invalid domain")
        
        res = []
        for index, prefix_seq in enumerate(prefix_seqs):

            if gen_seq_rate:
                if gen_seq_rate * len(prefix_seq) < 1:
                    continue
            true_effs = set()
            type_objs = defaultdict(set)

            # as we don't know the init states, we can only record the visited effects
            # we assume unvisited effects are true since all given action seqs are valid
            visited = set()
            for i, a in enumerate(prefix_seq):

                if limits and i > limits[index]:
                    break

                if gen_seq_rate and i == gen_seq_rate * len(prefix_seq):
                    break

                if isinstance(a, Step):
                    a = a.action
                
                
                action = domain.get_action(a.name)
                
                # if action not found, meaning it has not been learned properly, with no precond or effects
                # we skip it, and add error count by 1
                if not action:
                    break
                param_names = [p.name for p in action.parameters]
                param_types = [p.type_name for p in action.parameters]
                params = [obj.name for obj in a.obj_params]
                
            
                if ('s0' in param_types):
                    index= param_types.index('s0')
                    params.insert(index, 'zero')
                elif ('zero' in param_types):
                    index= param_types.index('zero')
                    params.insert(index, 'zero')
            
                var_mapping = dict(zip(param_names, params))
                objects_by_type = dict(zip(params, param_types))
            
                op = action.instantiate(var_mapping,None, None,None, objects_by_type,None)

                for (name, type) in objects_by_type.items():
                    type_objs[type].add(name)


                # check applicable
                preconditions = set(op.precondition)
                invalid = preconditions.difference(true_effs)
                invalid = invalid.intersection(visited)
                # not applicable
                if(len(invalid)>0):
                    break

                # apply action
                adds = set(e for _,e in op.add_effects)
                dels = set(e for _,e in op.del_effects)
                
                # mark visited effects
                visited = visited.union(adds).union(dels)

                true_effs = true_effs.union(adds)
                true_effs.difference_update(dels)


            new_action_sequences = self.generate_action_seq(domain_type, type_objs, true_effs, visited, len(prefix_seq)-i)
            res.append((new_action_sequences, true_effs, visited, i))
        return res

    def generate_action_seq(self,domain_type, type_objs, init_effs, init_visited, length, debug=False):
        if domain_type == 'l':
            domain = self.learned_domain
        elif domain_type == 'gt':
            domain = self.gt_domain
        else:
            raise Exception("Invalid domain")
        
        grounded_actions = domain.get_grounded_actions(type_objs)
        if debug:
            print("number of grounded actions:", len(grounded_actions))
            print("Grounded actions:", grounded_actions)
        
        def get_applicable_actions(_effs, _visited):
            applicable_actions = []
            for action in grounded_actions:
                preconditions = set(action.precondition)
                invalid = preconditions.difference(_effs)
                invalid = invalid.intersection(_visited)
                if (len(invalid)==0):
                    if self.invariant_check:
                        self.check_invariants(
                            domain_type, 
                            _effs,
                            set(e for _,e in action.add_effects),
                            set(e for _,e in action.del_effects)
                            )
                    applicable_actions.append(action)
            return applicable_actions
        
        
        plans = []
        for _ in range(1):
            plan = []
            true_effs = init_effs.copy()
            visited = init_visited.copy()
            for i in range(length):
                candiates = get_applicable_actions(true_effs, visited)
                if (len(candiates) == 0):
                    break
                action = random.choice(candiates)

                

                adds = set(e for _,e in action.add_effects)
                dels = set(e for _,e in action.del_effects)
                
                # mark visited effects
                visited = visited.union(adds).union(dels)

                true_effs = true_effs.union(adds)
                true_effs.difference_update(dels)

                a = action.name.strip().strip("()").split(" ")
                action_name = a[0]
                args = a[1:]

                params = [TypedObject(obj, "na") for obj in args if obj != 'zero'] 
                plan.append(ActionSignature(action_name, params))

            plans.append(plan)
        if debug:
            print("New action sequence:", plan)
    
        if debug:
            print(plans)
        return plans

    
    def get_overall_executability(self,domain_type, action_sequence, _true_effs, _visited):
        if domain_type == 'l':
            domain = self.learned_domain
        elif domain_type == 'gt':
            domain = self.gt_domain
        else:
            raise Exception("Invalid domain")
        if not domain:
            raise Exception("Domain not given")
        
        if (not action_sequence):
            return 0
        
        true_effs = _true_effs.copy()
        # as we don't know the init states, we can only record the visited effects
        # we assume unvisited effects are true since all given action seqs are valid
        visited = _visited.copy()
        error_count = 0
        for i, a in enumerate(action_sequence):
            if isinstance(a, Step):
                a = a.action
            action = domain.get_action(a.name)
            # if action not found, meaning it has not been learned properly, with no precond or effects
            # we skip it, and add error count by 1
            if not action:
                error_count += 1
                continue
            param_names = [p.name for p in action.parameters]
            param_types = [p.type_name for p in action.parameters]
            params = [obj.name for obj in a.obj_params]
            if ('s0' in param_types):
                index= param_types.index('s0')
                params.insert(index, 'zero')
            elif ('zero' in param_types):
                index= param_types.index('zero')
                params.insert(index, 'zero')
            var_mapping = dict(zip(param_names, params))
            
            objects_by_type = dict(zip(params, param_types))
            op = action.instantiate(var_mapping,None, None,None, objects_by_type,None)
            # check applicable
            preconditions = set(op.precondition)
            invalid = preconditions.difference(true_effs)
            invalid = invalid.intersection(visited)
            # not applicable
            if(len(invalid)>0):
                error_count += 1
                if self.debug:
                    print(f"action {op} not executable")
                    print("preconditions not satisfied: ", invalid)

            # apply action
            adds = set(e for _,e in op.add_effects)
            dels = set(e for _,e in op.del_effects)
            
            # mark visited effects
            visited = visited.union(adds).union(dels)

            true_effs = true_effs.union(adds)
            true_effs.difference_update(dels)
                
        return 1-error_count/len(action_sequence)

    
        
    def get_first_fail_executability(self,domain_type, action_sequence,  _true_effs, _visited, suffixes=None, binary=True):
        if domain_type == 'l':
            domain = self.learned_domain
        elif domain_type == 'gt':
            domain = self.gt_domain
        else:
            raise Exception("Invalid domain")
        if not domain:
            raise Exception("Domain not given")
        
        
        if not action_sequence:
            return 0
        true_effs = _true_effs.copy()
        visited = _visited.copy()
        
        for i, a in enumerate(action_sequence):
            if isinstance(a, Step):
                a = a.action
            action = domain.get_action(a.name)
            if not action:
                if binary:
                    return 0
                else:
                    i-=1
                    break

            param_names = [p.name for p in action.parameters]
            param_types = [p.type_name for p in action.parameters]
            params = [obj.name for obj in a.obj_params]
            if ('s0' in param_types):
                index= param_types.index('s0')
                params.insert(index, 'zero')
            elif ('zero' in param_types):
                index= param_types.index('zero')
                params.insert(index, 'zero')
            var_mapping = dict(zip(param_names, params))
            
            objects_by_type = dict(zip(params, param_types))
            op = action.instantiate(var_mapping,None, None,None, objects_by_type,None)
            # check applicable
            preconditions = set(op.precondition)
            invalid = preconditions.difference(true_effs)
            invalid = invalid.intersection(visited)

            # not applicable
            if(len(invalid)>0):
                if self.debug:
                    print(f"action {op} not executable")
                    print("preconditions not satisfied: ", invalid)
                if binary:
                    return 0
                else:
                    i-=1
                break
            
            # apply action
            adds = set(e for _,e in op.add_effects)
            dels = set(e for _,e in op.del_effects)
            
            # mark visited effects
            visited = visited.union(adds).union(dels)

            true_effs = true_effs.union(adds)
            true_effs.difference_update(dels)

        if not suffixes:
            return (i+1)/len(action_sequence)
        
        suffixes_res = []
        
        for suffix in suffixes:
            if isinstance(suffix, Step):
                suffix = suffix.action
            action = domain.get_action(suffix.name)
            if not action:
                if binary:
                    suffixes_res.append(0)
                    continue
                else:
                    suffixes_res.append((i+1)/(len(action_sequence)+1))
                    continue

            param_names = [p.name for p in action.parameters]
            param_types = [p.type_name for p in action.parameters]
            params = [obj.name for obj in suffix.obj_params]
            if ('s0' in param_types):
                index= param_types.index('s0')
                params.insert(index, 'zero')
            elif ('zero' in param_types):
                index= param_types.index('zero')
                params.insert(index, 'zero')
            var_mapping = dict(zip(param_names, params))
            
            objects_by_type = dict(zip(params, param_types))
            op = action.instantiate(var_mapping,None, None,None, objects_by_type,None)
            # check applicable
            preconditions = set(op.precondition)
            invalid = preconditions.difference(true_effs)
            invalid = invalid.intersection(visited)

            # not applicable
            if(len(invalid)>0):
                if self.debug:
                    print(f"action {op} not executable")
                    print("preconditions not satisfied: ", invalid)
                if binary:
                    suffixes_res.append(0)
                else:
                    suffixes_res.append((i+1)/(len(action_sequence)+1))
            else:
                suffixes_res.append(1)
        return sum(suffixes_res)/len(suffixes_res)
            
            


        
            




