import random
from pddl_parser.pddl_file import open
from translate import pddl_to_sas
from translate.normalize import normalize
from collections import Counter
from utils import set_timer_throw_exc, GeneralTimeOut



class InvalidPlanner:
    def __init__(self, domain, problem=None, num_traces=10):
        # Store only the file paths and minimal configuration
        self.domain = domain
        self.problem = problem
        self.num_traces = num_traces


        try: 
            self.initialize_task()
        except Exception as e:
            raise Exception(f"Invalid model {domain}, {problem}, {e}")


    def initialize_task(self):
        """Initialize the task from domain and problem files."""
        task = open(self.domain, self.problem)
        normalize(task)
        self.task = task
        self.sas_task = pddl_to_sas(task)

 


    def generate_traces(self, prefix_trace, duplicate_actions, new_actions):
        assert duplicate_actions or new_actions, "At least one of duplicate_actions or new_actions must be True"
    
        traces = []
        prefix_ops = set()
        added_ops = set()

        # apply prefix trace
        state = tuple(self.sas_task.init.values)
        for action in prefix_trace:
            op = self.action_to_op(action)
            assert op is not None, f"Action {action} not found in domain"
            assert self.isApplicable(op, state), f"Action {action} is not applicable in state {state}"
            state = self.apply_operator(state, op)
            prefix_ops.add(op)

        for i in range(self.num_traces):
            if duplicate_actions:
                invalid_action = self.find_invalid_action(prefix_ops, added_ops, state, True, False)
                if not invalid_action:
                    break
                new_trace = prefix_trace + [invalid_action]
                traces.append(new_trace)

            if new_actions:
                invalid_action = self.find_invalid_action(prefix_ops, added_ops, state, False, True)
                if not invalid_action:
                    break
                new_trace = prefix_trace + [invalid_action]
                traces.append(new_trace)
                
            
        return traces
    
    def generate_invalid_suffixes(self, prefix_trace, duplicate_actions=True, new_actions=True):
        assert duplicate_actions or new_actions, "At least one of duplicate_actions or new_actions must be True"

        #suffixes = []
        # added_ops = set()
        prefix_ops = set()
        

        # apply prefix trace
        state = tuple(self.sas_task.init.values)
        for action in prefix_trace:
            op = self.action_to_op(action)
            assert op is not None, f"Action {action} not found in domain"
            assert self.isApplicable(op, state), f"Action {action} is not applicable in state {state}"
            state = self.apply_operator(state, op)
            prefix_ops.add(op)

        # for i in range(self.num_traces):
        #     if duplicate_actions:
        #         invalid_action = self.find_invalid_action(prefix_ops, added_ops, state, True, False)
        #         if not invalid_action:
        #             break
                
        #         suffixes.append(invalid_action)

        #     if new_actions:
        #         invalid_action = self.find_invalid_action(prefix_ops, added_ops, state, False, True)
        #         if not invalid_action:
        #             break
        #         suffixes.append(invalid_action)
        inapplicable_ops = self.get_inapplicable_operators(state)
        if not inapplicable_ops:
            return []
        suffixes = random.sample(inapplicable_ops, min(len(inapplicable_ops), self.num_traces))
                
        return [self.op_to_action(op) for op in suffixes]


    
    def find_invalid_action(self, prefix_ops, added_ops ,state, duplicate_action, new_action):
        if duplicate_action:
            candidate_ops = self.get_applicable_operators(state)
            if not candidate_ops:
                return None
            random.shuffle(candidate_ops)
            # Find a duplicate action from prefix_actions that is not in added_actions
            for op in prefix_ops:
                if op not in candidate_ops and op not in added_ops:
                    action = self.op_to_action(op)
                    added_ops.add(op)
                    return action
        if new_action:
            candidate_ops = self.get_inapplicable_operators(state)
            if not candidate_ops:
                return None
            random.shuffle(candidate_ops)
            # Find a new action that is not in prefix_actions or added_actions
            for op in candidate_ops:
                if op not in prefix_ops and op not in added_ops:
                    action = self.op_to_action(op)
                    added_ops.add(op)
                    return action

        return None
    
    def isApplicable(self, op, state):
        conditions = op.get_applicability_conditions()
        applicable = all(state[var] == val for var, val in conditions)
        return applicable
    
    # Convert action string to operator object
    def action_to_op(self, action):
        action = action.strip().strip("()").split(" ")
        action_name = action[0]
        obj_names = [ obj.split("?")[0] for obj in action[1:] ]  # Extract object names without types
        op_name = f"({action_name} {' '.join(obj_names)})"
        return self.sas_task.get_operator(op_name)

    # Convert operator object to action string with obj types
    def op_to_action(self, op):
        action = op.name.strip().strip("()").split(" ")
        action_name = action[0]
        o = self.task.get_action(action_name)
        assert o is not None, f"Action {action_name} not found in domain"

        args = action[1:]
        arg_types = [p.type_name for p in o.parameters]
        arg_with_types = [arg + "?"+ t for arg, t in zip(args, arg_types)]
        new_action = f"({action_name} {' '.join(arg_with_types)})"
        return new_action


    def get_applicable_operators(self, state):
        """Get all operators applicable in the current state."""
        ops = []
        for op in self.sas_task.operators:
            if self.isApplicable(op, state):
                ops.append(op)
        return ops
    
    
    def get_inapplicable_operators(self, state):
        """Get all inapplicable operators in the current state."""
        ops = []
        for op in self.sas_task.operators:
            if not self.isApplicable(op, state):
                ops.append(op)
        return ops


    def apply_operator(self, state, op):
        """Apply an operator to a given state and return the new state."""
        new_state = list(state)
        for var, _, post, _ in op.pre_post:
            new_state[var] = post
        return tuple(new_state)
    
