import random
from pddl_parser.pddl_file import open
from translate import pddl_to_sas
from translate.normalize import normalize
from collections import Counter
from utils import set_timer_throw_exc, GeneralTimeOut


class RandomPlanner:
    def __init__(self, domain, problem, trace_len=10, num_traces=1, max_time=30):
        # Store only the file paths and minimal configuration
        self.domain = domain
        self.problem = problem
        self.trace_len = trace_len
        self.num_traces = num_traces
        self.max_time = max_time


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


    def generate_traces(self, is_valid_trace=True):
        if self.sas_task is None:
            # Make sure the task is initialized in the worker process
            self.initialize_task()

        traces = []
        visited = Counter()
        for i in range(self.num_traces):
            trace = self.generate_single_trace_setup(is_valid_trace, visited)
            if (not trace or len(trace) == 0):
                continue
            trace_with_type_info= self.add_type_info(trace)
            traces.append(trace_with_type_info)
        return traces

    
    

    def generate_single_trace_setup(self, is_valid_trace, visited):
        
        def weighted_random_choice(choices, visited, alpha = 0.6):
            weights = []
            for choice in choices:
                count = visited[choice]
                weights.append((1 / (1 + count)) ** alpha)

            if sum(weights) == 0:
                choosen = random.choice(choices)
            else:
                choosen = random.choices(choices, weights=weights, k=1)[0]
            visited[choosen] += 1
            return choosen
        

        @set_timer_throw_exc(num_seconds=self.max_time, exception=GeneralTimeOut, max_time=self.max_time)
        def generate_single_trace():
            found = False
            while not found:
                state = tuple(self.sas_task.init.values)
                trace = []
                for i in range(self.trace_len):
                    ops = self.get_applicable_operators(state)
                    if not ops or len(ops) == 0:
                        break
                    op = weighted_random_choice(ops, visited)
                    trace.append(op)
                    state = self.apply_operator(state, op)
                if len(trace) == self.trace_len:
                    found = True
                    if not is_valid_trace:
                        ops = self.get_inapplicable_operators(state)
                        if not ops or len(ops) == 0:  
                            raise Exception("No invalid trace found")
                        op = random.choice(ops)
                        trace.append(op)
            return trace
        
        try: 
            trace = generate_single_trace()
            return trace
        except Exception as e:
            print(f"Error occurred while generating trace: {e}")
            return []

    def get_applicable_operators(self, state):
        """Get all operators applicable in the current state."""
        ops = []
        for op in self.sas_task.operators:
            conditions = op.get_applicability_conditions()
            applicable = all(state[var] == val for var, val in conditions)
            if applicable:
                ops.append(op)
        return ops
    
    def get_inapplicable_operators(self, state):
        """Get all inapplicable operators in the current state."""
        ops = []
        for op in self.sas_task.operators:
            conditions = op.get_applicability_conditions()
            applicable = all(state[var] == val for var, val in conditions)
            if not applicable:
                ops.append(op)
        return ops


    def apply_operator(self, state, op):
        """Apply an operator to a given state and return the new state."""
        new_state = list(state)
        for var, _, post, _ in op.pre_post:
            new_state[var] = post
        return tuple(new_state)
    
  

    def add_type_info(self, trace):
        t = []
        for op in trace:
            action = op.name.strip().strip("()").split(" ")
            action_name = action[0]
            o = self.task.get_action(action_name)
            assert o is not None, f"Action {action_name} not found in domain"

            args = action[1:]
            arg_types = [p.type_name for p in o.parameters]
            arg_with_types = [arg + "?"+ t for arg, t in zip(args, arg_types)]
            new_action = f"({action_name} {' '.join(arg_with_types)})"
            t.append(new_action)
        return t
