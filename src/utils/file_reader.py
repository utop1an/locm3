"""
Read traces from different type of inputs.
Plans should be in the form of:
    without state info `(action1 param1)\n(action2 param1 param2)\n(action3 param2)...`
    with state info `(action param1), (predicate1 param1)\n(action2 param1 param2), (predicate2 param1 param2), (predicate3 param2)\n(action3 param2), (predicate4 param2)`
"""

from pddl import ActionSignature, TypedObject


def parse_action(action):
    a = action.strip().strip("()").split(" ")
    assert len(a)>0, "Invalid action string"
    action_name = a[0]
    if len(a) == 1:
        return ActionSignature(action_name, None)
    params = a[1:]
    objs = []
    for param in params:
        p = param.strip().split("?")
        obj_name = p[0]
        if len(p) == 1:
            objs.append(TypedObject(obj_name, "null"))
        elif len(p) == 2:
            objs.append(TypedObject(obj_name, p[1]))
        else:
            #TODO: Type distribution
            raise NotImplementedError
    return ActionSignature(action_name, objs)


def parse_state(predicates):
    raise NotImplementedError
    fluents = []
    for predicate in predicates:
        pred = predicate.strip().strip("()").split(" ")
        fluent_name = pred[0]
        params = pred[1:]
        for param in params:
            p = param.strip().split("?")

    return State()

def read_step(step, index):
    from traces import Step
    tmp = step.split(', ')
    action = parse_action(tmp[0])
    if len(tmp) >1:
        state = parse_state(tmp[1:])
    else:
        state = None
    return Step(action, state, index)

def read_plan(plan: str, splitter= ","):
    from traces import Trace
    """
    Parse trace from a plan string
    """
    lines = plan.split(splitter)
    steps = []
    for i, line in enumerate(lines):
        if line.startswith(";"):
            break
        steps.append(read_step(line, i))
    return Trace(steps)

def read_plan_file(file_path):
    from traces import Trace
    """
    Parse trace from a plan file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        steps = []
        for i, line in enumerate(lines):
            if line.startswith(";"):
                break
            steps.append(read_step(line, i))
    return Trace(steps)

def read_csv_file(file_path):
    raise NotImplementedError

def read_json_file(file_path):
    raise NotImplementedError