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
            objs.append(TypedObject(obj_name, "object"))
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
    import json
    from traces import Trace, Step, State, PartialOrderedStep, PartialOrderedTrace
    from pddl import ActionSignature, TypedObject
    with open(file_path, 'r') as f:
        data = json.load(f)
        for learning_obj in data:
            raw_traces = learning_obj['traces']
            traces = []
            for raw_trace in raw_traces:
                steps = []
                for i, raw_step in enumerate(raw_trace):
                    action_name = raw_step['action']
                    obj_names = raw_step['objs']
                    objs = []
                    for obj in obj_names:
                        obj_name, obj_type = obj.split("?")
                        objs.append(TypedObject(obj_name, obj_type))
                    action = ActionSignature(action_name, objs)
                    step = Step(action, State(), i)
                    steps.append(step)
                trace = Trace(steps)
                traces.append(trace)

                
            learning_obj['traces'] = traces

            if 'poats' in learning_obj:
                poats = learning_obj['poats']
                po_traces_by_flex = []
                for k, poat in enumerate(poats):
                    
                    actual_flex = poat['actual_flex']
                    pos = poat['pos']
                    inds = poat['inds']

                    po_traces = []

                    for l, trace in enumerate(traces):
                        po = pos[l]
                        ind = inds[l]
                        po_steps = []

                        for j, po_step_ind in enumerate(ind):
                            ori_step = trace[po_step_ind]
                            po_step = PartialOrderedStep(ori_step.state, ori_step.action, ori_step.index, po[j])
                            po_steps.append(po_step)
                        po_traces.append(PartialOrderedTrace(po_steps, actual_flex))
                    po_traces_by_flex.append(po_traces)
                learning_obj['po_traces'] = po_traces_by_flex

    return data