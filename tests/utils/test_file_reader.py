from utils import read_plan

def test_read_plan_no_state():
    plan = "(pickup A)\n(stack A B)\n(unstack C D)\nstack(C A)"
    trace = read_plan(plan)
    assert len(trace.steps) == 4
    assert trace.steps[0].action.name == "pickup"
    assert trace.steps[1].action.obj_params[0].name == "A"