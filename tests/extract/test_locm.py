from utils import read_plan, pprint_list
from extract import LOCM


def test_trace_to_obj_trace():
    plan = "(pickup A)\n(stack A B)\n(unstack C D)\n(stack C A)"
    trace = read_plan(plan)
    locm = LOCM(state_param=False)
    sorts, sort_to_type_dict = locm._get_sorts([trace], None)
    obj_trace_list = locm.trace_to_obj_trace([trace], sorts)
    for obj_trace in obj_trace_list:
        for obj, trace in obj_trace.items():
            print(f"Object: {obj}")
            pprint_list(trace)
    
def test_get_OS_TS():
    plan = "(pickup A)\n(stack A B)\n(unstack C D)\n(stack C A)"
    trace = read_plan(plan)
    locm = LOCM(state_param=False)
    sorts, sort_to_type_dict = locm._get_sorts([trace], None)
    obj_trace_list = locm.trace_to_obj_trace([trace], sorts)
    TS,OS,ap_state_pointers = locm.get_TS_OS(obj_trace_list, sorts)
    
    print("TS:")
    for sort, ts in TS.items():
        print(f"Sort: {sort}")
        for obj, seq in ts.items():
            print(f"Object: {obj}")
            pprint_list(seq)
    print("OS:")
    for sort, os in OS.items():
        print(f"Sort: {sort}")
        pprint_list(os)
    print("ap_state_pointers:")
    for sort, pointers in ap_state_pointers.items():
        print(f"Sort: {sort}")
        print(pointers)

def test_get_model():
    plan = "(pickup A)\n(stack A B)\n(unstack C D)\n(stack C A)"
    trace = read_plan(plan)
    locm = LOCM(state_param=False)
    sorts, sort_to_type_dict = locm._get_sorts([trace], None)
    print(f"Sorts: {sorts}")
    obj_trace_list = locm.trace_to_obj_trace([trace], sorts)
    TS,OS,ap_state_pointers = locm.get_TS_OS(obj_trace_list, sorts)

    model = locm.get_model(OS, ap_state_pointers, sorts, None)
    print("Model:")
    for fluent in model.fluents:
        print(f"Fluent: {fluent}")
    for action in model.actions:
        print(f"Action: {action}")

    pddl_domain = model.to_pddl_domain("test_domain")
    print("PDDL Domain:")
    pddl_domain.dump()
