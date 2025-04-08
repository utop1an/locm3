
from traces import Trace, Step
from pddl import TypedObject, ActionSignature
from extract import OCM

def test_get_sorts():
    # Define a mock tracelist with actions and objects
    action1 = ActionSignature(name="action1", obj_params=[TypedObject("obj1", "null"), TypedObject("obj2", "null")])
    action2 = ActionSignature(name="action2", obj_params=[TypedObject("obj3", "null")])
    step1 = Step(action1, 0)
    step2 = Step(action2, 1)
    trace = Trace([step1, step2])

    # Call the method to test
    result_sorts = OCM._get_sorts([trace], None)


def test_sorts_to_types():
    # Define a mock sort dictionary
    sorts = {
        "obj1": 1,
        "obj2": 1,
        "obj3": 2
    }

    # Define expected types
    expected_types = [
        TypedObject("obj1", "s1"),
        TypedObject("obj2", "s1"), 
        TypedObject("obj3", "s2")
    ]

    # Call the method to test
    result_types = OCM._sorts_to_types(sorts, None)

    # Assert the result is as expected
    assert result_types == expected_types, f"Expected {expected_types}, but got {result_types}"