
from traces import Trace, Action
from extract import OCM

def test_get_sorts():
    # Define a mock tracelist with actions and objects
    action1 = Action(name="action1", obj_params=["obj1", "obj2"])
    action2 = Action(name="action2", obj_params=["obj3"])
    tracelist = [Trace(action=action1), Trace(action=action2)]

    # Define expected sorts
    expected_sorts = {
        (action1.name, 0): {"obj1"},
        (action1.name, 1): {"obj2"},
        (action2.name, 0): {"obj3"}
    }

    # Call the method to test
    ocm = OCM()
    result_sorts = ocm._get_sorts(tracelist, None)

    # Assert the result is as expected
    assert result_sorts == expected_sorts, f"Expected {expected_sorts}, but got {result_sorts}"
