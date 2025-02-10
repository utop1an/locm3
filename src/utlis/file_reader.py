"""
Read traces from different type of inputs.
Plans should be in the form of:
    without state info `action1(param1)\naction2(param1 param2)\naction3(param2)...`
    with state info `action(param1), predicate1 param1\naction2(param1 param2), predicate2 param1 param2, predicate3 param2\naction3(param2), predicate4 param2`
"""

from traces import *

def parse_action(action):
    tmp = action.strip().strip("()").split(" ")
    action_name = action[0]
    

    return Action()

def parse_state(predicates):

    return State()

def read_step(step, index):
    tmp = step.split(', ')
    action = parse_action(tmp[0])
    if len(tmp) >1:
        state = parse_state(tmp[1:])
    return Step(action, state, index)


def read_plan(file_path):
    """
    
    """

    pass

def read_csv(file_path):
    pass

def read_json(file_path):
    pass