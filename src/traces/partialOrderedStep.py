from pddl.actions import ActionSignature
from .state import State
from .step import Step
from typing import List, Optional

class PartialOrderedStep(Step):
    """An unordered step in a Trace.

    A Step object stores a State and the action that is taken from the state.
    The final step in a trace will not have an action associated with it.

    Attributes:
        state (State):
            The state in this step.
        action (Action | None):
            The action taken from the state in this step.
        index (int | None):
            Not "The 'place' of the step in the trace" as for the Step.
            Just the the index to identify each unordered step in the trace.
        successors(List[int] | None):
            The indices of all other unordered steps that is partially ordered after this step.
    """

    def __init__(self, state: State, action: ActionSignature, index: int, successors: List[int]):
        """Initializes a Step with a state and optionally an action.

        Args:
            state (State):
                The state in this step.
            action (Action | None):
                The action taken from the state in this step. Must provide a
                value, but value can be None.
            index (int):
                The index of this unordered step in the trace.
            successors(List[int] | None):
                The indices of all other unordered steps that is partially ordered after this step.
        """
        self.state = state
        self.action = action
        self.index = index
        self.successors = successors

    def __str__(self) -> str:
        return f"{self.action.name} [{self.index}]"