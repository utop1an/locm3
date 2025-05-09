from pddl.actions import ActionSignature
from .state import State

class Step:
    def __init__ (self, action: ActionSignature,state: State=None, index: int=None  ):
        self.action = action
        self.state = state
        self.index = index

    def __str__(self) -> str:
        return f"[{self.index}]{self.action.name}"