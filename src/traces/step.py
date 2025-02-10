from .action import Action
from .state import State

class Step:
    def __init__ (self, action: Action, state: State, index: int):
        self.action = action
        self.state = state
        self.index = index

    def __str__(self) -> str:
        return f"[{self.index}]{self.action.name}"