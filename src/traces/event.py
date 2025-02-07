from .action import Action

class Event:
    def __init__(self, action: Action, positions: set):
        self.action = action
        self.positions = positions

    def __repr__(self):
        if len(self.positions == 1):
            return f'{self.action.name}.{self.positions}'
        if len(self.positions> 1):
            return f'{self.action.name}.({",".join(self.positions)})'
    