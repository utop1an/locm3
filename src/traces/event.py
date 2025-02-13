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
        
    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return (
            isinstance(other, Event)
            and self.action == other.action
            and self.positions == other.positions
        )
        
        
class IndexedEvent(Event):

    def __init__(self, action, positions, index):
        self.action = action
        self.positions = positions
        self.index = index

    def __repr__(self):
        if len(self.positions == 1):
            return f'[{self.index}]{self.action.name}.{self.positions}'
        if len(self.positions> 1):
            return f'[{self.index}]{self.action.name}.({",".join(self.positions)})'
        
    def __eq__(self, other):
        return (
            isinstance(other, IndexedEvent)
            and self.action == other.action
            and self.positions == other.positions
            and self.index == other.index
        )
    
    