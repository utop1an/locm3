from pddl.actions import ActionSignature
from typing import List

class Event:
    def __init__(self, action: ActionSignature, position: set, sort: List[int] = None):
        self.action = action
        self.position = position
        self.sort = sort

    def __repr__(self):
        return f'{self.action.name}.({",".join(self.position)})'
    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return (
            isinstance(other, Event)
            and self.action == other.action
            and self.position == other.position
        )

class SingletonEvent(Event):
    def __init__(self, action, position: int, sort: int = None):
        self.action = action
        self.position = position
        self.sort = sort

    def __repr__(self):
        return f'{self.action.name}.{self.position}'

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return (
            isinstance(other, SingletonEvent)
            and self.action == other.action
            and self.position == other.position
        )
        
        
class IndexedEvent(Event):

    def __init__(self, action, position, index):
        self.action = action
        self.position = position
        self.index = index

    def __repr__(self):
        return f'[{self.index}]{self.action.name}.({",".join(self.position)})'
            
        
    def __eq__(self, other):
        return (
            isinstance(other, IndexedEvent)
            and self.action == other.action
            and self.position == other.position
            and self.index == other.index
        )
    
    