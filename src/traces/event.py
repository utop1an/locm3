from pddl.actions import ActionSignature
from typing import List

class Event:
    def __init__(self, action: ActionSignature, pos: List[int], sort: List[int] = None):
        self.action = action
        self.pos = pos
        self.sort = sort

    def __repr__(self):
        return f'{self.action.name}.({",".join(self.pos)})'
    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return (
            isinstance(other, Event)
            and self.action == other.action
            and self.pos == other.pos
        )

class SingletonEvent(Event):
    def __init__(self, action, pos: int, sort: int = None):
        self.action = action
        self.pos = pos
        self.sort = sort

    def __repr__(self):
        return f'{self.action.name}.{self.pos}'

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return (
            isinstance(other, SingletonEvent)
            and self.action == other.action
            and self.pos == other.pos
        )
        
        
class IndexedEvent(Event):

    def __init__(self, action, pos, index):
        self.action = action
        self.pos = pos
        self.index = index

    def __repr__(self):
        return f'[{self.index}]{self.action.name}.({",".join(self.pos)})'
            
        
    def __eq__(self, other):
        return (
            isinstance(other, IndexedEvent)
            and self.action == other.action
            and self.pos == other.pos
            and self.index == other.index
        )
    
    