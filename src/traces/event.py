from pddl.actions import ActionSignature
from typing import List

def to_tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)

class Event:
    def __init__(self, action: ActionSignature, pos: List[int], sort: int = None):
        self.action = action
        self.pos = pos
        self.sort = sort

    def __repr__(self):
        return f'{self.action.name}.({" ".join(str(p) for p in self.pos)})'
    def __hash__(self):
        return hash((self.action.name, to_tuple(self.pos)))

    def __eq__(self, other):
        return (
            isinstance(other, Event)
            and hash(self) == hash(other)
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
            and hash(self) == hash(other)
        )
        
        
class IndexedEvent(Event):

    def __init__(self, action, index, pos, sort=None):
        self.action = action
        self.index = index
        self.pos = pos
        self.sort =sort
        

    def __repr__(self):
        return f'[{self.index}]{self.action.name}.{self.pos}'
            
    def __hash__(self):
        return hash((self.action.name, self.index, self.pos))    

    def __eq__(self, other):
        return (
            isinstance(other, IndexedEvent)
            and hash(self) == hash(other)
        )
    
    def to_indexed_action(self):
        return IndexedEvent(self.action, self.index, None, None)
    
    def to_event(self):
        return SingletonEvent(self.action, self.pos, self.sort)
    
    