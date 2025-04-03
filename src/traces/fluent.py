from .planning_object import PlanningObject
from typing import List

class Fluent:
    
    def __init__(self, name: str, objects: List[PlanningObject]):
        self.name = name
        self.objects = objects

    def __hash__(self):
        return hash(repr(self))
    
    def __str__(self) -> str:
        return f"{self.name} {' '.join(o.name for o in self.objects)}"
    
    def __repr__(self) -> str:
        return f"{self.name} {' '.join(repr(o) for o in self.objects)}"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Fluent):
            return False
        return self.name == other.name and self.objects == other.objects


class LearnedFluent:
        
        def __init__(self, name: str, param_sorts: List[int], param_act_idx: List[int],):
            self.name = name
            self.param_sorts = param_sorts
            self.param_act_idx = param_act_idx

        def __hash__(self):
            return hash(repr(self))
        
        def __str__(self) -> str:
            return f"{self.name} {' '.join(self.param_sorts)}"
        
        def __repr__(self)-> str:
            return f"{self.name} {' '.join(self.param_sorts)}"
        
        def __eq__(self, other) -> bool:
            if not isinstance(other, LearnedFluent):
                return False
            return self.name == other.name and hash(self) == hash(other)
        
