from typing import List
from .planning_object import PlanningObject

class Action:
    def __init__ (self, name: str, obj_params: List[PlanningObject]=None):
        self.name = name
        self.obj_params = obj_params

    def __str__(self)-> str:
        return f"{self.name} {' '.join([o.name for o in self.obj_params])}"

    def __repr__(self)-> str:
        return f"{self.name} {' '.join([repr(o) for o in self.obj_params])}"

    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        return (
            isinstance(other, Action)
            and self.name == other.name
            and self.obj_params == other.obj_params
        )
    
class IndexedAction(Action):
    def __init__(self,index: int, name: str, obj_params: List[PlanningObject] = None, ):
        self.name = name
        self.obj_params = obj_params
        self.obj_params = obj_params

    #TODO...
    