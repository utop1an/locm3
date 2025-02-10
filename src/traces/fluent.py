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
        pass