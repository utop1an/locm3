from typing import List
from .planning_object import PlanningObject

class Action:
    def __init__ (self, name: str, parameters: List[PlanningObject]=None):
        self.name = name
        self.parameters = parameters