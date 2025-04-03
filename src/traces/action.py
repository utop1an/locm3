from typing import List, Union, Set
from .planning_object import PlanningObject
from .fluent import LearnedFluent

class Action:
    def __init__ (self, name: str, obj_params: List[PlanningObject]):
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
    def __init__(self, name: str, obj_params: List[PlanningObject], index: int, ):
        
        self.name = name
        self.obj_params = obj_params
        self.index = index

    
    def __str__(self)-> str:
        return f"[{self.index}]{self.name} {' '.join([o.name for o in self.obj_params])}"

    def __repr__(self)-> str:
        return f"[{self.index}]{self.name} {' '.join([repr(o) for o in self.obj_params])}"

    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        return (
            isinstance(other, IndexedAction)
            and self.name == other.name
            and self.obj_params == other.obj_params
            and self.index == other.index
        )
    
class LearnedAction:
    def __init__(self, name: str, param_sorts: List[str], **kwargs):
        self.name = name
        self.param_sorts = param_sorts
        self.precond = set() if "precond" not in kwargs else kwargs["precond"]
        self.add = set() if "add" not in kwargs else kwargs["add"]
        self.delete = set() if "delete" not in kwargs else kwargs["delete"]

    def __eq__(self, other):
        return (
            isinstance(other, LearnedAction)
            and self.name == other.name
            and self.param_sorts == other.param_sorts
        )

    def __hash__(self):
        # Order of param_sorts is important!
        return hash(self.details())

    def __repr__(self) -> str:
        return f"({self.name} {' '.join(self.param_sorts)})"
        

    def update_precond(
        self, fluents: Union[LearnedFluent, Set[LearnedFluent]]
    ):
        """Adds preconditions to the action.

        Args:
            fluents (set):
                The set of fluents to be added to the action's preconditions.
        """
        if isinstance(fluents, LearnedFluent):
            fluents = {fluents}
        self.precond.update(fluents)

    def update_add(self, fluents: Union[LearnedFluent, Set[LearnedFluent]]):
        """Adds add effects to the action.

        Args:
            fluents (set):
                The set of fluents to be added to the action's add effects.
        """
        if isinstance(fluents, LearnedFluent):
            fluents = {fluents}
        self.add.update(fluents)

    def update_delete(
        self, fluents: Union[LearnedFluent, Set[LearnedFluent]]
    ):
        """Adds delete effects to the action.

        Args:
            fluents (set):
                The set of fluents to be added to the action's delete effects.
        """
        if isinstance(fluents, LearnedFluent):
            fluents = {fluents}
        self.delete.update(fluents)
    