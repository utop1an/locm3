from . import TypedObject
from typing import List

class Fluent:
    
    def __init__(self, name: str, objects: List[TypedObject]):
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


class LearnedLiftedFluent:
        
        def __init__(self, name: str, param_types: List[int], param_act_idx: List[int],):
            self.name = name
            self.param_types = param_types
            self.param_act_idx = param_act_idx

        def __hash__(self):
            return hash(repr(self))
        
        def __str__(self) -> str:
            return f"{self.name} {' '.join(str(s) for s in self.param_types)} [{' '.join(str(s) for s in self.param_act_idx)}]"
        
        def __repr__(self)-> str:
            return f"{self.name} {' '.join(str(s) for s in self.param_types)}"
        
        def __eq__(self, other) -> bool:
            if not isinstance(other, LearnedLiftedFluent):
                return False
            return self.name == other.name and hash(self) == hash(other)
        
        def to_pddl_predicate(self ):
            arguments = []
            for i, sort in enumerate(self.param_types):
                arg = TypedObject(f"?x{i}", sort)
                arguments.append(arg)
            return Predicate(self.name, arguments)
        

class Predicate:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    def __str__(self):
        return "%s(%s)" % (self.name, ", ".join(map(str, self.arguments)))

    def get_arity(self):
        return len(self.arguments)
