from pddl.predicates import Fluent
from typing import Dict
from typing import NamedTuple

class State:
    def __init__ (self, fluents: Dict[Fluent, bool]=None):
        self.fluents = fluents

    def __eq__(self, other)-> bool:
        return (
            isinstance(other, State)
            and self.fluents == other.fluents
        )

    def __str__(self)->str:
        return ", ".join([str(fluent) for (fluent, value) in self.fluents.items() if value])

    def __repr__(self)->str:
        out = []
        for fluent, value in self.fluents.items():
            if value:
                out.append(f"+{repr(fluent)}")
            else:
                out.append(f"-{repr(fluent)}")
        return ", ".join(out)

    def __hash__(self):
        return hash(repr(self))

class StatePointers(NamedTuple):
    start: int
    end: int

    def __repr__(self)-> str:
        return f"({self.start}-{self.end})"