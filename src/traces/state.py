from .fluent import Fluent
from typing import Dict

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