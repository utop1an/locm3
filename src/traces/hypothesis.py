from .event import Event
from .fsm import FSM
from dataclasses import asdict

class HIndex:

    B: Event
    k: int
    C: Event
    l: int

    def __hash__(self):
        return hash((self.B, self.k, self.C, self.l))

class HItem:
    S: int
    k_: int
    l_: int
    G: int
    G_: int
    supported: bool
    fsm: FSM

    def __hash__(self):
        return hash((self.S, self.k_, self.l_, self.G, self.G_))
    
class Hypothesis:
    S: int
    B: Event
    k: int
    k_: int
    C: Event
    l: int
    l_: int
    G: int
    G_: int
    fsm: FSM

    def __hash__(self):
        return hash((self.S, self.B, self.k, self.k_, self.C, self.l, self.l_, self.G_))

    def __repr__(self):
        out = "<\n"
        for k, v in asdict(self).items():
            out += f"  {k}={v}\n"
        return out.strip() + "\n>"
    
    def from_dict(hs):
        raise NotImplementedError
