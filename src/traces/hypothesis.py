from .event import Event
from .fsm import FSM
from dataclasses import asdict, dataclass
from typing import Dict, Set, List, Union
from collections import defaultdict

def to_tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)

@dataclass
class HIndex:

    B: Event
    k: Union[int, List[int]]
    C: Event
    l: Union[int, List[int]]

    def __hash__(self):
        return hash((self.B, to_tuple(self.k), self.C, to_tuple(self.l)))

@dataclass
class HItem:
    S: int
    k_: Union[int, List[int]]
    l_: Union[int, List[int]]
    G: int
    G_: int
    supported: bool
    fsm: FSM = None

    def __hash__(self):
        return hash((self.S, to_tuple(self.k_), to_tuple(self.l_), self.G, self.G_))
    
@dataclass
class Hypothesis:
    S: int
    B: Event
    k: Union[int, List[int]]
    k_: Union[int, List[int]]
    C: Event
    l: Union[int, List[int]]
    l_: Union[int, List[int]]
    G: int
    G_: int
    fsm: FSM

    def __hash__(self):
        return hash((self.S, self.B, to_tuple(self.k), to_tuple(self.k_), self.C, to_tuple(self.l), to_tuple(self.l_), self.G_))

    def __repr__(self):
        out = "<\n"
        for k, v in asdict(self).items():
            out += f"  {k}={v}\n"
        return out.strip() + "\n>"
    
    def from_dict(
        hs: Dict[HIndex, Set[HItem]]
    ) -> Dict[int, Dict[int, Set["Hypothesis"]]]:
        """Converts a dict of HSIndex -> HSItem to a dict of G -> S -> Hypothesis"""
        HS: Dict[int, Dict[int, Set["Hypothesis"]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for hsind, hsitems in hs.items():
            hsind = hsind.__dict__
            for hsitem in hsitems:
                hsitem_dict = hsitem.__dict__
                hsitem_dict.pop("supported")
                HS[hsitem.G][hsitem.S].add(Hypothesis(**{**hsind, **hsitem_dict}))
        return HS
    
    def from_dict_fsm(
        hs: Dict[HIndex, Set[HItem]]
    ) -> Dict[FSM, Dict[int, Set["Hypothesis"]]]:
        """Converts a dict of HSIndex -> HSItem to a dict of FSM -> S -> Hypothesis"""
        HS: Dict[FSM, Dict[int, Set["Hypothesis"]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for hsind, hsitems in hs.items():
            hsind = hsind.__dict__
            for hsitem in hsitems:
                hsitem_dict = hsitem.__dict__
                hsitem_dict.pop("supported")
                HS[hsitem.fsm][hsitem.S].add(Hypothesis(**{**hsind, **hsitem_dict}))
        return HS
