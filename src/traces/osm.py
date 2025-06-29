from .event import Event
from typing import Set, Tuple
import networkx as nx

import os

"""
Object state machine
"""
class OSM:
    
    sort: int
    event: Event
    del_events: Set
    self_included: bool

    def __init__(self, sort_tuple: Tuple[int], event: Event, del_events: Set[Event], self_included: bool):
        self.sort_tuple = sort_tuple
        self.event = event
        self.del_events = del_events
        self.self_included = self_included

    def draw(self):
        G = nx.DiGraph()
        G.add_node(0)
        G.add_node(1)
        G.add_edge(0, 1, label=f"{self.event.action.name}.{self.event.pos}")
        if self.self_included:
            G.add_edge(1, 1, label=f"{self.event.action.name}.{self.event.pos}")
        G.add_edge(1, 1, label="E-\{e\}")
        for del_event in self.del_events:
            if G.has_edge(1, 0):
                G.edges[1, 0]["label"] += f"\n{del_event.action.name}.{del_event.pos}"
            else:
                G.add_edge(1, 0, label=f"{del_event.action.name}.{del_event.pos}")
        
        # write to dot file
        dot = nx.drawing.nx_pydot.to_pydot(G)
        dot.write_png(f"l3-{self.event}.png")
    