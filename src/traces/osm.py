from .event import Event
from typing import Set

"""
Object state machine
"""
class OSM:
    
    sort: int
    event: Event
    del_events: Set
    self_included: bool

    def __init__(self, sort: int, event: Event, del_events: Set[Event], self_included: bool):
        self.sort = sort
        self.event = event
        self.del_events = del_events
        self.self_included = self_included