from .event import Event
from typing import Set

"""
Object state machine
"""
class OSM:
    
    sort: int
    event: Event
    del_events: Set

    def __init__(self, sort: int, event: Event, del_events: Set[Event]):
        self.sort = sort
        self.event = event
        self.del_events = del_events