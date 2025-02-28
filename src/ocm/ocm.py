from abc import ABC, abstractmethod
from typing import List, Set
from traces import Trace, Event, Step, Action


class OCM(ABC):

    def __init__(self):
        pass
    

    @staticmethod
    def _pointer_to_set(states: List[Set], pointer1, pointer2 = None):
        pass

    def trace_to_event(trace: Trace):
        for step in trace.steps:
            action = step.action
            assert action, "Invalid input, missing Action"
            
        

    @abstractmethod
    def extract_action_model(self):
        pass

   