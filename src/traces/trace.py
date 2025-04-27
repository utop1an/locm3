from typing import List
from .step import Step
from .partialOrderedTrace import PartialOrderedTrace
from utils.helpers import convert_trace_to_potrace

class Trace:
    def __init__ (self, steps: List[Step]=None):
        self.steps = steps

    def __len__(self):
        return len(self.steps)
    
    def __iter__(self):
        return iter(self.steps)
    
    def to_partial_ordered_trace(self, target_flex: float)-> PartialOrderedTrace:
        return convert_trace_to_potrace(self, target_flex, True, True)
