from typing import List
from .step import Step

class PartialOrderedTrace:
    def __init__ (self, steps: List[Step]=None, flex=0):
        self.steps = steps
        self.flex = flex

    def __len__(self):
        return len(self.steps)
    
    def __iter__(self):
        return iter(self.steps)
