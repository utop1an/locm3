from typing import List
from .step import Step
from .partialOrderedStep import PartialOrderedStep

class PartialOrderedTrace:
    def __init__ (self, steps: List[PartialOrderedStep]=None, flex=0):
        self.steps = steps
        self.flex = flex

    def __len__(self):
        return len(self.steps)
    
    def __iter__(self):
        return iter(self.steps)
    
    def __getitem__(self, index):
        for step in self.steps:
            if step.index == index:
                return step
        raise KeyError(f"PO Step with index {index} not found in the PO trace.")
