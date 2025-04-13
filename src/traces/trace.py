from typing import List
from .step import Step

class Trace:
    def __init__ (self, steps: List[Step]=None):
        self.steps = steps

    def __len__(self):
        return len(self.steps)
    
    def __iter__(self):
        return iter(self.steps)
