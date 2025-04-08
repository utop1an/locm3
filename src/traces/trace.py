from typing import List
from .step import Step

class Trace:
    def __init__ (self, steps: List[Step]=None):
        self.steps = steps
