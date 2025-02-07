from typing import List
from traces import Step

class Trace:
    def __init__ (self, steps: List[Step]=None):
        self.steps = steps
