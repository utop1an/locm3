from .ocm import OCM
from typing import Dict

class LOCM2(OCM):

    def __init__(self, state_param:bool=True, timeout:int=600, debug: Dict[str, bool]=None):
        super().__init__(state_param, timeout, debug)

    def extract_action_model(self):
        return super().extract_action_model()