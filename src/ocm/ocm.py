from abc import ABC, abstractmethod


class OCM(ABC):

    def __init__(self):
        pass


    @abstractmethod
    def extract_action_model(self):
        pass

    @staticmethod
    def _pointer_to_set(states: List[Set], pointer1, pointer2 = None):
        pass