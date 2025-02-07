from abc import ABC, abstractmethod


class OCM(ABC):

    def __init__(self):
        pass


    @abstractmethod
    def extract_action_model(self):
        pass