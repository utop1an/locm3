from .ocm import OCM

class POLOCM(OCM):

    def __init__(self):
        super().__init__()

    def extract_action_model(self):
        return super().extract_action_model()