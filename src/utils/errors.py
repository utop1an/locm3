

class GeneralTimeOut(Exception):
    """
    Raised when the time it takes to complete a task is longer than the generator's `max_time` attribute.
    """

    def __init__(self, max_time: float, source="Unknown"):
        message = f"Could not complete the {source} task in {max_time} seconds or less."
        super().__init__(message)