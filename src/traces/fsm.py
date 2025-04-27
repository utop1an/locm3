
class FSM:

    def __init__(self, sort: int, index: int):
        """Initializes a FSM with a sort and an index.

        Args:
            sort (int):
                The sort of the FSM.
            index (int):
                The index of the FSM.
        """
        self.sort = sort
        self.index = index

    def __repr__(self):
        return f"S{self.sort}F{self.index}"

    def __hash__(self):
        return hash((self.sort, self.index))
    
    def __str__(self):
        return f"Sort{self.sort}FSM{self.index}"
    
    def __eq__(self, other):
        if not isinstance(other, FSM):
            return False
        return hash(self) == hash(other)