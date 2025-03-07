
class FSM:
    sort: int
    index: int

    def __repr__(self):
        return f"S{self.sort}F{self.index}"

    def __hash__(self):
        return hash((self.sort, self.index))
    
    def __str__(self):
        return f"Sort{self.sort}FSM{self.index}"
    
    def __eq__(self, other):
        if not isinstance(other, FSM):
            return False
        return self.sort == other.sort and self.index == other.index