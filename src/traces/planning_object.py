from .type import Type

class PlanningObject:
    def __init__ (self, name: str, type: Type):
        self.name = name
        self.type = type
    
    def __str__ (self):
        return self.name
    
    def __repr__(self):
        return f"{self.name}?{self.type}"