from typing import Dict
class Type:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
    def __hash__(self):
        return hash(str(self))
    
class TypeDistribution(Type):
    def __init__(self, type_distribution: Dict[str, float]):
        self.type_distribution = type_distribution
    
    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __hash__(self):
        pass
        
