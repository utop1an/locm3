from pddl.actions import LearnedLiftedFluent, LearnedAction
from pddl.tasks import Task, Requirements
from pddl.pddl_types import Type
from typing import List


class LearnedModel:

    def __init__(self, fluents: List[LearnedLiftedFluent], actions: List[LearnedAction], types, sort_to_type_dict ):
        self.fluents = fluents
        self.actions = actions
        self.types = types
        self.sort_to_type_dict  = sort_to_type_dict 

    
    def to_pddl_task(self, domain_name):
        """
        Converts the learned model to a PDDL task.
        """

        task_name,requirements, objects, init, goal, use_metric = None, None, None, None, None,None
        requirements = Requirements(["strips", "typing"])
        axioms = []
        functions = []

        types = [Type("object")]
     
        for t in self.types:
            types.append(Type(t))
        type_dict = {t.name: t for t in types}


        predicates = []
        for fluent in self.fluents:
            predicates.append(fluent.to_pddl_predicate(self.sort_to_type_dict ))
        predicate_dict = {p.name: p for p in predicates}


        actions = []
        for action in self.actions:
            actions.append(action.to_pddl_action(predicate_dict, self.sort_to_type_dict ))
        


        task = Task( domain_name, task_name, requirements, types, objects,
        predicates, functions, init, goal, actions, axioms, use_metric)
        return task
    
    

    def to_pddl_file(self):
        pass

