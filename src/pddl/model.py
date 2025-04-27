from .actions import LearnedLiftedFluent, LearnedAction
from .tasks import Task, Requirements
from .pddl_types import Type
from typing import List


class LearnedModel:

    def __init__(self, fluents: List[LearnedLiftedFluent], actions: List[LearnedAction], types ):
        self.fluents = fluents
        self.actions = actions
        self.types = types

    
    def to_pddl_domain(self, domain_name):
        """
        Converts the learned model to a PDDL task.
        """

        task_name,requirements, objects, init, goal, use_metric = None, None, None, None, None,None
        requirements = Requirements([":strips", ":typing"])
        axioms = []
        functions = []

        types = [Type("object")]
     
        types.extend(self.types)
        type_dict = {t.name: t for t in types}


        predicates = []
        for fluent in self.fluents:
            predicates.append(fluent.to_pddl_predicate())
        predicate_dict = {p.name: p for p in predicates}
       

        actions = []
        for action in self.actions:
            actions.append(action.to_pddl_action(predicate_dict ))
       


        task = Task( domain_name, task_name, requirements, types, objects,
        predicates, functions, init, goal, actions, axioms, use_metric)
        return task
    
    

    def to_pddl_file(self):
        pass

