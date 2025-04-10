
from pddl import LearnedModel, LearnedLiftedFluent, LearnedAction, Type, TypedObject

def test_to_pddl_task():
    fluents = {
        LearnedLiftedFluent("on", [1, 1], [1,2]),
        LearnedLiftedFluent("clear", [1], [2]),
        LearnedLiftedFluent("holding", [1], [1])
    }

    actions = {
        LearnedAction(
            "stack", 
            [1,1], 
            precond = {
                LearnedLiftedFluent("clear", [1], [2]),
                LearnedLiftedFluent("holding", [1], [1])
                },
            add = {
                LearnedLiftedFluent("clear", [1], [1]),
                LearnedLiftedFluent("on", [1, 1], [1,2]),
            },
            delete = {
                LearnedLiftedFluent("clear", [1], [2]),
                LearnedLiftedFluent("holding", [1], [1])
            }
        )
    }

    types = [
        Type("block", "object")
    ]
        
       

    sort_to_type_dict = {
        1: Type("block", "object"),
      
    }

    model = LearnedModel(fluents, actions, types, sort_to_type_dict)

    pddl_task = model.to_pddl_task("test_domain")
    pddl_task.dump()