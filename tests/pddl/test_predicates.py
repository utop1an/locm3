from pddl import LearnedLiftedFluent, Predicate, TypedObject

def test_to_predicate():
    # Create a mock type dictionary
    type_dict = {
        1: 'Type1',
        2: 'Type2'
    }

    # Create a LearnedLiftedFluent instance
    fluent = LearnedLiftedFluent(name='test_fluent', param_sorts=[1, 2], param_act_idx=[0, 1])

    # Convert to Predicate
    predicate = fluent.to_predicate(type_dict)

    # Check the name and arguments of the Predicate
    assert predicate.name == 'test_fluent'
    assert len(predicate.arguments) == 2
    assert predicate.arguments[0] == TypedObject('x0', 'Type1')
    assert predicate.arguments[1] == TypedObject('x1', 'Type2')