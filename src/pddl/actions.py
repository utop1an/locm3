import copy
from typing import List, Set, Union
from .conditions import NegatedAtom, Atom, Conjunction, Impossible, Truth
from .effects import SimpleEffect, ConjunctiveEffect, Effect
from .predicates import LearnedLiftedFluent
from .pddl_types import TypedObject

class ActionSignature:
    def __init__ (self, name: str, obj_params: List[TypedObject]):
        self.name = name
        self.obj_params = obj_params

    def __str__(self)-> str:
        return f"{self.name} {' '.join([o.name for o in self.obj_params])}"

    def __repr__(self)-> str:
        return f"{self.name} {' '.join([repr(o) for o in self.obj_params])}"

    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        return (
            isinstance(other, Action)
            and self.name == other.name
            and self.obj_params == other.obj_params
        )
    
class IndexedActionSignature(ActionSignature):
    def __init__(self, name: str, obj_params: List[TypedObject], index: int, ):
        
        self.name = name
        self.obj_params = obj_params
        self.index = index

    
    def __str__(self)-> str:
        return f"[{self.index}]{self.name} {' '.join([o.name for o in self.obj_params])}"

    def __repr__(self)-> str:
        return f"[{self.index}]{self.name} {' '.join([repr(o) for o in self.obj_params])}"

    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        return (
            isinstance(other, IndexedActionSignature)
            and self.name == other.name
            and self.obj_params == other.obj_params
            and self.index == other.index
        )
    
class LearnedAction:
    def __init__(self, name: str, param_types: List[str], **kwargs):
        self.name = name
        self.param_types = param_types
        self.precond:Set[LearnedLiftedFluent] = set() if "precond" not in kwargs else kwargs["precond"]
        self.add:Set[LearnedLiftedFluent] = set() if "add" not in kwargs else kwargs["add"]
        self.delete:Set[LearnedLiftedFluent] = set() if "delete" not in kwargs else kwargs["delete"]

    def __eq__(self, other):
        return (
            isinstance(other, LearnedAction)
            and self.name == other.name
            and self.param_types == other.param_types
        )

    def __hash__(self):
        # Order of param_types is important!
        return hash(repr(self))

    def __repr__(self) -> str:
        return f"({self.name} {' '.join(str(s) for s in self.param_types)})"
        
    def update_param_types(self, index, sort):
        """Updates the parameter sorts of the action.

        Args:
            index (int): The index of the parameter to be updated.
            sort (str): The new sort to be assigned to the parameter.
        """
        self.param_types[index] = sort

    def update_precond(
        self, fluents: Union[LearnedLiftedFluent, Set[LearnedLiftedFluent]]
    ):
        """Adds preconditions to the action.

        Args:
            fluents (set):
                The set of fluents to be added to the action's preconditions.
        """
        if isinstance(fluents, LearnedLiftedFluent):
            fluents = {fluents}
        self.precond.update(fluents)

    def update_add(self, fluents: Union[LearnedLiftedFluent, Set[LearnedLiftedFluent]]):
        """Adds add effects to the action.

        Args:
            fluents (set):
                The set of fluents to be added to the action's add effects.
        """
        if isinstance(fluents, LearnedLiftedFluent):
            fluents = {fluents}
        self.add.update(fluents)

    def update_delete(
        self, fluents: Union[LearnedLiftedFluent, Set[LearnedLiftedFluent]]
    ):
        """Adds delete effects to the action.

        Args:
            fluents (set):
                The set of fluents to be added to the action's delete effects.
        """
        if isinstance(fluents, LearnedLiftedFluent):
            fluents = {fluents}
        self.delete.update(fluents)
    
    def to_pddl_action(self, predicate_dict ):
        """Converts the learned action to a PDDL action.
        """
        
        parameters = [TypedObject(f"?x{i}", t) for i, t in enumerate(self.param_types)]
        preconds = []
        for learned_pre in self.precond:
            atom = Atom(learned_pre.name, [f"?x{i}" for i in learned_pre.param_act_idx])
            preconds.append(atom)
        precondition = Conjunction(preconds)
        effs=[]
        condition = Truth().simplified()
        for learned_add in self.add:
            atom = Atom(learned_add.name, [f"?x{i}" for i in learned_add.param_act_idx])
            
            add_eff = Effect([], condition, atom)
           
            effs.append(add_eff)
        
        for learned_del in self.delete:
            negatedAtom = NegatedAtom(learned_del.name, [f"?x{i}" for i in learned_del.param_act_idx])
            del_eff = Effect([], condition, negatedAtom)
            effs.append(del_eff)
        conjunctive_effect = ConjunctiveEffect(effs)
        return Action(self.name, parameters, len(parameters), precondition, effs)

class Action:
    def __init__(self, name, parameters, num_external_parameters,
                 precondition, effects, cost=None):
        assert 0 <= num_external_parameters <= len(parameters)
        self.name = name
        self.parameters = parameters
        # num_external_parameters denotes how many of the parameters
        # are "external", i.e., should be part of the grounded action
        # name. Usually all parameters are external, but "invisible"
        # parameters can be created when compiling away existential
        # quantifiers in conditions.
        self.num_external_parameters = num_external_parameters
        self.precondition = precondition
        self.effects = effects
        self.cost = cost
        self.uniquify_variables() # TODO: uniquify variables in cost?

    def __repr__(self):
        return "<Action %r at %#x>" % (self.name, id(self))

    def dump(self):
        print("%s(%s)" % (self.name, ", ".join(map(str, self.parameters))))
        print("Precondition:")
        self.precondition.dump()
        print("Effects:")
        for eff in self.effects:
            eff.dump()
        print("Cost:")
        if(self.cost):
            self.cost.dump()
        else:
            print("  None")

    def uniquify_variables(self):
        self.type_map = {par.name: par.type_name for par in self.parameters}
        self.precondition = self.precondition.uniquify_variables(self.type_map)
        for effect in self.effects:
            effect.uniquify_variables(self.type_map)

    def relaxed(self):
        new_effects = []
        for eff in self.effects:
            relaxed_eff = eff.relaxed()
            if relaxed_eff:
                new_effects.append(relaxed_eff)
        return Action(self.name, self.parameters, self.num_external_parameters,
                      self.precondition.relaxed().simplified(),
                      new_effects, self.cost)

    def untyped(self):
        # We do not actually remove the types from the parameter lists,
        # just additionally incorporate them into the conditions.
        # Maybe not very nice.
        result = copy.copy(self)
        parameter_atoms = [par.to_untyped_strips() for par in self.parameters]
        new_precondition = self.precondition.untyped()
        result.precondition = Conjunction(parameter_atoms + [new_precondition])
        result.effects = [eff.untyped() for eff in self.effects]
        return result

    def instantiate(self, var_mapping, init_facts, init_assignments,
                    fluent_facts, objects_by_type, metric):
        """Return a PropositionalAction which corresponds to the instantiation of
        this action with the arguments in var_mapping. Only fluent parts of the
        conditions (those in fluent_facts) are included. init_facts are evaluated
        while instantiating.
        Precondition and effect conditions must be normalized for this to work.
        Returns None if var_mapping does not correspond to a valid instantiation
        (because it has impossible preconditions or an empty effect list.)"""
        arg_list = [var_mapping[par.name]
                    for par in self.parameters[:self.num_external_parameters]]
        name = "(%s %s)" % (self.name, " ".join(arg_list))

        precondition = []
        try:
            self.precondition.instantiate(var_mapping, init_facts,
                                          fluent_facts, precondition)
        except Impossible:
            return None
        effects = []
        for eff in self.effects:
            eff.instantiate(var_mapping, init_facts, fluent_facts,
                            objects_by_type, effects)
        if effects:
            if metric:
                if self.cost is None:
                    cost = 0
                else:
                    cost = int(self.cost.instantiate(
                        var_mapping, init_assignments).expression.value)
            else:
                cost = 1
            return PropositionalAction(name, precondition, effects, cost)
        else:
            if metric:
                if self.cost is None:
                    cost = 0
                else:
                    cost = int(self.cost.instantiate(
                        var_mapping, init_assignments).expression.value)
            else:
                cost = 1
            return PropositionalAction(name, precondition, [], cost)


class PropositionalAction:
    def __init__(self, name, precondition, effects, cost):
        self.name = name
        self.precondition = precondition
        self.add_effects = []
        self.del_effects = []
        for condition, effect in effects:
            if not effect.negated:
                self.add_effects.append((condition, effect))
        # Warning: This is O(N^2), could be turned into O(N).
        # But that might actually harm performance, since there are
        # usually few effects.
        # TODO: Measure this in critical domains, then use sets if acceptable.
        for condition, effect in effects:
            if effect.negated and (condition, effect.negate()) not in self.add_effects:
                self.del_effects.append((condition, effect.negate()))
        self.cost = cost

    def __repr__(self):
        return "<PropositionalAction %r at %#x>" % (self.name, id(self))

    def dump(self):
        print(self.name)
        for fact in self.precondition:
            print("PRE: %s" % fact)
        for cond, fact in self.add_effects:
            print("ADD: %s -> %s" % (", ".join(map(str, cond)), fact))
        for cond, fact in self.del_effects:
            print("DEL: %s -> %s" % (", ".join(map(str, cond)), fact))
        print("cost:", self.cost)
