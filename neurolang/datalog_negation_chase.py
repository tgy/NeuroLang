from collections import namedtuple
from itertools import chain
from operator import invert

from .expressions import Constant
from .exceptions import NeuroLangException
from . import solver_datalog_naive as sdb
from .unification import (
    apply_substitution_arguments, compose_substitutions,
    most_general_unifier_arguments
)
from .datalog_chase import (
    merge_instances, obtain_substitutions, evaluate_builtins,
    compute_result_set
)


def chase_step(datalog, instance, builtins, rule, restriction_instance=None):
    if restriction_instance is None:
        restriction_instance = set()

    rule_predicates = extract_rule_predicates(
        rule, instance, builtins, restriction_instance=restriction_instance
    )

    if all(len(predicate_list) == 0 for predicate_list in rule_predicates):
        return {}

    restricted_predicates, nonrestricted_predicates, negative_predicates, \
        builtin_predicates, negative_builtin_predicates = rule_predicates

    rule_predicates_iterator = chain(
        restricted_predicates, nonrestricted_predicates
    )

    substitutions = obtain_substitutions(rule_predicates_iterator)

    substitutions = obtain_negative_substitutions(
        negative_predicates, substitutions
    )

    substitutions = evaluate_builtins(
        builtin_predicates, substitutions, datalog
    )

    substitutions = evaluate_negative_builtins(
        negative_builtin_predicates, substitutions, datalog
    )

    return compute_result_set(
        rule, substitutions, instance, restriction_instance
    )


def obtain_negative_substitutions(negative_predicates, substitutions):
    for predicate, representation in negative_predicates:
        new_substitutions = []
        for substitution in substitutions:
            new_substitutions += unify_negative_substitution(
                predicate, substitution, representation
            )
        substitutions = new_substitutions
    return substitutions


def unify_negative_substitution(predicate, substitution, representation):
    new_substitutions = []
    subs_args = apply_substitution_arguments(predicate.args, substitution)

    for element in representation:
        mgu_substituted = most_general_unifier_arguments(
            subs_args, element.value
        )

        if mgu_substituted is not None:
            break
    else:
        new_substitution = {predicate: element.value}
        new_substitutions.append(
            compose_substitutions(substitution, new_substitution)
        )
    return new_substitutions


def evaluate_negative_builtins(builtin_predicates, substitutions, datalog):
    for predicate, _ in builtin_predicates:
        functor = predicate.functor
        new_substitutions = []
        for substitution in substitutions:
            new_substitutions += unify_negative_builtin_substitution(
                predicate, substitution, datalog, functor
            )
        substitutions = new_substitutions
    return substitutions


def unify_negative_builtin_substitution(
    predicate, substitution, datalog, functor
):
    subs_args = apply_substitution_arguments(predicate.args, substitution)

    mgu_substituted = most_general_unifier_arguments(subs_args, predicate.args)

    if mgu_substituted is not None:
        predicate_res = datalog.walk(
            predicate.apply(functor, mgu_substituted[1])
        )

        if (
            isinstance(predicate_res, Constant[bool]) and
            not predicate_res.value
        ):
            return [compose_substitutions(substitution, mgu_substituted[0])]
    return []


def extract_rule_predicates(
    rule, instance, builtins, restriction_instance=None
):
    if restriction_instance is None:
        restriction_instance = set()

    head_functor = rule.consequent.functor
    rule_predicates = sdb.extract_datalog_predicates(rule.antecedent)
    restricted_predicates = []
    nonrestricted_predicates = []
    negative_predicates = []
    negative_builtin_predicates = []
    builtin_predicates = []
    recursive_calls = 0
    for predicate in rule_predicates:
        functor = predicate.functor

        if functor == head_functor:
            recursive_calls += 1
            if recursive_calls > 1:
                raise ValueError(
                    'Non-linear rule {rule}, solver non supported'
                )

        if functor in restriction_instance:
            restricted_predicates.append(
                (predicate, restriction_instance[functor].value)
            )
        elif functor in instance:
            nonrestricted_predicates.append(
                (predicate, instance[functor].value)
            )
        elif functor in builtins:
            builtin_predicates.append((predicate, builtins[functor]))
        elif functor == invert and predicate.args[0].functor in builtins:
            negative_builtin_predicates.append(
                (predicate.args[0], builtins[predicate.args[0].functor])
            )
        elif functor == invert:
            negative_predicates.append(
                (predicate.args[0], instance[predicate.args[0].functor].value)
            )
        else:
            return ([], [], [], [])

    return (
        restricted_predicates,
        nonrestricted_predicates,
        negative_predicates,
        builtin_predicates,
        negative_builtin_predicates,
    )


ChaseNode = namedtuple('ChaseNode', 'instance children')


def build_chase_tree(datalog_program, chase_set=chase_step):
    builtins = datalog_program.builtins()
    root = ChaseNode(datalog_program.extensional_database(), dict())
    rules = []
    for expression_block in datalog_program.intensional_database().values():
        for rule in expression_block.expressions:
            rules.append(rule)

    nodes_to_process = [root]
    while len(nodes_to_process) > 0:
        node = nodes_to_process.pop()
        for rule in rules:
            new_node = build_nodes_from_rules(
                datalog_program, node, builtins, rule
            )
            if new_node is not None:
                nodes_to_process.append(new_node)
    return root


def build_nodes_from_rules(datalog_program, node, builtins, rule):
    instance_update = chase_step(
        datalog_program, node.instance, builtins, rule
    )
    if len(instance_update) > 0:
        new_instance = merge_instances(node.instance, instance_update)
        new_node = ChaseNode(new_instance, dict())
        node.children[rule] = new_node
        return new_node
    else:
        return None


def build_chase_solution(datalog_program, chase_step=chase_step):
    rules = []
    for expression_block in datalog_program.intensional_database().values():
        for rule in expression_block.expressions:
            rules.append(rule)

    instance = dict()
    builtins = datalog_program.builtins()
    instance_update = datalog_program.extensional_database()
    check_contradiction(datalog_program, instance_update)

    while len(instance_update) > 0:
        instance = merge_instances(instance, instance_update)
        instance_update = merge_instances(
            *(
                chase_step(
                    datalog_program,
                    instance,
                    builtins,
                    rule,
                    restriction_instance=instance_update
                ) for rule in rules
            )
        )

    return instance


def check_contradiction(datalog_program, instance_update):
    for symbol, args in datalog_program.negated_symbols.items():
        instance_values = [x for x in instance_update[symbol].value]
        if symbol in instance_update and next(
            iter(args.value)
        ) in instance_values:
            raise NeuroLangException(f'There is a contradiction in your facts')
