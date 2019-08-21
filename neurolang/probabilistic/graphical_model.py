import itertools
from collections import defaultdict
from typing import Iterable, Callable, AbstractSet, Mapping

import numpy as np

from ..expressions import (
    Expression, NeuroLangException, FunctionApplication, Constant, Symbol,
    Definition, ExpressionBlock
)
from ..solver_datalog_naive import Implication, Fact
from ..expression_walker import ExpressionWalker
from ..expression_pattern_matching import add_match
from .. import unification
from .ppdl import (
    DeltaTerm, get_antecedent_predicates, get_antecedent_literals,
    is_gdatalog_rule, get_dterm
)
from .distributions import TableDistribution
from ..datalog.instance import SetInstance


def produce(rule, facts):
    if (
        not isinstance(facts, (list, tuple)) or
        any(not isinstance(f, Fact) for f in facts)
    ):
        raise Exception(
            'Expected a list/tuple of facts but got {}'.format(type(facts))
        )
    consequent = rule.consequent
    antecedent_literals = get_antecedent_literals(rule)
    if len(antecedent_literals) != len(facts):
        raise Exception(
            'Expected same number of facts as number of antecedent literals'
        )
    n = len(facts)
    for i in range(n):
        res = unification.most_general_unifier(
            antecedent_literals[i], facts[i].fact
        )
        if res is None:
            return None
        else:
            unifier, _ = res
            for j in range(n):
                consequent = unification.apply_substitution(
                    consequent, unifier
                )
                antecedent_literals[j] = unification.apply_substitution(
                    antecedent_literals[j], unifier
                )
    return Fact(consequent)


def substitute_dterm(datom, value):
    return FunctionApplication[datom.type](
        datom.functor,
        tuple(
            Constant(value) if isinstance(arg, DeltaTerm) else arg
            for arg in datom.args
        )
    )


def is_dterm_constant(dterm):
    return all(isinstance(param, Constant) for param in dterm.args)


def get_constant_dterm_table_cpd(dterm):
    if not is_dterm_constant(dterm):
        raise NeuroLangException('Expected a constant Δ-term')
    if dterm.functor.name == Constant[str]('bernoulli'):
        p = dterm.args[0].value
        return TableDistribution({1: p, 0: 1.0 - p})
    else:
        raise NeuroLangException(f'Unknown distribution {dterm.functor.name}')


FactSet = AbstractSet[Fact]
FactSetSymbol = Symbol[FactSet]
FactSetTableCPD = Mapping[FactSet, float]
FactSetTableCPDFunctor = Definition[
    Callable[[Iterable[FactSet]], FactSetTableCPD]]


class ExtensionalTableCPDFunctor(FactSetTableCPDFunctor):
    def __init__(self, predicate):
        self.predicate = predicate
        self.facts = set()


class IntensionalTableCPDFunctor(FactSetTableCPDFunctor):
    def __init__(self, rule):
        self.rule = rule


class UnionFactSetTableCPDFunctor(FactSetTableCPDFunctor):
    def __init__(self, predicate):
        self.predicate = predicate


class GraphicalModel(Expression):
    def __init__(self):
        self.rv_to_cpd_functor = dict()
        self.parents = defaultdict(frozenset)

    def add_parent(self, child, parent):
        self.parents[child] = self.parents[child].union({parent})


class GDatalogToGraphicalModelTranslator(ExpressionWalker):
    '''Expression walker generating the graphical model
    representation of a GDatalog[Δ] program.
    '''
    def __init__(self):
        self.gm = GraphicalModel()
        self.intensional_predicate_rule_count = defaultdict(int)

    @add_match(ExpressionBlock)
    def expression_block(self, block):
        for exp in block.expressions:
            self.walk(exp)

    @add_match(Fact)
    def fact(self, expression):
        predicate = expression.consequent.functor.name
        rv_symbol = FactSetSymbol(predicate)
        if rv_symbol not in self.gm.rv_to_cpd_functor:
            self.gm.rv_to_cpd_functor[rv_symbol] = (
                ExtensionalTableCPDFunctor(predicate)
            )
        self.gm.rv_to_cpd_functor[rv_symbol].facts.add(expression)

    @add_match(Implication(Definition, ...))
    def rule(self, rule):
        predicate = rule.consequent.functor.name
        self.intensional_predicate_rule_count[predicate] += 1
        rule_id = self.intensional_predicate_rule_count[predicate]
        rule_rv_symbol = FactSetSymbol(f'{predicate}_{rule_id}')
        pred_rv_symbol = FactSetSymbol(f'{predicate}')
        if rule_rv_symbol in self.gm.rv_to_cpd_functor:
            raise NeuroLangException(
                f'Random variable {rule_rv_symbol} already defined'
            )
        self.gm.rv_to_cpd_functor[rule_rv_symbol] = (
            IntensionalTableCPDFunctor(rule)
        )
        for antecedent_pred in get_antecedent_predicates(rule):
            self.gm.add_parent(rule_rv_symbol, antecedent_pred)
        if pred_rv_symbol not in self.gm.rv_to_cpd_functor:
            self.gm.rv_to_cpd_functor[pred_rv_symbol] = \
                UnionFactSetTableCPDFunctor(predicate)
            self.gm.add_parent(pred_rv_symbol, rule_rv_symbol)


def gdatalog2gm(program):
    translator = GDatalogToGraphicalModelTranslator()
    translator.walk(program)
    return translator.gm


def sort_rvs(gm):
    result = list()
    sort_rvs_aux(gm, '__dummy__', set(gm.rv_to_cpd_functor.keys()), result)
    return result[:-1]


def sort_rvs_aux(gm, rv, parents, result):
    for parent_rv in parents:
        sort_rvs_aux(gm, parent_rv, gm.parents[parent_rv], result)
    if rv not in result:
        result.append(rv)


def delta_infer1(rule, instance):
    if not isinstance(instance, SetInstance):
        raise NeuroLangException('Expected instance to be a SetInstance')
    antecedent_facts = tuple(
        instance[pred] for pred in get_antecedent_predicates(rule)
    )
    inferred_facts = set()
    for fact_list in itertools.product(*antecedent_facts):
        new = produce(rule, fact_list)
        if new is not None:
            inferred_facts.add(new)
    if is_gdatalog_rule(rule):
        table = dict()
        for cpd_entries in itertools.product(
            *[
                get_constant_dterm_table_cpd(get_dterm(dfact.consequent)
                                             ).table.items()
                for dfact in inferred_facts
            ]
        ):
            new_facts = frozenset(
                Fact(substitute_dterm(dfact.consequent, entry[0]))
                for dfact, entry in zip(inferred_facts, cpd_entries)
            )
            prob = np.prod([entry[1] for entry in cpd_entries])
            table[new_facts] = prob
    else:
        table = {frozenset(inferred_facts): 1.0}
    return Constant[TableDistribution](TableDistribution(table))


class ConditionalProbabilityQuery(Definition):
    def __init__(self, evidence):
        if not isinstance(evidence, Constant[FactSet]):
            raise NeuroLangException('Expected evidence to be a fact set')
        self.evidence = evidence


class TableCPDGraphicalModelSolver(ExpressionWalker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphical_model = None
        self.ordered_rvs = None

    @add_match(ExpressionBlock)
    def program(self, program):
        if self.graphical_model is not None:
            raise NeuroLangException('GraphicalModel already constructed')
        self.graphical_model = gdatalog2gm(program)
        self.ordered_rvs = sort_rvs(self.graphical_model)

    @add_match(ConditionalProbabilityQuery)
    def conditional_probability_query_resolution(self, query):
        outcomes = self.generate_possible_outcomes()
        matches_query = lambda outcome: query.evidence.value <= outcome
        return Constant(outcomes.value.conditioned_on(matches_query))

    def generate_possible_outcomes(self):
        if self.graphical_model is None:
            raise NeuroLangException(
                'No GraphicalModel generated. Try walking a program'
            )
        results = dict()
        self.generate_possible_outcomes_aux(0, dict(), 1.0, results)
        return Constant[TableDistribution](TableDistribution(results))

    def generate_possible_outcomes_aux(
        self, rv_idx, rv_values, result_prob, results
    ):
        if rv_idx >= len(self.ordered_rvs):
            result = frozenset.union(*rv_values.values())
            if result in results:
                old_prob = results[result]
                new_prob = old_prob + result_prob
                results[result] = new_prob
            else:
                results[result] = result_prob
        else:
            rv_symbol = self.ordered_rvs[rv_idx]
            cpd_functor = self.graphical_model.rv_to_cpd_functor[rv_symbol]
            parent_rvs = self.graphical_model.parents[rv_symbol]
            parent_values = tuple(Constant(rv_values[rv]) for rv in parent_rvs)
            cpd = self.walk(cpd_functor(*parent_values))
            for facts, prob in cpd.value.table.items():
                new_rv_values = rv_values.copy()
                new_rv_values[rv_symbol] = facts
                self.generate_possible_outcomes_aux(
                    rv_idx + 1, new_rv_values, result_prob * prob, results
                )

    @add_match(FunctionApplication(ExtensionalTableCPDFunctor, ...))
    def extensional_table_cpd(self, expression):
        return Constant[TableDistribution](
            TableDistribution({frozenset(expression.functor.facts): 1.0})
        )

    @add_match(FunctionApplication(UnionFactSetTableCPDFunctor, ...))
    def union_table_cpd(self, expression):
        parent_facts = frozenset(
        ).union(*[arg.value for arg in expression.args])
        return Constant[TableDistribution](
            TableDistribution({parent_facts: 1.0})
        )

    @add_match(FunctionApplication(IntensionalTableCPDFunctor, ...))
    def intensional_table_cpd(self, expression):
        parent_facts = frozenset(
        ).union(*[arg.value for arg in expression.args])
        rule = expression.functor.rule
        return delta_infer1(rule, parent_facts)
