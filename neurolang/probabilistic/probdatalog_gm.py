from typing import Mapping
from collections import defaultdict

import numpy as np

from ..expression_walker import PatternWalker
from ..expression_pattern_matching import add_match
from ..exceptions import NeuroLangException
from ..expressions import (
    Definition,
    Symbol,
    Constant,
    FunctionApplication,
    ExpressionBlock,
)
from ..datalog.expression_processing import extract_datalog_predicates
from .expressions import VectorisedTableDistribution
from .probdatalog import Grounding, is_probabilistic_fact
from .probdatalog_bn import BayesianNetwork


class GraphicalModel(Definition):
    def __init__(self, edges, cpd_factories, groundings):
        self.edges = edges
        self.cpd_factories = cpd_factories
        self.groundings = groundings

    @property
    def random_variables(self):
        return set(self.cpd_factories.value)


def always_true_cpd_factory(parent_values, parent_groundings, grounding):
    return VectorisedTableDistribution(
        Constant[np.ndarray[float]](
            np.vstack(
                [
                    np.zeros(len(grounding), dtype=np.float32),
                    np.ones(len(grounding), dtype=np.float32),
                ]
            )
        )
    )

    return always_true_cpd_factory


def make_probfact_cpd_factory(probability):
    def probfact_cpd_factory(parent_values, parent_groundings, grounding):
        return VectorisedTableDistribution(
            Constant[np.ndarray[float]](
                np.vstack(
                    [
                        np.repeat(1.0 - probability.value, len(grounding)),
                        np.repeat(probability.value, len(grounding)),
                    ]
                )
            )
        )

    return probfact_cpd_factory


def and_cpd_factory(parent_values, parent_groundings, grounding):
    true_probs = np.prod(np.vstack(parent_values), axis=0)
    probs = np.vstack([true_probs, 1.0 - true_probs])


class TranslateGroundedProbDatalogToGraphicalModel(PatternWalker):
    def __init__(self):
        self.edges = dict()
        self.cpd_factories = dict()
        self.groundings = defaultdict(set)

    @add_match(
        ExpressionBlock,
        lambda block: all(
            isinstance(exp, Grounding) for exp in block.expressions
        ),
    )
    def block_of_groundings(self, block):
        for grounding in block.expressions:
            self.walk(grounding)
        return GraphicalModel(
            Constant[Mapping](self.edges),
            Constant[Mapping](self.cpd_factories),
            Constant[Mapping](self.groundings),
        )

    @add_match(
        Grounding, lambda exp: isinstance(exp.expression, FunctionApplication)
    )
    def extensional_grounding(self, grounding):
        self._add_random_variable(
            grounding.expression.functor,
            always_true_cpd_factory,
            grounding,
        )

    @add_match(Grounding, lambda exp: is_probabilistic_fact(exp.expression))
    def probfact_grounding(self, grounding):
        self._add_random_variable(
            grounding.expression.consequent.body.functor,
            make_probfact_cpd_factory(grounding.expression.consequent.head),
            grounding,
        )

    @add_match(Grounding)
    def rule_grounding(self, grounding):
        self._add_random_variable(
            grounding.expression.consequent.functor,
            and_cpd_factory,
            grounding,
        )
        self.edges[grounding.expression.consequent.functor] |= {
            pred.functor
            for pred in extract_datalog_predicates(
                grounding.expression.consequent
            )
        }

    def _add_random_variable(self, pred_symb, cpd_factory, grounding):
        self._check_random_variable_not_already_defined(pred_symb)
        self.cpd_factories[pred_symb] = cpd_factory
        self.groundings[pred_symb] = grounding

    def _check_random_variable_not_already_defined(self, pred_symb):
        if pred_symb in self.cpd_factories:
            raise NeuroLangException(
                f"Already processed predicate symbol {pred_symb}"
            )
