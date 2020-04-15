from typing import AbstractSet, Mapping

import numpy as np
import pandas as pd
import pytest

from ....datalog.expressions import Conjunction, Fact, Implication
from ....exceptions import NeuroLangException
from ....expressions import Constant, ExpressionBlock, Symbol
from ....relational_algebra import (
    ColumnStr,
    NaturalJoin,
    RelationalAlgebraSolver,
)
from ....utils.relational_algebra_set import NamedRelationalAlgebraFrozenSet
from ...expressions import (
    Grounding,
    ProbabilisticPredicate,
    RandomVariableValuePointer,
    VectorisedTableDistribution,
)
from ..graphical_model import (
    CPLogicToGraphicalModelTranslator,
    QueryGraphicalModelSolver,
    SuccQuery,
    and_vect_table_distribution,
    bernoulli_vect_table_distrib,
    extensional_vect_table_distrib,
    succ_query,
)
from ..grounding import ground_cplogic_program

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
T = Symbol("T")
H = Symbol("H")
Y = Symbol("Y")
w = Symbol("w")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
p = Symbol("p")
a = Constant[str]("a")
b = Constant[str]("b")
c = Constant[str]("c")
d = Constant[str]("d")


def _assert_relations_almost_equal(r1, r2):
    assert len(r1.value) == len(r2.value)
    if r1.value.arity == 1 and r2.value.arity == 1:
        np.testing.assert_array_almost_equal(
            r1.value._container[r1.value.columns[0]].values,
            r2.value._container[r2.value.columns[0]].values,
        )
    else:
        joined = RelationalAlgebraSolver().walk(NaturalJoin(r1, r2))
        _, num_cols = _split_numerical_cols(joined)
        if len(num_cols) == 2:
            arr1 = joined.value._container[num_cols[0]].values
            arr2 = joined.value._container[num_cols[1]].values
            np.testing.assert_array_almost_equal(arr1, arr2)
        elif len(num_cols) == 0:
            assert len(joined.value) == len(r1.value)


def test_extensional_grounding():
    grounding = Grounding(
        Implication(P(x, y), Constant[bool](True)),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(
                columns=("x", "y"), iterable={(a, b), (c, d)}
            )
        ),
    )
    translator = CPLogicToGraphicalModelTranslator()
    translator.walk(grounding)
    assert not translator.edges
    assert translator.cpds == {
        P: VectorisedTableDistribution(
            Constant[Mapping](
                {
                    Constant[bool](False): Constant[float](0.0),
                    Constant[bool](True): Constant[float](1.0),
                }
            ),
            grounding,
        )
    }
    translator = CPLogicToGraphicalModelTranslator()
    gm = translator.walk(ExpressionBlock([grounding]))
    assert not gm.edges.value
    assert gm.cpds == Constant[Mapping](
        {
            P: VectorisedTableDistribution(
                Constant[Mapping](
                    {
                        Constant[bool](False): Constant[float](0.0),
                        Constant[bool](True): Constant[float](1.0),
                    }
                ),
                grounding,
            )
        }
    )
    assert gm.groundings == Constant[Mapping]({P: grounding})


def test_probabilistic_grounding():
    probfact = Implication(
        ProbabilisticPredicate(Constant[float](0.3), P(x)),
        Constant[bool](True),
    )
    grounding = Grounding(
        probfact,
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    gm = CPLogicToGraphicalModelTranslator().walk(ExpressionBlock([grounding]))
    assert not gm.edges.value
    assert gm.cpds == Constant[Mapping](
        {
            P: VectorisedTableDistribution(
                Constant[Mapping](
                    {
                        Constant[bool](False): Constant[float](0.7),
                        Constant[bool](True): Constant[float](0.3),
                    }
                ),
                grounding,
            )
        }
    )
    assert gm.groundings == Constant[Mapping]({P: grounding})


def test_intensional_grounding():
    extensional_grounding = Grounding(
        Implication(T(x), Constant[bool](True)),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    probabilistic_grounding = Grounding(
        Implication(
            ProbabilisticPredicate(Constant[float](0.3), P(x)),
            Constant[bool](True),
        ),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    intensional_grounding = Grounding(
        Implication(Q(x), Conjunction([P(x), T(x)])),
        Constant[AbstractSet](
            NamedRelationalAlgebraFrozenSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    groundings = ExpressionBlock(
        [extensional_grounding, probabilistic_grounding, intensional_grounding]
    )
    gm = CPLogicToGraphicalModelTranslator().walk(groundings)
    assert gm.edges == Constant[Mapping]({Q: {P, T}})


def test_bernoulli_vect_table_distrib():
    grounding = Grounding(
        P(x),
        Constant[AbstractSet](
            ExtendedAlgebraSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    distrib = bernoulli_vect_table_distrib(Constant[float](1.0), grounding)
    assert distrib.table.value[Constant[bool](True)] == Constant[float](1.0)
    assert distrib.table.value[Constant[bool](False)] == Constant[float](0.0)
    distrib = bernoulli_vect_table_distrib(Constant[float](0.2), grounding)
    assert distrib.table.value[Constant[bool](True)] == Constant[float](0.2)
    assert distrib.table.value[Constant[bool](False)] == Constant[float](0.8)

    with pytest.raises(NeuroLangException):
        bernoulli_vect_table_distrib(p, grounding)

    solver = ExtendedRelationalAlgebraSolver({})
    walked_distrib = solver.walk(
        bernoulli_vect_table_distrib(Constant[float](0.7), grounding)
    )
    assert isinstance(walked_distrib, Constant[AbstractSet])


def test_extensional_vect_table_distrib():
    grounding = Grounding(
        P(x),
        Constant[AbstractSet](
            ExtendedAlgebraSet(iterable=[1, 2, 3], columns=["x"])
        ),
    )
    distrib = extensional_vect_table_distrib(grounding)
    assert distrib.table.value[Constant[bool](True)] == Constant[float](1.0)
    assert distrib.table.value[Constant[bool](False)] == Constant[float](0.0)


def test_rv_value_pointer():
    parent_values = {
        P: Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[("a", 1), ("b", 1), ("c", 0)],
                columns=["x", make_numerical_col_symb().name],
            )
        )
    }
    solver = ExtendedRelationalAlgebraSolver(parent_values)
    walked = solver.walk(RandomVariableValuePointer(P))
    assert isinstance(walked, Constant[AbstractSet])
    assert isinstance(walked.value, ExtendedAlgebraSet)


@pytest.mark.skip
def test_compute_marginal_probability_single_parent():
    parent_symb = P
    parent_marginal_distrib = Constant[AbstractSet](
        ExtendedAlgebraSet(
            iterable=[("a", 0.2), ("b", 0.8), ("c", 1.0)],
            columns=["x", make_numerical_col_symb().name],
        )
    )
    grounding = Grounding(
        Implication(Q(x), P(x)),
        Constant[AbstractSet](
            ExtendedAlgebraSet(iterable=["c", "a"], columns=["x"])
        ),
    )
    cpd = and_vect_table_distribution(
        rule_grounding=grounding,
        parent_groundings={
            parent_symb: Grounding(
                P(y),
                Constant[AbstractSet](
                    ExtendedAlgebraSet(iterable=["a", "b", "c"], columns=["y"])
                ),
            )
        },
    )
    marginal = compute_marginal_probability(
        cpd, {parent_symb: parent_marginal_distrib}, {parent_symb: grounding}
    )
    _assert_relations_almost_equal(
        marginal,
        Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[("c", 1.0), ("a", 0.2)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


@pytest.mark.skip
def test_compute_marginal_probability_two_parents():
    parent_marginal_distribs = {
        P: Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[("a", 0.2), ("b", 0.8), ("c", 1.0)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
        Q: Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[("b", 0.2), ("c", 0.0), ("d", 0.99)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    }
    parent_groundings = {
        P: Grounding(
            P(y),
            Constant[AbstractSet](
                ExtendedAlgebraSet(iterable=["a", "b", "c"], columns=["y"])
            ),
        ),
        Q: Grounding(
            Q(z),
            Constant[AbstractSet](
                ExtendedAlgebraSet(iterable=["b", "c", "d"], columns=["z"])
            ),
        ),
    }
    grounding = Grounding(
        Implication(Z(x), Conjunction([P(x), Q(x)])),
        Constant[AbstractSet](
            ExtendedAlgebraSet(iterable=["c", "b"], columns=["x"])
        ),
    )
    cpd = and_vect_table_distribution(
        rule_grounding=grounding, parent_groundings=parent_groundings
    )
    marginal = compute_marginal_probability(
        cpd, parent_marginal_distribs, parent_groundings
    )
    _assert_relations_almost_equal(
        marginal,
        Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[("c", 0.0), ("b", 0.16)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_simple():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(b)),
                Constant[bool](True),
            ),
            Implication(Q(x), P(x)),
            # Implication(Q(x), Conjunction([P(x), ])),
        ]
    )
    result = succ_query(code, Q(x))
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[("a", 0.3), ("b", 0.3)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_simple_const_in_antecedent():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Fact(R(a)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(b)),
                Constant[bool](True),
            ),
            Implication(Q(x), Conjunction([P(x), T(x), R(a)])),
        ]
    )
    result = succ_query(code, Q(x))
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[("a", 0.3), ("b", 0.3)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_with_constant():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.3), P(b)),
                Constant[bool](True),
            ),
            Implication(Q(x), Conjunction([P(x), T(x)])),
        ]
    )
    grounded = ground_cplogic_program(code)
    gm = CPLogicToGraphicalModelTranslator().walk(grounded)
    query = SuccQuery(Q(a))
    solver = QueryGraphicalModelSolver(gm)
    result = solver.walk(query)
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[("a", 0.3)],
                columns=["x", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_multiple_parents():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Fact(R(b)),
            Fact(R(c)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(b)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), Z(b)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), Z(c)),
                Constant[bool](True),
            ),
            Implication(Q(x, y), Conjunction([P(x), Z(y), T(x), R(y)])),
        ]
    )
    grounded = ground_cplogic_program(code)
    gm = CPLogicToGraphicalModelTranslator().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    result = solver.walk(SuccQuery(Q(x, y)))
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[
                    ("a", "b", 0.1),
                    ("b", "b", 0.1),
                    ("a", "c", 0.1),
                    ("b", "c", 0.1),
                ],
                columns=["x", "y", make_numerical_col_symb().name],
            )
        ),
    )


def test_succ_query_multi_level():
    code = ExpressionBlock(
        [
            Fact(T(a)),
            Fact(T(b)),
            Fact(R(b)),
            Fact(R(c)),
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(b)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), Z(b)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), Z(c)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.7), Y(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.7), Y(b)),
                Constant[bool](True),
            ),
            Implication(Q(x, y), Conjunction([P(x), Z(y), T(x), R(y)])),
            Implication(H(x, y), Conjunction([Q(x, y), T(y), Y(y)])),
        ]
    )
    grounded = ground_cplogic_program(code)
    gm = CPLogicToGraphicalModelTranslator().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    result = solver.walk(SuccQuery(H(x, y)))
    _assert_relations_almost_equal(
        result,
        Constant[AbstractSet](
            ExtendedAlgebraSet(
                iterable=[("a", "b", 0.07), ("b", "b", 0.07)],
                columns=["x", "y", make_numerical_col_symb().name],
            )
        ),
    )


@pytest.mark.slow
def test_succ_query_hundreds_of_facts():
    facts_t = [Fact(T(Constant[int](i))) for i in range(1000)]
    facts_r = [Fact(R(Constant[int](i))) for i in range(300)]
    code = ExpressionBlock(
        facts_t
        + facts_r
        + [
            Implication(
                ProbabilisticPredicate(
                    Constant[float](0.5), P(Constant[int](i))
                ),
                Constant[bool](True),
            )
            for i in range(1000)
        ]
        + [
            Implication(
                ProbabilisticPredicate(
                    Constant[float](0.2), Z(Constant[int](i))
                ),
                Constant[bool](True),
            )
            for i in range(300)
        ]
        + [Implication(Q(x, y), Conjunction([P(x), Z(y), T(x), R(y)])),]
    )
    grounded = ground_cplogic_program(code)
    gm = CPLogicToGraphicalModelTranslator().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    solver.walk(SuccQuery(Q(x)))


def test_succ_query_hundreds_of_facts_fast():
    extensional_predicate_sets = {
        T: {(i,) for i in range(1000)},
        R: {(i,) for i in range(300)},
    }
    probfacts_sets = {
        P: {(0.2, i,) for i in range(1000)},
        Z: {(0.5, i,) for i in range(300)},
    }
    code = ExpressionBlock(
        (Implication(Q(x, y), Conjunction((P(x), Z(y), T(x), R(y)))),)
    )
    grounded = ground_cplogic_program(
        code,
        probfacts_sets=probfacts_sets,
        extensional_predicate_sets=extensional_predicate_sets,
    )
    gm = CPLogicToGraphicalModelTranslator().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    solver.walk(SuccQuery(Q(x)))


def test_exact_inference_pfact_params():
    param_symb = Symbol.fresh()
    pfact = Implication(
        ProbabilisticPredicate(param_symb, P(x, y)), Constant[bool](True)
    )
    relation = Constant[AbstractSet](
        ExtendedAlgebraSet(
            iterable=[
                ("a", "b", "p1"),
                ("a", "c", "p2"),
                ("b", "b", "p1"),
                ("b", "c", "p2"),
            ],
            columns=("x", "y", param_symb.name),
        )
    )
    pfact_grounding = Grounding(pfact, relation)
    interpretations = {
        P: ExtendedAlgebraSet(
            iterable=[
                ("a", "b", 1),
                ("a", "c", 1),
                ("a", "c", 2),
                ("b", "b", 2),
                ("b", "c", 2),
                ("a", "b", 3),
                ("a", "b", 3),
                ("b", "b", 3),
            ],
            columns=("x", "y", "__interpretation_id__"),
        )
    }
    infer_pfact_params(pfact_grounding, interpretations, 3)


def test_succ_query_with_probchoice_simple():
    probchoice_as_tuples_iterable = [
        (0.2, "a"),
        (0.8, "b"),
    ]
    code = ExpressionBlock((Implication(Q(x), P(x)),))
    grounded = ground_cplogic_program(
        code, probchoice_sets={P: probchoice_as_tuples_iterable}
    )
    gm = CPLogicToGraphicalModelTranslator().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    res = solver.walk(SuccQuery(Q(x)))
    expected = Constant[AbstractSet](
        ExtendedAlgebraSet(
            iterable=[("a", 0.2), ("b", 0.8)],
            columns=["x", make_numerical_col_symb().name],
        )
    )
    _assert_relations_almost_equal(res, expected)


def test_succ_query_with_two_probchoices():
    probchoice_sets = {
        P: [(0.2, "a"), (0.8, "b")],
        Q: [(0.1, "a"), (0.9, "c")],
    }
    code = ExpressionBlock([Implication(Z(x), Conjunction([P(x), Q(x)]))])
    grounded = ground_cplogic_program(code, probchoice_sets=probchoice_sets)
    gm = CPLogicToGraphicalModelTranslator().walk(grounded)
    solver = QueryGraphicalModelSolver(gm)
    res = solver.walk(SuccQuery(Q(x)))
    expected = Constant[AbstractSet](
        ExtendedAlgebraSet(
            iterable=[("a", 0.1), ("c", 0.9)],
            columns=["x", make_numerical_col_symb().name],
        )
    )
    _assert_relations_almost_equal(res, expected)
    res = solver.walk(SuccQuery(Z(x)))
    expected = Constant[AbstractSet](
        ExtendedAlgebraSet(
            iterable=[("a", 0.02),],
            columns=["x", make_numerical_col_symb().name],
        )
    )
    _assert_relations_almost_equal(res, expected)
