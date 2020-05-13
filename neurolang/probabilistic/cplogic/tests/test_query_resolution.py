import pytest

from ....datalog import Fact
from ....expressions import Constant, Symbol
from ....logic import Conjunction, Implication, Union
from ....relational_algebra import NaturalJoin, RenameColumns
from ....relational_algebra_provenance import (
    ProvenanceAlgebraSet,
    RelationalAlgebraProvenanceCountingSolver,
)
from .. import testing
from ..gm_provenance_solver import (
    SymbolicTupleEquality,
    SelectionOutPusher,
    UnionOverTuples,
    UnionRemover,
    solve_succ_query,
)
from ..program import CPLogicProgram

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
H = Symbol("H")
A = Symbol("A")
B = Symbol("B")
C = Symbol("C")
x = Symbol("x")
y = Symbol("y")

a = Constant("a")
b = Constant("b")


LATEX_TEMPLATE = r"""
{}
\\
{}
\\
{}
"""


def test_deterministic():
    """
    We define the program

        P(x) <- Q(x)

    And we expect the provenance set resulting from the
    marginalisation of P(x) to be

        _p_ | x
        ====|===
        1.0 | a
        1.0 | b

    """
    code = Union((Fact(Q(a)), Fact(Q(b)), Implication(P(x), Q(x)),))
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    query_pred = P(x)
    result = solve_succ_query(query_pred, cpl_program)
    expected = testing.make_prov_set([(1.0, "a"), (1.0, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_deterministic_conjunction_varying_arity():
    code = Union(
        (
            Fact(Q(a, b)),
            Fact(P(a)),
            Fact(P(b)),
            Implication(Z(x, y), Conjunction((Q(x, y), P(x), P(y)))),
        )
    )
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    query_pred = Z(x, y)
    result = solve_succ_query(query_pred, cpl_program)
    expected = testing.make_prov_set([(1.0, "a", "b")], ("_p_", "x", "y"))
    assert testing.eq_prov_relations(result, expected)


def test_simple_bernoulli():
    """
    We define the program

        P(a) : 0.7 <- T
        P(b) : 0.8 <- T

    And expect the provenance set resulting from the
    marginalisation of P(x) to be

        _p_ | x
        ====|===
        0.7 | a
        0.8 | b

    """
    code = Union(())
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    cpl_program.add_probabilistic_facts_from_tuples(
        P, {(0.7, "a"), (0.8, "b")}
    )
    result = solve_succ_query(P(x), cpl_program)
    expected = testing.make_prov_set([(0.7, "a"), (0.8, "b")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)


def test_bernoulli_conjunction():
    code = Union((Implication(Z(x), Conjunction((P(x), Q(x), R(x)))),))
    probfacts_sets = {
        P: {(1.0, "a"), (0.5, "b")},
        Q: {(0.9, "b"), (0.1, "c")},
        R: {(0.9, "b"), (0.1, "c")},
    }
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    for pred_symb, pfact_set in probfacts_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    result = solve_succ_query(Z(x), cpl_program)
    assert len(result.value) == 1
    assert set(result.value) == {(0.5 * 0.9 * 0.9, "b")}


def test_multi_level_conjunction():
    """
    We consider the program

           P(a) : 0.2 <- T
           Q(a) : 0.9 <- T
        R(a, a) : 0.1 <- T
        R(a, b) : 0.5 <- T
                 Z(x) <- P(x), Q(x)
              H(x, y) <- Z(x), R(x, y)

    And expect the prov set resulting from the
    marginalisation of H(x, y) to be

        _p_   | x  | y
        ======|====|===
        0.018 | a  | a
        0.09  | a  | b

    """
    probfacts_sets = {
        P: {(0.2, "a")},
        Q: {(0.9, "a")},
        R: {(0.1, "a", "a"), (0.5, "a", "b")},
    }
    code = Union(
        (
            Implication(Z(x), Conjunction((P(x), Q(x)))),
            Implication(H(x, y), Conjunction((Z(x), R(x, y)))),
        )
    )
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    for pred_symb, pfact_set in probfacts_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    result = solve_succ_query(H(x, y), cpl_program)
    expected = testing.make_prov_set(
        [(0.2 * 0.9 * 0.1, "a", "a"), (0.2 * 0.9 * 0.5, "a", "b"),],
        ("_p_", "x", "y"),
    )
    assert testing.eq_prov_relations(result, expected)


def test_intertwined_conjunctions_and_probfacts():
    """
    We consider the program

        P(a) : 0.8  <-  T
        C(a) : 0.5  <-  T
        C(b) : 0.9  <-  T
              A(x)  <-  B(x), C(x)
              B(x)  <-  P(x)
              Z(x)  <-  A(x), B(x), C(x)

    And expect SUCC[ Z(x) ] to yield the provenance relation

        _p_ | x
        ====|===
        0.4 | a

    """
    cpl_code = Union(
        (
            Implication(A(x), Conjunction((B(x), C(x)))),
            Implication(B(x), P(x)),
            Implication(Z(x), Conjunction((A(x), B(x), C(x)))),
        )
    )
    cpl = CPLogicProgram()
    cpl.walk(cpl_code)
    cpl.add_probabilistic_facts_from_tuples(P, {(0.8, "a")})
    cpl.add_probabilistic_facts_from_tuples(C, {(0.5, "a"), (0.9, "b")})
    result = solve_succ_query(Z(y), cpl)
    expected = testing.make_prov_set([(0.8 * 0.5, "a")], ("_p_", "y"))
    assert testing.eq_prov_relations(result, expected)


def test_simple_probchoice():
    pchoice_as_sets = {P: {(0.2, "a"), (0.8, "b")}}
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    result_rap_exp, latex = testing.get_succ_query_rap_expression(
        P(x), cpl_program
    )
    assert isinstance(result_rap_exp, UnionOverTuples)
    assert isinstance(result_rap_exp.relation, NaturalJoin)
    assert isinstance(result_rap_exp.relation.relation_left, Selection)
    assert isinstance(
        result_rap_exp.relation.relation_left.formula, SymbolicTupleEquality
    )
    assert isinstance(result_rap_exp.relation.relation_right, RenameColumns)
    assert isinstance(
        result_rap_exp.relation.relation_right.relation, Selection
    )
    assert isinstance(
        result_rap_exp.relation.relation_right.relation.formula,
        SymbolicTupleEquality,
    )
    return
    result = solve_succ_query(P(x), cpl_program)
    expected = testing.make_prov_set([(0.2, "a"), (0.8, "b"),], ("_p_", "x"),)
    assert testing.eq_prov_relations(result, expected)


def test_mutual_exclusivity():
    pchoice_as_sets = {P: {(0.2, "a"), (0.8, "b")}}
    pfact_sets = {Q: {(0.5, "a", "b")}}
    code = Union((Implication(Z(x, y), Conjunction((P(x), P(y), Q(x, y)))),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    for pred_symb, pfact_set in pfact_sets.items():
        cpl_program.add_probabilistic_facts_from_tuples(pred_symb, pfact_set)
    cpl_program.walk(code)
    gm = testing.build_gm(cpl_program)
    result_rap_exp, latex = testing.get_succ_query_rap_expression(
        Z(x, y), cpl_program
    )
    spusher = SelectionOutPusher()
    sexp = spusher.walk(result_rap_exp)
    slatex = testing.rap_expression_to_latex(sexp, cpl_program, gm)
    uremover = UnionRemover()
    uexp = uremover.walk(sexp)
    ulatex = testing.rap_expression_to_latex(uexp, cpl_program, gm)
    with open("/tmp/exp.tex", "w") as f:
        f.write(LATEX_TEMPLATE.format(latex, slatex, ulatex))
    assert isinstance(result_rap_exp, UnionOverTuples)
    assert isinstance(result_rap_exp.relation, NaturalJoin)
    assert isinstance(result_rap_exp.relation.relation_right, RenameColumns)
    solver = RelationalAlgebraProvenanceCountingSolver()
    result = solver.walk(uexp)
    expected = testing.make_prov_set([], ("_p_", "x", "y"))
    assert testing.eq_prov_relations(result, expected)


@pytest.mark.skip
def test_simple_existential():
    """
    We define the following program

        P(a, a) : 0.2 v P(a, b) : 0.8 <- T
                           Q(x) <- ∃y, P(x, y)

    We expect the following to hold

        - Pr[P(a, a)] = 0.2
        - Pr[P(a, b)] = 0.8
        - Pr[Q(a)] = 1.0

    """
    pchoice_as_sets = {P: {(0.2, "a", "a"), (0.8, "a", "b")}}
    code = Union((Implication(Q(x), P(x, y)),))
    cpl_program = CPLogicProgram()
    for pred_symb, pchoice_as_set in pchoice_as_sets.items():
        cpl_program.add_probabilistic_choice_from_tuples(
            pred_symb, pchoice_as_set
        )
    cpl_program.walk(code)
    result = solve_succ_query(Q(x), cpl_program)
    expected = testing.make_prov_set([(1.0, "a")], ("_p_", "x"))
    assert testing.eq_prov_relations(result, expected)
