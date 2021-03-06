from operator import eq

from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import Constant, ExpressionBlock, Symbol
from ...logic import ExistentialPredicate, Implication, Negation
from .. import DatalogProgram, Fact
from ..expression_processing import (
    TranslateToDatalogSemantics, extract_logic_free_variables,
    extract_logic_predicates,
    implication_has_existential_variable_in_antecedent,
    is_conjunctive_expression,
    is_conjunctive_expression_with_nested_predicates, is_linear_rule,
    reachable_code, stratify)


S_ = Symbol
C_ = Constant
Imp_ = Implication
B_ = ExpressionBlock
EP_ = ExistentialPredicate
T_ = Fact


DT = TranslateToDatalogSemantics()


def test_conjunctive_expression():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert is_conjunctive_expression(
        Q()
    )

    assert is_conjunctive_expression(
        Q(x)
    )

    assert is_conjunctive_expression(
        DT.walk(Q(x) & R(y, C_(1)))
    )

    assert not is_conjunctive_expression(
        DT.walk(R(x) | R(y))
    )

    assert not is_conjunctive_expression(
        DT.walk(R(x) & R(y) | R(x))
    )

    assert not is_conjunctive_expression(
        DT.walk(~R(x))
    )

    assert not is_conjunctive_expression(
        R(Q(x))
    )


def test_conjunctive_expression_with_nested():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert is_conjunctive_expression_with_nested_predicates(
        Q()
    )

    assert is_conjunctive_expression_with_nested_predicates(
        Q(x)
    )

    assert is_conjunctive_expression_with_nested_predicates(
        Q(x) & R(y, C_(1))
    )

    assert is_conjunctive_expression_with_nested_predicates(
        Q(x) & R(Q(y), C_(1))
    )

    assert not is_conjunctive_expression_with_nested_predicates(
        R(x) | R(y)
    )

    assert not is_conjunctive_expression_with_nested_predicates(
        R(x) & R(y) | R(x)
    )

    assert not is_conjunctive_expression_with_nested_predicates(
        ~R(x)
    )


def test_extract_free_variables():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    emptyset = set()
    assert extract_logic_free_variables(Q()) == emptyset
    assert extract_logic_free_variables(Q(C_(1))) == emptyset
    assert extract_logic_free_variables(x) == {x}
    assert extract_logic_free_variables(Q(x, y)) == {x, y}
    assert extract_logic_free_variables(Q(x, C_(1))) == {x}
    assert extract_logic_free_variables(Q(x) & R(y)) == {x, y}
    assert extract_logic_free_variables(EP_(x, Q(x, y))) == {y}
    assert extract_logic_free_variables(Imp_(R(x), Q(x, y))) == {y}
    assert extract_logic_free_variables(Imp_(R(x), Q(y) & Q(x))) == {y}
    assert extract_logic_free_variables(Q(R(y))) == {y}
    assert extract_logic_free_variables(Q(x) | R(y)) == {x, y}
    assert extract_logic_free_variables(~(R(y))) == {y}
    assert extract_logic_free_variables(B_([Q(x), R(y)])) == {x, y}


def test_extract_datalog_predicates():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert extract_logic_predicates(C_(1)) == set()
    assert extract_logic_predicates(Q) == set()

    expression = Q(x)
    assert extract_logic_predicates(expression) == {Q(x)}

    expression = DT.walk(Q(x) & R(y))
    assert extract_logic_predicates(expression) == {Q(x), R(y)}

    expression = DT.walk(B_([Q(x), Q(y) & R(y)]))
    assert extract_logic_predicates(expression) == {Q(x), Q(y), R(y)}

    expression = DT.walk(B_([Q(x), Q(y) & ~R(y)]))
    assert extract_logic_predicates(expression) == {
        Q(x), Q(y), Negation(R(y))
    }


def test_is_linear_rule():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    assert is_linear_rule(T_(Q(C_(1))))
    assert is_linear_rule(T_(Q(C_(x))))
    assert is_linear_rule(Imp_(Q(x), R(x, y)))
    assert is_linear_rule(Imp_(Q(x), Q(x)))
    assert is_linear_rule(DT.walk(Imp_(Q(x), R(x, y) & Q(x))))
    assert is_linear_rule(DT.walk(Imp_(Q(x), R(x, y) & ~Q(x))))
    assert not is_linear_rule(DT.walk(Imp_(Q(x), R(x, y) & Q(x) & Q(y))))
    assert not is_linear_rule(DT.walk(Imp_(Q(x), R(x, y) & Q(x) & ~Q(y))))


class Datalog(
    DatalogProgram,
    ExpressionBasicEvaluator
):
    pass


def test_stratification():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    S = S_('S')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')

    code = DT.walk(B_([
        T_(Q(C_(1), C_(2))),
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), Q(y, x)),
        Imp_(S(x), R(x, y) & S(y))
    ]))

    datalog = Datalog()
    datalog.walk(code)

    strata, stratifyiable = stratify(code, datalog)

    assert stratifyiable
    assert strata == [
        list(code.formulas[:1]),
        list(code.formulas[1: 3]),
        list(code.formulas[3:])
    ]

    code = DT.walk(B_([
        Imp_(Q(x, y), C_(eq)(x, y)),
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), Q(y, x)),
        Imp_(S(x), R(x, y) & T(y)),
        Imp_(T(x), R(x, y) & S(x))
    ]))

    datalog = Datalog()
    datalog.walk(code)

    strata, stratifyiable = stratify(code, datalog)

    assert not stratifyiable
    assert strata == [
        list(code.formulas[:1]),
        list(code.formulas[1: 3]),
        list(code.formulas[3:])
    ]


def test_reachable():
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    S = S_('S')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')

    code = DT.walk(B_([
        Imp_(R(x, y), Q(x, y)),
        Imp_(R(x, y), Q(y, x)),
        Imp_(S(x), R(x, y) & S(y)),
        Imp_(T(x), Q(x, y)),
    ]))

    datalog = Datalog()
    datalog.walk(code)

    reached = reachable_code(code.formulas[-2], datalog)

    assert set(reached.formulas) == set(code.formulas[:-1])


def test_implication_has_existential_variable_in_antecedent():
    Q = S_('Q')
    R = S_('R')
    x = S_('x')
    y = S_('y')
    assert implication_has_existential_variable_in_antecedent(
        Implication(Q(x), R(x, y)),
    )
    assert not implication_has_existential_variable_in_antecedent(
        Implication(Q(x), R(x)),
    )
