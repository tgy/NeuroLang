from typing import AbstractSet

import pytest

from ...expression_walker import ExpressionBasicEvaluator, IdentityWalker
from ...expressions import (Constant, ExpressionBlock, FunctionApplication,
                            Lambda, NeuroLangException, Query, Symbol,
                            is_leq_informative)
from ...logic import Union, ExistentialPredicate, Implication
from .. import DatalogProgram, Fact
from ..expressions import TranslateToLogic

S_ = Symbol
C_ = Constant
Imp_ = Implication
F_ = FunctionApplication
L_ = Lambda
B_ = ExpressionBlock
EP_ = ExistentialPredicate
Q_ = Query
T_ = Fact


class DatalogTranslator(
    TranslateToLogic, IdentityWalker
):
    pass


DT = DatalogTranslator()


class Datalog(
    DatalogProgram,
    ExpressionBasicEvaluator
):
    pass


def test_facts_constants():
    dl = Datalog()

    f1 = T_(S_('Q')(C_(1), C_(2)))

    dl.walk(DT.walk(f1))

    assert 'Q' in dl.symbol_table
    assert isinstance(dl.symbol_table['Q'], Constant[AbstractSet])
    fact_set = dl.symbol_table['Q']
    assert isinstance(fact_set, Constant)
    assert is_leq_informative(fact_set.type, AbstractSet)
    expected_result = {C_((C_(1), C_(2)))}
    assert expected_result == fact_set.value

    f2 = T_(S_('Q')(C_(3), C_(4)))
    dl.walk(DT.walk(f2))
    assert (
        {C_((C_(1), C_(2))), C_((C_(3), C_(4)))} ==
        fact_set.value
    )


def test_atoms_variables():
    dl = Datalog()

    eq = S_('equals')
    x = S_('x')
    y = S_('y')
    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    T = S_('T')  # noqa: N806

    f1 = Imp_(Q(x,), eq(x, x))

    dl.walk(DT.walk(f1))

    assert 'Q' in dl.symbol_table
    assert isinstance(dl.symbol_table['Q'], Union)
    fact = dl.symbol_table['Q'].formulas[-1]
    assert isinstance(fact, Implication)
    assert isinstance(fact.consequent, FunctionApplication)
    assert fact.consequent.functor is Q
    assert fact.consequent.args == (x,)
    assert fact.antecedent == eq(x, x)

    f2 = Imp_(T(x, y), eq(x, y))

    dl.walk(DT.walk(f2))

    assert 'T' in dl.symbol_table
    assert isinstance(dl.symbol_table['T'], Union)
    fact = dl.symbol_table['T'].formulas[-1]
    assert isinstance(fact, Implication)
    assert isinstance(fact.consequent, FunctionApplication)
    assert fact.consequent.functor is T
    assert fact.consequent.args == (x, y)
    assert fact.antecedent == eq(x, y)

    f3 = Imp_(R(x, C_(1)), eq(x, x))
    dl.walk(DT.walk(f3))

    assert 'R' in dl.symbol_table
    assert isinstance(dl.symbol_table['R'], Union)
    fact = dl.symbol_table['R'].formulas[-1]
    assert isinstance(fact, Implication)
    assert isinstance(fact.consequent, FunctionApplication)
    assert fact.consequent.functor is R
    assert fact.consequent.args == (x, C_(1))
    assert fact.antecedent == eq(x, x)

    with pytest.raises(NeuroLangException):
        dl.walk(DT.walk(Imp_(Q(x), ...)))

    with pytest.raises(NeuroLangException):
        dl.walk(DT.walk(Imp_(Q(x, y), eq(x, y))))


def test_intensional_extensional_database():

    dl = Datalog()

    Q = S_('Q')  # noqa: N806
    R0 = S_('R0')  # noqa: N806
    R = S_('R')  # noqa: N806
    T = S_('T')  # noqa: N806
    x = S_('x')
    y = S_('y')
    z = S_('z')

    extensional = ExpressionBlock((
        T_(Q(C_(1), C_(1))),
        T_(Q(C_(1), C_(2))),
        T_(Q(C_(1), C_(4))),
        T_(Q(C_(2), C_(4))),
        T_(R0(C_('a'), C_(1), C_(3))),
    ))

    intensional = ExpressionBlock((
        Imp_(R(x, y, z), R0(x, y, z)),
        Imp_(R(x, y, z), Q(x, y) & Q(y, z)),
        Imp_(T(x, z), Q(x, y) & Q(y, z)),
    ))

    dl.walk(DT.walk(extensional))

    edb = dl.extensional_database()

    assert edb.keys() == {'R0', 'Q'}

    assert edb['Q'] == C_(frozenset((
        C_((C_(1), C_(1))),
        C_((C_(1), C_(2))),
        C_((C_(1), C_(4))),
        C_((C_(2), C_(4))),
    )))

    assert edb['R0'] == C_(frozenset((
        C_((C_('a'), C_(1), C_(3))),
    )))

    dl.walk(DT.walk(intensional))
    edb = dl.extensional_database()

    assert edb.keys() == {'R0', 'Q'}

    assert edb['Q'] == C_(frozenset((
        C_((C_(1), C_(1))),
        C_((C_(1), C_(2))),
        C_((C_(1), C_(4))),
        C_((C_(2), C_(4))),
    )))

    assert edb['R0'] == C_(frozenset((
        C_((C_('a'), C_(1), C_(3))),
    )))

    idb = dl.intensional_database()
    assert len(idb) == 2
    assert len(idb['R'].formulas) == 2
    assert len(idb['T'].formulas) == 1


def test_not_conjunctive():

    dl = Datalog()

    Q = S_('Q')  # noqa: N806
    R = S_('R')  # noqa: N806
    x = S_('x')
    y = S_('y')

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x, y), R(x) | R(y)))

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x, y), R(x) & R(y) | R(x)))

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x, y), ~R(x)))

    with pytest.raises(NeuroLangException):
        dl.walk(Imp_(Q(x, y), R(Q(x))))
