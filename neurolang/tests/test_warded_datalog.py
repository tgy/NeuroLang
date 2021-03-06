import pytest

from .. import expression_walker as ew
from .. import expressions
from ..logic import Implication, ExistentialPredicate
from ..existential_datalog import ExistentialDatalog
from ..datalog import Fact
from ..warded_datalog import (NeuroLangNonWardedException,
                              WardedDatalogDangerousVariableCheck,
                              WardedDatalogDangerousVariableExtraction)


class Datalog(ExistentialDatalog, ew.ExpressionBasicEvaluator):
    def function_gt(self, x: int, y: int) -> bool:
        return x > y


C_ = expressions.Constant
S_ = expressions.Symbol
Fa_ = expressions.FunctionApplication
Ep_ = ExistentialPredicate
Eb_ = expressions.ExpressionBlock


def test_warded_walker():
    x = S_('x')
    y = S_('y')
    z = S_('z')
    P = S_('P')
    Q = S_('Q')
    R = S_('R')
    T = S_('T')

    datalog_program = Eb_((
        Fact(Q(x, z)),
        Implication(R(y, z), P(x, y)),
        Implication(R(x, z),
                    Q(x, z) & P(x)),
    ))

    wde = WardedDatalogDangerousVariableExtraction()
    can_be_dangerous = wde.walk(datalog_program)

    extraction_result = dict({R: [{1}]})
    assert can_be_dangerous == extraction_result

    wdc = WardedDatalogDangerousVariableCheck(can_be_dangerous)
    wdc.walk(datalog_program)

    datalog_program = Eb_((
        Implication(Ep_(z, Q(z, x)), P(x)),
        Implication(T(x),
                    Q(x, y) & P(y)),
    ))

    wde = WardedDatalogDangerousVariableExtraction()
    can_be_dangerous = wde.walk(datalog_program)

    extraction_result = dict({Q: [{0}]})
    assert can_be_dangerous == extraction_result

    wdc = WardedDatalogDangerousVariableCheck(can_be_dangerous)
    wdc.walk(datalog_program)


def test_variables_outside_ward():
    P = S_('P')
    Q = S_('Q')
    T = S_('T')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    datalog_program = Eb_((
        Implication(Ep_(z, Q(z, x)), P(x)),
        Implication(T(x),
                    Q(x, y) & P(x)),
    ))

    wde = WardedDatalogDangerousVariableExtraction()
    can_be_dangerous = wde.walk(datalog_program)

    extraction_result = dict({Q: [{0}]})
    assert can_be_dangerous == extraction_result

    wdc = WardedDatalogDangerousVariableCheck(can_be_dangerous)

    with pytest.raises(
        NeuroLangNonWardedException, match=r".*outside the ward.*"
    ):
        wdc.walk(datalog_program)


def test_more_one_atom():
    P = S_('P')
    Q = S_('Q')
    R = S_('R')
    S = S_('S')
    T = S_('T')
    x = S_('x')
    y = S_('y')
    z = S_('z')

    datalog_program = Eb_((
        Implication(Ep_(z, Q(z, x)), P(x)),
        Implication(Ep_(y, R(y, x)), P(x)),
        Implication(T(x),
                    Q(x, y) & P(y) & R(x, z) & S(z)),
    ))

    wde = WardedDatalogDangerousVariableExtraction()
    can_be_dangerous = wde.walk(datalog_program)

    extraction_result = dict({Q: [{0}], R: [{0}]})
    assert can_be_dangerous == extraction_result

    wdc = WardedDatalogDangerousVariableCheck(can_be_dangerous)

    with pytest.raises(
        NeuroLangNonWardedException, match=r".*that appear in more than.*"
    ):
        wdc.walk(datalog_program)


def test_warded_chase():
    hsbc = C_('HSBC')
    hsb = C_('HSB')
    iba = C_('IBA')

    _company = S_('_Company')
    company = S_('Company')
    controls = S_('Controls')
    owns = S_('Owns')
    stock = S_('Stock')
    PSC = S_('PSC')
    strong_link = S_('StrongLink')

    x = S_('x')
    p = S_('p')
    s = S_('s')
    y = S_('y')

    datalog_program = Eb_((
        Fact(_company(hsbc)),
        Fact(_company(hsb)),
        Fact(_company(iba)),
        Fact(controls(hsbc, hsb)),
        Fact(controls(hsb, iba)),
        Implication(Ep_(p, Ep_(s, owns(p, s, x))), company(x)),
        Implication(stock(x, s), owns(p, s, x)),
        Implication(PSC(x, p), owns(p, s, x)),
        Implication(Ep_(s, owns(p, s, y)),
                    PSC(x, p) & controls(x, y)),
        Implication(strong_link(x, y),
                    PSC(x, p) & PSC(y, p)),
        Implication(Ep_(p, Ep_(s, owns(p, s, x))), strong_link(x, y)),
        Implication(Ep_(p, Ep_(s, owns(p, s, y))), strong_link(x, y)),
        Implication(company(x), stock(x, s)),
        Implication(company(x), _company(x)),
    ))

    wde = WardedDatalogDangerousVariableExtraction()
    can_be_dangerous = wde.walk(datalog_program)

    extraction_result = dict({owns: [{0, 1}, {1}, {0, 1}, {0, 1}]})
    assert can_be_dangerous == extraction_result

    wdc = WardedDatalogDangerousVariableCheck(can_be_dangerous)
    wdc.walk(datalog_program)

    dl = Datalog()
    dl.walk(datalog_program)
