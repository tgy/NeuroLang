import numpy as np

from ..expressions import Symbol, Constant, ExpressionBlock
from ..expression_pattern_matching import add_match
from ..solver_datalog_naive import Fact, Implication, DatalogBasic
from ..graphical_model import (
    produce, GraphicalModel, GDatalogToGraphicalModelTranslator,
    substitute_dterm, gdatalog2gm, sort_rvs, delta_infer1, GraphicalModelSolver
)
from ..generative_datalog import (
    DeltaTerm, DeltaAtom, GenerativeDatalogSugarRemover
)

C_ = Constant
S_ = Symbol

P = S_('P')
Q = S_('Q')
R = S_('R')
Z = S_('Z')
x = S_('x')
y = S_('y')
z = S_('z')
a = C_(2)
b = C_(3)
p = S_('p')
p_a = C_(0.2)
p_b = C_(0.7)
bernoulli = C_('bernoulli')

program_1 = ExpressionBlock((
    Fact(P(a)),
    Fact(P(b)),
    Implication(Q(x, y),
                P(x) & P(y)),
))
program_2 = ExpressionBlock((
    Fact(P(a, b)),
    Fact(P(b, a)),
    Implication(Q(x), P(x, x)),
    Implication(R(x),
                P(x, y) & P(y, x)),
))
program_3 = ExpressionBlock((
    Fact(P(a)),
    Fact(P(b)),
    Implication(DeltaAtom(Q, (x, DeltaTerm(bernoulli, C_(0.7)))), P(x)),
    Implication(Z(x, C_(0.2)), Q(x, C_(1))),
    Implication(Z(x, C_(0.8)), Q(x, C_(0))),
    Implication(DeltaAtom(R, (x, DeltaTerm(bernoulli, y))), Z(x, y)),
))
program_4 = ExpressionBlock((
    Fact(P(a)),
    Implication(DeltaAtom(Q, (x, DeltaTerm(bernoulli, C_(0.2)))), P(x)),
    Implication(DeltaAtom(R, (x, DeltaTerm(bernoulli, C_(0.9)))), Q(x, y)),
))


class GenerativeDatalogSugarRemoverTest(
    GenerativeDatalogSugarRemover, DatalogBasic
):
    @add_match(Implication(DeltaAtom, ...))
    def ignore_gdatalog_rule(self, rule):
        return rule


def sugar_remove(program):
    return GenerativeDatalogSugarRemoverTest().walk(program)


def test_produce():
    fact = Fact(P(a))
    rule = Implication(Q(x), P(x))
    result = produce(rule, [fact])
    assert result is not None
    assert result == Fact(Q(a))


def test_substitute_dterm():
    fa = DeltaAtom(Q, (x, DeltaTerm(bernoulli, p)))
    assert substitute_dterm(fa, a) == Q(x, a)


def test_delta_produce():
    fact_a = Fact(P(a, p_a))
    fact_b = Fact(P(b, p_b))
    rule = Implication(Q(x, DeltaTerm(bernoulli, p)), P(x, p))
    assert produce(rule, [fact_a]) == Fact(Q(a, DeltaTerm(bernoulli, p_a)))
    assert produce(rule, [fact_b]) == Fact(Q(b, DeltaTerm(bernoulli, p_b)))


def test_delta_infer1():
    fact_a = Fact(P(a, p_a))
    fact_b = Fact(P(b, p_b))
    rule = Implication(DeltaAtom(Q, (x, DeltaTerm(bernoulli, p))), P(x, p))
    result = delta_infer1(rule, frozenset({fact_a, fact_b}))
    result_as_dict = dict(result)
    expected_dist = {
        frozenset({Fact(Q(a, C_(0))), Fact(Q(b, C_(0)))}):
        (1 - p_a.value) * (1 - p_b.value),
        frozenset({Fact(Q(a, C_(1))), Fact(Q(b, C_(0)))}):
        p_a.value * (1 - p_b.value),
        frozenset({Fact(Q(a, C_(0))), Fact(Q(b, C_(1)))}):
        (1 - p_a.value) * p_b.value,
        frozenset({Fact(Q(a, C_(1))), Fact(Q(b, C_(1)))}):
        p_a.value * p_b.value,
    }
    for outcome, prob in expected_dist.items():
        assert outcome in result_as_dict
        assert np.allclose([prob], [result_as_dict[outcome].value])


def test_sort_rv():
    gm = gdatalog2gm(program_1)
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['Q'] == {'Q_1'}
    assert sort_rvs(gm) == ['P', 'Q_1', 'Q']


def test_gdatalog_translation_to_gm():
    gm = gdatalog2gm(program_1)
    assert set(gm.rv_to_cpd_functor.keys()) == {'P', 'Q', 'Q_1'}
    assert gm.parents['P'] == set()
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['Q'] == {'Q_1'}

    gm = gdatalog2gm(program_2)
    assert gm.parents['Q_1'] == {'P'}
    assert gm.parents['Q'] == {'Q_1'}
    assert gm.parents['R_1'] == {'P'}
    assert gm.parents['R'] == {'R_1'}


def test_delta_term():
    program = ExpressionBlock((
        Fact(P(a)),
        Implication(Q(x, DeltaTerm(bernoulli, x)), P(x)),
    ))
    program = sugar_remove(program)
    gm = gdatalog2gm(program)
    assert 'Q_1' in gm.rv_to_cpd_functor


def test_2levels_model():
    program = sugar_remove(program_4)
    gm = gdatalog2gm(program)

    program = sugar_remove(
        ExpressionBlock((
            Fact(P(a)),
            Fact(P(b)),
            Implication(Q(x, DeltaTerm(bernoulli, C_(0.5))), P(x)),
            Implication(Z(C_(0.1)), Q(x, C_(0))),
            Implication(Z(C_(0.9)), Q(x, C_(1))),
            Implication(R(x, DeltaTerm(bernoulli, z)),
                        Q(x, y) & Z(z)),
        ))
    )
    gm = gdatalog2gm(program)
    assert set(gm.rv_to_cpd_functor.keys()) == {
        'P', 'Q_1', 'Q', 'Z_1', 'Z_2', 'Z', 'R_1', 'R'
    }


def test_gm_solver():
    outcomes = GraphicalModelSolver().walk(gdatalog2gm(program_1))
    assert len(outcomes) == 1
    outcome = list(outcomes.items())[0]
    assert outcome[1].value == 1.0

    outcomes = GraphicalModelSolver().walk(gdatalog2gm(program_4))
    expected_outcomes = {
        frozenset({Fact(P(a)),
                   Fact(Q(a, C_(1))),
                   Fact(R(a, C_(1)))}):
        C_(0.2 * 0.9),
        frozenset({Fact(P(a)),
                   Fact(Q(a, C_(1))),
                   Fact(R(a, C_(0)))}):
        C_(0.2 * 0.1),
        frozenset({Fact(P(a)),
                   Fact(Q(a, C_(0))),
                   Fact(R(a, C_(0)))}):
        C_(0.8 * 0.1),
        frozenset({Fact(P(a)),
                   Fact(Q(a, C_(0))),
                   Fact(R(a, C_(1)))}):
        C_(0.8 * 0.9),
    }
    for outcome, prob in expected_outcomes.items():
        assert outcome in outcomes
        assert np.allclose([prob.value], [outcomes[outcome].value])

    outcomes = GraphicalModelSolver().walk(gdatalog2gm(program_3))
