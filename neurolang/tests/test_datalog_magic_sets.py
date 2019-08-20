from ..datalog import chase
from ..datalog import magic_sets
from .. import expression_walker
from ..datalog import naive_solver

C_ = naive_solver.Constant
S_ = naive_solver.Symbol
Imp_ = naive_solver.Implication
F_ = naive_solver.Fact
Eb_ = naive_solver.ExpressionBlock


class Datalog(
    naive_solver.DatalogBasic,
    expression_walker.ExpressionBasicEvaluator
):
    pass


def test_resolution_works():
    x = S_('X')
    y = S_('Y')
    z = S_('Z')
    anc = S_('anc')
    par = S_('par')
    q = S_('q')
    a = C_('a')
    b = C_('b')
    c = C_('c')
    d = C_('d')

    edb = Eb_([
        F_(par(a, b)),
        F_(par(b, c)),
        F_(par(c, d)),
    ])

    code = Eb_([
        Imp_(q(x), anc(a, x)),
        Imp_(anc(x, y), par(x, y)),
        Imp_(anc(x, y), anc(x, z) & par(z, y)),
    ])

    dl = Datalog()
    dl.walk(code)
    dl.walk(edb)
    goal, mr = magic_sets.magic_rewrite(q(x), dl)

    dl = Datalog()
    dl.walk(mr)
    dl.walk(edb)

    solution = chase.build_chase_solution(dl)
    assert solution[goal].value == {C_((e,)) for e in (b, c, d)}
