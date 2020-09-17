from ...expressions import Constant, Symbol
from ...logic import Conjunction, Disjunction
from ..lifted_processing import is_ucq_ranked, is_ucq_shattered

P = Symbol("P")
Q = Symbol("Q")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
h = Symbol("h")
a = Constant("a")

UCQ_SHATTERED_RANKED = Disjunction(
    (Conjunction((Q(x, y), P(y, z))), Conjunction((Q(y, z), Q(z, h))))
)
UCQ_SHATTERED_NOT_RANKED = Disjunction(
    (Conjunction((Q(x, y), P(y))), Conjunction((P(z), Q(y, x))))
)
UCQ_SHATTERED_NOT_RANKED_BIS = Disjunction(
    (Conjunction((Q(x, y), P(y))), Conjunction((P(z), Q(x, x))))
)
UCQ_NOT_SHATTERED = Disjunction(
    (Conjunction((Q(a, y), P(y))), Conjunction((P(z), Q(x, a))))
)


def test_is_ucq_ranked():
    assert is_ucq_ranked(Disjunction(tuple()))
    assert is_ucq_ranked(UCQ_SHATTERED_RANKED)
    assert not is_ucq_ranked(UCQ_SHATTERED_NOT_RANKED)
    assert not is_ucq_ranked(UCQ_SHATTERED_NOT_RANKED_BIS)


def test_is_ucq_shattered():
    assert is_ucq_shattered(Disjunction(tuple()))
    assert is_ucq_shattered(UCQ_SHATTERED_RANKED)
    assert is_ucq_shattered(UCQ_SHATTERED_NOT_RANKED)
    assert is_ucq_shattered(UCQ_SHATTERED_NOT_RANKED_BIS)
    assert not is_ucq_shattered(UCQ_NOT_SHATTERED)
