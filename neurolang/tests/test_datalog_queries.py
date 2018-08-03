import pytest
from pytest import raises

import typing

from .. import solver
from .. import expressions
from ..neurolang import TypedSymbolTable


def test_simple_symbol_query():
    ds = solver.DatalogSolver(TypedSymbolTable())

    for s in range(5):
        sym = expressions.Symbol[int](str(s))
        ds.symbol_table[sym] = expressions.Constant[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[expressions.Symbol('gt')] = expressions.Constant(gt)

    x = expressions.Symbol[int]('x')
    query = expressions.Query[typing.AbstractSet[int]](
        x, ds.symbol_table['gt'](ds.symbol_table['3'], x)
    )

    res = ds.walk(query)
    print(res)
    assert isinstance(res, expressions.Constant[query.type])
    assert expressions.type_validation_value(
        res.value, typing.AbstractSet[int]
    )
    assert set(['0', '1', '2']) == res.value


def test_multiple_symbol_query():
    ds = solver.DatalogSolver(TypedSymbolTable())

    for s in range(5):
        sym = expressions.Symbol[int](str(s))
        ds.symbol_table[sym] = expressions.Constant[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[expressions.Symbol('gt')] = expressions.Constant(gt)

    x = expressions.Symbol[int]('x')
    y = expressions.Symbol[int]('y')

    query = expressions.Query[typing.AbstractSet[typing.Tuple[int, int]]](
        expressions.Constant((x, y)),
        ds.symbol_table['gt'](ds.symbol_table['3'], x) &
        ds.symbol_table['gt'](x, y)
    )

    res = ds.walk(query)
    assert isinstance(res, expressions.Constant[query.type])
    assert expressions.type_validation_value(
        res.value, typing.AbstractSet[typing.Tuple[int, int]]
    )
    assert set(
        (str(x), str(y))
        for x in range(5)
        for y in range(5)
        if 3 > x and x > y
    ) == res.value


@pytest.mark.skip
def test_too_many_symbols_in_query_body():
    ds = solver.DatalogSolver(TypedSymbolTable())

    for s in range(5):
        sym = expressions.Symbol[int](str(s))
        ds.symbol_table[sym] = expressions.Constant[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[expressions.Symbol('gt')] = expressions.Constant(gt)

    x = expressions.Symbol[int]('x')
    y = expressions.Symbol[int]('y')
    z = expressions.Symbol[int]('z')

    query = expressions.Query[typing.AbstractSet[typing.Tuple[int, int]]](
        expressions.Constant((x, y)),
        ds.symbol_table['gt'](ds.symbol_table['3'], x) &
        ds.symbol_table['gt'](x, y) & ds.symbol_table['gt'](y, z)
    )

    with raises(NotImplementedError):
        ds.walk(query)


def test_existential_predicate():
    ds = solver.DatalogSolver(TypedSymbolTable())

    for s in range(5):
        sym = expressions.Symbol[int](str(s))
        ds.symbol_table[sym] = expressions.Constant[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[expressions.Symbol('gt')] = expressions.Constant(gt)

    x = expressions.Symbol[int]('x')

    expression = expressions.ExistentialPredicate(
        x, ds.symbol_table['gt'](x, expressions.Constant(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert res.value

    expression = expressions.ExistentialPredicate(
        x, ds.symbol_table['gt'](x, expressions.Constant(10))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert not res.value


def test_existential_predicate_trivial():
    ds = solver.DatalogSolver(TypedSymbolTable())

    for s in range(5):
        sym = expressions.Symbol[int](str(s))
        ds.symbol_table[sym] = expressions.Constant[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[expressions.Symbol('gt')] = expressions.Constant(gt)

    x = expressions.Symbol[int]('x')

    expression = expressions.ExistentialPredicate(
        x,
        expressions.Constant(True) |
        ds.symbol_table['gt'](x, expressions.Constant(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert res.value

    expression = expressions.ExistentialPredicate(
        x,
        expressions.Constant(False) &
        ds.symbol_table['gt'](x, expressions.Constant(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.Constant)
    assert res.type is bool
    assert not res.value


def test_existential_predicate_not_solved():
    ds = solver.DatalogSolver(TypedSymbolTable())

    for s in range(5):
        sym = expressions.Symbol[int](str(s))
        ds.symbol_table[sym] = expressions.Constant[int](s)

    def gt(a: int, b: int) -> bool:
        return a > b

    ds.symbol_table[expressions.Symbol('gt')] = expressions.Constant(gt)

    x = expressions.Symbol[int]('x')
    y = expressions.Symbol[int]('y')

    expression = expressions.ExistentialPredicate(
        x,
        ds.symbol_table['gt'](x, expressions.Constant(2)) &
        ds.symbol_table['gt'](y, expressions.Constant(2))
    )
    res = ds.walk(expression)
    assert isinstance(res, expressions.ExistentialPredicate)
    assert res.head == expression.head
    assert res.body == res.body