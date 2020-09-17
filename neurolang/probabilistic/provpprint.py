import operator

from ..expression_pattern_matching import add_match
from ..expression_walker import ExpressionWalker
from ..expressions import Constant, FunctionApplication, Expression, Symbol


class ProvenancePrettyPrinter(ExpressionWalker):
    @add_match(FunctionApplication(Constant(operator.add), ...))
    def addition(self, fa):
        return " + ".join(self.walk(arg) for arg in fa.args)

    @add_match(FunctionApplication(Constant(operator.mul), ...))
    def multiplication(self, fa):
        return " * ".join(self.walk(arg) for arg in fa.args)

    @add_match(Expression)
    def expression(self, exp):
        return repr(exp)


def provpprint(expression):
    pprinter = ProvenancePrettyPrinter()
    return pprinter.walk(expression)
