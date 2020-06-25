class NeuroLangException(Exception):
    """Base class for NeuroLang Exceptions"""

    pass


class UnexpectedExpressionError(NeuroLangException):
    pass


class NeuroLangNotImplementedError(NeuroLangException):
    pass


class ForbiddenExpressionError(NeuroLangException):
    pass


class ForbiddenDisjunctionError(ForbiddenExpressionError):
    pass


class ForbiddenExistentialError(ForbiddenExpressionError):
    pass


class ForbiddenDuplicatedAntecedentPredicateSymbol(ForbiddenExpressionError):
    pass


class RelationalAlgebraError(NeuroLangException):
    pass


class RelationalAlgebraNotImplementedError(
    RelationalAlgebraError, NotImplementedError
):
    pass
