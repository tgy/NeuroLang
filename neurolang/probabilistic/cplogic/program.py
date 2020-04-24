import typing

from ...datalog import DatalogProgram
from ...exceptions import NeuroLangException
from ...expression_pattern_matching import add_match
from ...expression_walker import ExpressionWalker, PatternWalker
from ...expressions import Constant, ExpressionBlock, Symbol
from ...logic import Implication
from ..expression_processing import (
    block_contains_probabilistic_facts,
    build_probabilistic_fact_set,
    check_probabilistic_choice_set_probabilities_sum_to_one,
    concatenate_to_expression_block,
    group_probabilistic_facts_by_pred_symb,
    is_probabilistic_fact,
)


class CPLogicMixin(PatternWalker):
    """
    Datalog extended with probabilistic facts semantics from ProbLog.

    It adds a probabilistic database which is a set of probabilistic facts.

    Probabilistic facts are stored in the symbol table of the program such that
    the key in the symbol table is the symbol of the predicate of the
    probabilsitic fact and the value is the probabilistic fact itself.
    """

    pfact_pred_symb_set_symb = Symbol("__pfact_pred_symb_set_symb__")
    pchoice_pred_symb_set_symb = Symbol("__pchoice_pred_symb_set_symb__")

    @property
    def predicate_symbols(self):
        return (
            set(self.intensional_database())
            | set(self.extensional_database())
            | set(self.pfact_pred_symbs)
            | set(self.pchoice_pred_symbs)
        )

    @property
    def pfact_pred_symbs(self):
        return self._get_pred_symbs(self.pfact_pred_symb_set_symb)

    @property
    def pchoice_pred_symbs(self):
        return self._get_pred_symbs(self.pchoice_pred_symb_set_symb)

    def probabilistic_facts(self):
        """Return probabilistic facts of the symbol table."""
        return {
            k: v
            for k, v in self.symbol_table.items()
            if k in self.pfact_pred_symbs
        }

    def _get_pred_symbs(self, set_symb):
        return self.symbol_table.get(
            set_symb, Constant[typing.AbstractSet](set())
        ).value

    def extensional_database(self):
        exclude = (
            self.protected_keywords
            | self.pfact_pred_symbs
            | self.pchoice_pred_symbs
        )
        ret = self.symbol_table.symbols_by_type(typing.AbstractSet)
        for keyword in exclude:
            if keyword in ret:
                del ret[keyword]
        return ret

    def add_probabilistic_facts_from_tuples(self, symbol, iterable):
        self._register_prob_pred_symb_set_symb(
            symbol, self.pfact_pred_symb_set_symb
        )
        type_, iterable = self.infer_iterable_type(iterable)
        self._check_iterable_prob_type(type_)
        constant = Constant[typing.AbstractSet[type_]](
            self.new_set(iterable), auto_infer_type=False, verify_type=False,
        )
        symbol = symbol.cast(constant.type)
        self.symbol_table[symbol] = constant

    def add_probabilistic_choice_from_tuples(self, symbol, iterable):
        """
        Add a probabilistic choice to the symbol table.

        """
        self._register_prob_pred_symb_set_symb(
            symbol, self.pchoice_pred_symb_set_symb
        )
        type_, iterable = self.infer_iterable_type(iterable)
        self._check_iterable_prob_type(type_)
        if symbol in self.symbol_table:
            raise NeuroLangException("Symbol already used")
        ra_set = Constant[typing.AbstractSet](
            self.new_set(iterable), auto_infer_type=False, verify_type=False,
        )
        check_probabilistic_choice_set_probabilities_sum_to_one(ra_set)
        self.symbol_table[symbol] = ra_set

    @staticmethod
    def _check_iterable_prob_type(iterable_type):
        if not (
            issubclass(iterable_type.__origin__, typing.Tuple)
            and iterable_type.__args__[0] is float
        ):
            raise NeuroLangException(
                "Expected tuples to have a probability as their first element"
            )

    @add_match(ExpressionBlock, block_contains_probabilistic_facts)
    def block_with_probabilistic_facts(self, code):
        pfacts, other_expressions = group_probabilistic_facts_by_pred_symb(
            code
        )
        for pred_symb, pfacts in pfacts.items():
            self._register_prob_pred_symb_set_symb(
                pred_symb, self.pfact_pred_symb_set_symb
            )
            if len(pfacts) > 1:
                self.symbol_table[pred_symb] = build_probabilistic_fact_set(
                    pred_symb, pfacts
                )
            else:
                self.walk(list(pfacts)[0])
        self.walk(ExpressionBlock(other_expressions))

    def _register_prob_pred_symb_set_symb(self, pred_symb, set_symb):
        if set_symb.name not in self.protected_keywords:
            self.protected_keywords.add(set_symb.name)
        if set_symb not in self.symbol_table:
            self.symbol_table[set_symb] = Constant[typing.AbstractSet](set())
        self.symbol_table[set_symb] = Constant[typing.AbstractSet](
            self.symbol_table[set_symb].value | {pred_symb}
        )

    @add_match(Implication, is_probabilistic_fact)
    def probabilistic_fact(self, expression):
        pred_symb = expression.consequent.body.functor
        if pred_symb not in self.symbol_table:
            self.symbol_table[pred_symb] = ExpressionBlock(tuple())
        self.symbol_table[pred_symb] = concatenate_to_expression_block(
            self.symbol_table[pred_symb], [expression]
        )
        return expression


class CPLogicProgram(CPLogicMixin, DatalogProgram, ExpressionWalker):
    pass
