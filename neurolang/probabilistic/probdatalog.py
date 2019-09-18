import uuid
import itertools

from ..expressions import Expression, Constant, Symbol, FunctionApplication
from ..datalog.expressions import Fact, Implication, Disjunction, Conjunction
from ..exceptions import NeuroLangException
from ..datalog.chase import Chase
from ..datalog import DatalogProgram
from ..expression_pattern_matching import add_match
from ..expression_walker import PatternWalker
from ..probabilistic.ppdl import is_gdatalog_rule
from .ppdl import (
    concatenate_to_expression_block, get_antecedent_formulas, get_dterm,
    get_dterm_index, DeltaTerm
)


class ProbFact(Fact):
    '''
    A fact that occurs with a certain probability.

    A probabilistic fact is of the form p :: R(x). where R(x) is an atom and p
    is a probability.

    Notes
    -----
    - If x is a variable, its possible values are determined a typing unary
      predicate within the rules of the program.
    - If p is a variable, then it's a learnable parameter whose most probable
      value should be inferred from data.
    '''
    def __init__(self, probability, consequent):
        super().__init__(consequent)
        if not isinstance(probability, Expression):
            raise NeuroLangException('The probability must be an expression')
        self.probability = probability

    def __repr__(self):
        return 'ProbFact{{{} : {} \u2190 {}}}'.format(
            repr(self.probability), repr(self.consequent), True
        )


class ProbChoice(Implication):
    '''
    A choice over probabilistic facts.

    A probabilistic choice is of the form p_1 :: R(x_1), ..., p_n :: R(x_n).
    where p_1, ..., p_n are probabilities whose sum must be less than or equal
    to 1, where x_1, ..., x_n are tuples of terms and where R is a predicate.

    One of the probabilistic fact is chosen to be true with its corresponding
    probability while the other will be false (i.e. not 2 of the probabilistic
    facts can be true at the same time).
    '''
    def __init__(self, probfacts):
        sum_const_probs = 0.
        for probfact in probfacts:
            if isinstance(probfact.probability, Constant[float]):
                sum_const_probs += probfact.probability.value
        if sum_const_probs > 1:
            raise NeuroLangException(
                'Sum of probabilities in probabilistic '
                'choice cannot be greater than 1'
            )
        consequent = Disjunction(probfacts)
        super().__init__(consequent, Constant[bool](True))


class ProbDatalogProgram(DatalogProgram):
    '''
    Datalog extended with probabilistic facts semantics from ProbLog.

    It adds a probabilistic database which is a set of probabilistic facts.

    Probabilistic facts are stored in the symbol table of the program such that
    the key in the symbol table is the symbol of the predicate of the
    probabilsitic fact and the value is the probabilistic fact itself.
    '''
    @add_match(ProbFact)
    def probabilistic_fact(self, probfact):
        predicate = probfact.consequent.functor
        if predicate in self.symbol_table:
            raise NeuroLangException(f'Predicate {predicate} already defined')
        self.symbol_table[predicate] = probfact
        return probfact

    def probabilistic_database(self):
        '''Returns probabilistic facts of the symbol table.'''
        return {
            k: v
            for k, v in self.symbol_table.items()
            if isinstance(v, ProbFact)
        }


class GDatalogToProbDatalogTranslator(PatternWalker):
    '''
    Translate a GDatalog program to a ProbDatalog program.

    A GDatalog probabilsitic rule whose delta term's distribution is finite can
    be represented as a probabilistic choice. If the distribution is a
    bernoulli distribution, it can be represented as probabilistic fact.
    '''
    @add_match(Implication, is_gdatalog_rule)
    def rule(self, rule):
        '''
        Translate a GDatalog rule whose delta term is bernoulli distributed to
        an expression block containing a probabilistic fact and a
        (deterministic) rule.

        Example
        -------
        Let tau be the following walked GDatalog rule (with syntactic sugar)

            Q(x_1, ..., x_{i-1}, B[[0.3]], x_{i+1}, ..., x_n) :- P(x).

        where x is the tuple (x_1, ..., x_{i-1}, x_{i+1}, ..., x_n) and where
        P(x) is a conjunction of atoms over x.

        The following block is returned

            0.3 :: ProbFact_Q_<uid_1>(x).
            Q(x) :- P(x), ProbFact_Q_<uid_1>(x).

        '''
        datom = rule.consequent
        dterm = get_dterm(datom)
        predicate = datom.functor
        if not dterm.functor.name == 'bernoulli':
            raise NeuroLangException(
                'Other distributions than bernoulli are not supported'
            )
        probability = dterm.args[0]
        probfact_predicate = Symbol(
            'ProbFact_{}_{}'.format(predicate.name, uuid.uuid1())
        )
        terms = tuple(
            arg for arg in datom.args if not isinstance(arg, DeltaTerm)
        )
        probfact_atom = probfact_predicate(*terms)
        new_rule = Implication(
            predicate(*terms),
            conjunct_expressions(rule.antecedent, probfact_atom)
        )
        return Disjunction((
            self.walk(ProbFact(probability,
                               probfact_atom)), self.walk(new_rule)
        ))

    @add_match(Disjunction)
    def disjunction(self, disjunction):
        formulas = []
        for formula in disjunction.formulas:
            result = self.walk(formula)
            if isinstance(result, Disjunction):
                formulas += list(result.formulas)
            else:
                formulas.append(result)
        return Disjunction(formulas)


def conjunct_formulas(f1, f2):
    '''Conjunct two logical expressions.'''
    if isinstance(f1, Conjunction) and isinstance(f2, Conjunction):
        return Conjunction(list(f1.formulas) + list(f2.formulas))
    elif isinstance(f1, Conjunction):
        return Conjunction(list(f1.formulas) + [f2])
    elif isinstance(f2, Conjunction):
        return Conjunction([f1] + list(f2.formulas))
    else:
        return Conjunction([f1, f2])


class GDatalogToProbDatalog(
    GDatalogToProbDatalogTranslator, ProbDatalogProgram
):
    pass


def get_antecedent_predicates(rule):
    antecedent_literals = get_antecedent_formulas(rule)
    return [literal.functor for literal in antecedent_literals]


def substitute_dterm(atom, substitute):
    new_args = tuple(
        substitute if isinstance(arg, DeltaTerm) else arg for arg in atom.args
    )
    return FunctionApplication[atom.type](atom.functor, new_args)


def split_probfacts_and_background_knowledge(program):
    '''
    Given a Datalog program with probabilistic facts, seperate
    probabilistic facts from the background knowledege.

    The background knowledge contains all rules and ground
    non-probabilistic facts.
    '''
    probabilistic_facts = ExpressionBlock(tuple())
    background_knowledge = ExpressionBlock(tuple())
    for expression in program:
        if isinstance(expression, ProbabilisticFact):
            probabilistic_facts = concatenate_to_expression_block(
                probabilistic_facts, expression
            )
        else:
            background_knowledge = concatenate_to_expression_block(
                background_knowledge, expression
            )
    return probabilistic_facts, background_knowledge


def get_probfacts_possible_ground_substitutions_in_interpretation(
    probfacts, background_knowledge, interpretation
):
    '''
    Generate all possible substitutions that ground probabilistic facts in a
    given interpretation.

    It is assumed that the interpretation contains an explicit definition of
    the different types in the form of fully-observable unary predicates (see
    section 3.1 of "Learning the Parameters of Probabilistic Logic Programs"
    by Gutmann et al.).

    For example, let `p :: P(x)` be a probabilistic fact and let `Q(x) :- T(x),
    P(x)` be a rule in the background knowledge of the program. `T` is the
    unary predicate that defines the type of the variable x which is also the
    first argument of the literal P(x) whose predicate is the same as the one
    of the probabilistic fact. The interpretation (a set of ground facts) must
    contain the `k` ground facts `T(a_1), ..., T(a_k)`.
    '''
    pass


def generate_possible_outcomes(probabilistic_program):
    pfacts, base_program = split_program_probabilistic_deterministic(
        probabilistic_program
    )
    for facts in itertools.product(
        *[(({pfact.fact}, pfact.probability), (set(), 1.0 - pfact.probability))
          for pfact in pfacts]
    ):
        probability = None
        facts = set.union(*facts)
        program = concatenate_to_expression_block(base_program, facts)
        chase = Chase()
        solution = chase.build_chase_solution(program)
