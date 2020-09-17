from ..expressions import Symbol


def is_ucq_shattered(ucq):
    """
    A union of conjunctive queries (UCQ) is shattered if it does not contain
    any constant.

    Note that this definition extends to any first-order sentence. This
    function, however, assumes its input to be a UCQ. [1]_.

    .. [1] Van den Broeck, G., and Suciu, D. (2017). Query Processing on
       Probabilistic Data: A Survey. FNT in Databases 7, 197–341.

    """
    for conjunction in ucq.formulas:
        for predicate in conjunction.formulas:
            if any(not isinstance(arg, Symbol) for arg in predicate.args):
                return False
    return True


def _variables_respects_or_add_total_order(variables, total_order):
    """
    Check that variables respect a given total order. If the variable is not
    yet in the total order, it is added to it.

    """
    if len(variables) <= 1:
        return True
    for i in range(len(variables) - 1):
        for j in range(i + 1, len(variables)):
            try:
                idx_i = total_order.index(variables[i])
            except ValueError:
                idx_i = None
            try:
                idx_j = total_order.index(variables[j])
            except ValueError:
                idx_j = None
            if idx_i is None and idx_j is None:
                total_order.append(variables[i])
                total_order.append(variables[j])
            elif idx_i is None:
                total_order.insert(idx_j, variables[i])
            elif idx_j is None:
                total_order.insert(idx_i + 1, variables[j])
            elif idx_i >= idx_j:
                return False
    return True


def is_ucq_ranked(ucq):
    """
    A union of conjunctive queries (UCQ) is ranked if there exists a total
    order on its variables such that whenever x_i, x_j occur in the same atom
    and x_i occurs before x_j, then x_i strictly precedes x_j in the order; in
    particular, no atom contains the same variable twice.

    Note that this definition extends to any first-order sentence. This
    function, however, assumes its input to be a UCQ. [1]_.

    .. [1] Van den Broeck, G., and Suciu, D. (2017). Query Processing on
       Probabilistic Data: A Survey. FNT in Databases 7, 197–341.

    """
    total_order = list()
    for conjunction in ucq.formulas:
        for predicate in conjunction.formulas:
            if not _variables_respects_or_add_total_order(
                predicate.args, total_order
            ):
                return False
    return True
