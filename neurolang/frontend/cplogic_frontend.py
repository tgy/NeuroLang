from ..datalog.aggregation import Chase
from . import QueryBuilderDatalog, RegionFrontendDatalogSolver


class CPLogicFrontend(QueryBuilderDatalog):
    def __init__(self, solver=None):
        if solver is None:
            solver = RegionFrontendDatalogSolver()
        super().__init__(solver, chase_class=Chase)
