# Copyright 2021 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from typing import Dict, Union, List, Set, Optional
from itertools import product
import upf.environment
import upf.walkers as walkers
from upf.simplifier import Simplifier
from upf.substituter import Substituter
from upf.fnode import FNode
from upf.expression import Expression
from upf.problem import Problem
from upf.object import Object
from upf.plan import SequentialPlan

class QuantifierSimplifier(Simplifier):
    """Same to the upf.Simplifier, but does not expand quantifiers and solves them locally."""
    def __init__(self, env: 'upf.environment.Environment', simplifier: Simplifier = None, substituter: Substituter = None):
        walkers.DagWalker.__init__(self, True)
        self._env = env
        self.manager = env.expression_manager
        if simplifier is None:
            self._simplifier = Simplifier(self._env)
        else:
            self._simplifier = simplifier
        if substituter is None:
            self._substituer = Substituter(self._env)
        else:
            self._substituter = substituter
        self._problem: Optional[Problem] = None

    def set_problem(self, problem: Problem):
        self._problem = problem

    def qsimplify(self, expression: FNode, assignments: Dict[Expression, Expression]):
        assert self._problem is not None
        self._assignments = assignments.copy() #make a copy so it can be modified
        return self.walk(expression)

    def _push_with_children_to_stack(self, expression: FNode, **kwargs):
        """Add children to the stack."""
        self.stack.append((True, expression))
        if not (expression.is_forall() or expression.is_exists()):
            for s in self._get_children(expression):
                # Add only if not memoized already
                key = self._get_key(s, **kwargs)
                if key not in self.memoization:
                    self.stack.append((False, s))

    def _compute_node_result(self, expression: FNode, **kwargs):
        """Apply function to the node and memoize the result.
        Note: This function assumes that the results for the children
              are already available.
        """
        key = self._get_key(expression, **kwargs)
        if key not in self.memoization:
            try:
                f = self.functions[expression.node_type()]
            except KeyError:
                f = self.walk_error

            if not (expression.is_forall() or expression.is_exists()):
                args = [self.memoization[self._get_key(s, **kwargs)] \
                        for s in self._get_children(expression)]
                self.memoization[key] = f(expression, args=args, **kwargs)
            else:
                self.memoization[key] = f(expression, args=expression.args(), **kwargs)
        else:
            pass

    def _deep_subs_simplify(self, expression: FNode, variables_assignments: Dict[Expression, Expression]) -> FNode:
        self._assignments.update(variables_assignments)
        old_exp = None
        new_exp = expression
        while old_exp != new_exp:
        #This do-while loop is necessary because when we have a FluentExp with
        #  some parameters, the first substitution substitutes the parameters with
        #  the object: then every ground fluent is substituted with it's value.
        #  It is a while loop because sometimes more than 2 substitutions can be
        #  required.
            old_exp = new_exp
            new_exp = self._substituter.substitute(new_exp, self._assignments)
        r = self._simplifier.simplify(new_exp)
        assert r.is_constant()
        return r

    def walk_exists(self, expression: FNode, args: List[FNode]) -> FNode:
        assert self._problem is not None
        assert len(args) == 1
        if args[0].is_bool_constant():
            if args[0].bool_constant_value():
                return self.manager.TRUE()
            return self.manager.FALSE()
        vars = expression.variables()
        type_list = [v.type() for v in vars]
        possible_objects: List[List[Object]] = [self._problem.objects(t) for t in type_list]
        #product of n iterables returns a generator of tuples where
        # every tuple has n elements and the tuples make every possible
        # combination of 1 item for each iterable. For example:
        #product([1,2], [3,4], [5,6], [7]) =
        # (1,3,5,7) (1,3,6,7) (1,4,5,7) (1,4,6,7) (2,3,5,7) (2,3,6,7) (2,4,5,7) (2,4,6,7)
        for o in product(*possible_objects):
            subs: Dict[Expression, Expression] = dict(zip(vars, list(o)))
            result = self._deep_subs_simplify(args[0], subs)
            assert result.is_bool_constant()
            if result.bool_constant_value():
                return self.manager.TRUE()
        return self.manager.FALSE()

    def walk_forall(self, expression: FNode, args: List[FNode]) -> FNode:
        assert self._problem is not None
        assert len(args) == 1
        if args[0].is_bool_constant():
            if args[0].bool_constant_value():
                return self.manager.TRUE()
            return self.manager.FALSE()
        vars = expression.variables()
        type_list = [v.type() for v in vars]
        possible_objects: List[List[Object]] = [self._problem.objects(t) for t in type_list]
        #product of n iterables returns a generator of tuples where
        # every tuple has n elements and the tuples make every possible
        # combination of 1 item for each iterable. For example:
        #product([1,2], [3,4], [5,6], [7]) =
        # (1,3,5,7) (1,3,6,7) (1,4,5,7) (1,4,6,7) (2,3,5,7) (2,3,6,7) (2,4,5,7) (2,4,6,7)
        for o in product(*possible_objects):
            subs: Dict[Expression, Expression] = dict(zip(vars, list(o)))
            result = self._deep_subs_simplify(args[0], subs)
            assert result.is_bool_constant()
            if not result.bool_constant_value():
                return self.manager.FALSE()
        return self.manager.TRUE()


class PlanValidator(object):
    """Performs plan validation."""
    def __init__(self, env: 'upf.environment.Environment'):
        self._env = env
        self.manager = env.expression_manager
        self._substituter = Substituter(self._env)
        self._qsimplifier = QuantifierSimplifier(self._env, substituter=self._substituter)
        self._last_error: Union[str, None] = None

    def is_valid_plan(self, problem: Problem, plan: SequentialPlan) -> bool:
        self._qsimplifier.set_problem(problem)
        self._last_error = None
        assignments: Dict[Expression, Expression] = problem.initial_values() # type: ignore
        count = 0 #used for better error indexing
        for ai in plan.actions():
            count = count + 1
            new_assignments: Dict[Expression, Expression] = {}
            for ap, oe in zip(ai.action().parameters(), ai.actual_parameters()):
                assignments[ap] = oe
            for p in ai.action().preconditions():
                ps = self._subs_simplify(p, assignments)
                if not (ps.is_bool_constant() and ps.bool_constant_value()):
                    self._last_error = f'Precondition {p} of {str(count)}-th action instance {str(ai)} is not satisfied.'
                    return False
            for e in ai.action().effects():
                cond = True
                if e.is_conditional():
                    ec = self._subs_simplify(e.condition(), assignments)
                    assert ec.is_bool_constant()
                    cond = ec.bool_constant_value()
                if cond:
                    ge = self._get_ground_fluent(e.fluent(), assignments)
                    if e.is_assignment():
                        new_assignments[ge] = self._subs_simplify(e.value(), assignments)
                    elif e.is_increase():
                        new_assignments[ge] = self._subs_simplify(self.manager.Plus(e.fluent(),
                                                e.value()), assignments)
                    elif e.is_decrease():
                        new_assignments[ge] = self._subs_simplify(self.manager.Minus(e.fluent(),
                                                e.value()), assignments)
            assignments.update(new_assignments)
            for ap in ai.action().parameters():
                del assignments[ap]
        for g in problem.goals():
            gs = self._subs_simplify(g, assignments)
            if not (gs.is_bool_constant() and gs.bool_constant_value()):
                    self._last_error = f'Goal {str(g)} is not reached by the plan.'
                    return False
        return True

    def get_last_error_info(self):
        assert not self._last_error is None
        return self._last_error

    def _get_ground_fluent(self, fluent:FNode, assignments: Dict[Expression, Expression]) -> FNode:
        assert fluent.is_fluent_exp()
        new_args = []
        for p in fluent.args():
            new_args.append(self._subs_simplify(p, assignments))
        return self.manager.FluentExp(fluent.fluent(), tuple(new_args))

    def _subs_simplify(self, expression: FNode, assignments: Dict[Expression, Expression]) -> FNode:
        old_exp = None
        new_exp = expression
        while old_exp != new_exp:
        #This do-while loop is necessary because when we have a FluentExp with
        #  some parameters, the first substitution substitutes the parameters with
        #  the object: then every ground fluent is substituted with it's value.
        #  It is a while loop because sometimes more than 2 substitutions can be
        #  required.
            old_exp = new_exp
            new_exp = self._substituter.substitute(new_exp, assignments)
        r = self._qsimplifier.qsimplify(new_exp, assignments)
        assert r.is_constant()
        return r
