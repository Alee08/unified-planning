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

import upf.environment
import upf.walkers as walkers
import upf.operators as op
from upf.fnode import FNode
from typing import List, Set


class Simplifier(walkers.DagWalker):
    """Performs basic simplifications of the input expression."""

    def __init__(self, env: 'upf.environment.Environment'):
        walkers.DagWalker.__init__(self)
        self.env = env
        self.manager = env.expression_manager

    def simplify(self, expression: FNode) -> FNode:
        """Performs basic simplification of the given expression."""
        return self.walk(expression)

    def walk_and(self, expression: FNode, args: List[FNode]) -> FNode:
        if len(args) == 2 and args[0] == args[1]:
            return args[0]

        new_args: Set[FNode] = set()
        for a in args:
            if a.is_true():
                continue
            if a.is_false():
                return self.manager.FALSE()
            if a.is_and():
                for s in a.args():
                    if self.walk_not(self.manager.Not(s), [s]) in new_args:
                        return self.manager.FALSE()
                    new_args.add(s)
            else:
                if self.walk_not(self.manager.Not(a), [a]) in new_args:
                    return self.manager.FALSE()
                new_args.add(a)

        if len(new_args) == 0:
            return self.manager.TRUE()
        elif len(new_args) == 1:
            return next(iter(new_args))
        else:
            return self.manager.And(new_args)

    def walk_or(self, expression: FNode, args: List[FNode]) -> FNode:
        if len(args) == 2 and args[0] == args[1]:
            return args[0]

        new_args: Set[FNode] = set()
        for a in args:
            if a.is_false():
                continue
            if a.is_true():
                return self.manager.TRUE()
            if a.is_or():
                for s in a.args():
                    if self.walk_not(self.manager.Not(s), [s]) in new_args:
                        return self.manager.TRUE()
                    new_args.add(s)
            else:
                if self.walk_not(self.manager.Not(a), [a]) in new_args:
                    return self.manager.TRUE()
                new_args.add(a)

        if len(new_args) == 0:
            return self.manager.FALSE()
        elif len(new_args) == 1:
            return next(iter(new_args))
        else:
            return self.manager.Or(new_args)

    def walk_not(self, expression: FNode, args: List[FNode]) -> FNode:
        assert len(args) == 1
        child = args[0]
        if child.is_bool_constant():
            l = child.constant_value()
            return self.manager.Bool(not l)
        elif child.is_not():
            return child.arg(0)

        return self.manager.Not(child)

    def walk_iff(self, expression: FNode, args: List[FNode]) -> FNode:
        assert len(args) == 2

        sl = args[0]
        sr = args[1]

        if sl.is_bool_constant() and sr.is_bool_constant():
            l = sl.constant_value()
            r = sr.constant_value()
            return self.manager.Bool(l == r)
        elif sl.is_bool_constant():
            if sl.constant_value():
                return sr
            else:
                return self.manager.Not(sr)
        elif sr.is_bool_constant():
            if sr.constant_value():
                return sl
            else:
                return self.manager.Not(sl)
        elif sl == sr:
            return self.manager.TRUE()
        else:
            return self.manager.Iff(sl, sr)

    def walk_implies(self, expression: FNode, args: List[FNode]) -> FNode:
        assert len(args) == 2

        sl = args[0]
        sr = args[1]

        if sl.is_bool_constant():
            l = sl.constant_value()
            if l:
                return sr
            else:
                return self.manager.TRUE()
        elif sr.is_bool_constant():
            r = sr.constant_value()
            if r:
                return self.manager.TRUE()
            else:
                return self.manager.Not(sl)
        elif sl == sr:
            return self.manager.TRUE()
        else:
            return self.manager.Implies(sl, sr)

    def walk_equals(self, expression: FNode, args: List[FNode]) -> FNode:
        assert len(args) == 2

        sl = args[0]
        sr = args[1]

        if sl.is_constant() and sr.is_constant():
            l = sl.constant_value()
            r = sr.constant_value()
            return self.manager.Bool(l == r)
        elif sl == sr:
            return self.manager.TRUE()
        else:
            return self.manager.Equals(sl, sr)

    def walk_le(self, expression: FNode, args: List[FNode]) -> FNode:
        assert len(args) == 2

        sl = args[0]
        sr = args[1]

        if sl.is_constant() and sr.is_constant():
            l = sl.constant_value()
            r = sr.constant_value()
            return self.manager.Bool(l <= r)
        return  self.manager.LE(sl, sr)

    def walk_lt(self, expression: FNode, args: List[FNode]) -> FNode:
        assert len(args) == 2

        sl = args[0]
        sr = args[1]

        if sl.is_constant() and sr.is_constant():
            l = sl.constant_value()
            r = sr.constant_value()
            return self.manager.Bool(l < r)
        return self.manager.LT(sl, sr)

    def walk_fluent_exp(self, expression: FNode, args: List[FNode]) -> FNode:
        return self.manager.FluentExp(expression.fluent(), tuple(args))

    @walkers.handles(op.IRA_OPERATORS)
    @walkers.handles(op.CONSTANTS)
    @walkers.handles(op.PARAM_EXP, op.OBJECT_EXP)
    def walk_identity(self, expression: FNode, args: List[FNode]) -> FNode:
        return expression
