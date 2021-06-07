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

import upf.typing
import upf.environment
import upf.walkers as walkers
import upf.operators as op
from upf.typing import BOOL
from upf.fnode import FNode
from upf.exceptions import UPFTypeError
from typing import List, Optional


class TypeChecker(walkers.DagWalker):
    def __init__(self, env: 'upf.environment.Environment'):
        walkers.DagWalker.__init__(self)
        self.env = env

    def get_type(self, expression: FNode) -> upf.typing.Type:
        """ Returns the pysmt.types type of the expression """
        res = self.walk(expression)
        if res is None:
            raise UPFTypeError("The expression '%s' is not well-formed" \
                               % str(expression))
        return res

    @walkers.handles(op.AND, op.OR, op.NOT, op.IMPLIES, op.IFF)
    def walk_bool_to_bool(self, expression: FNode,
                          args: List[upf.typing.Type]) -> Optional[upf.typing.Type]:
        assert expression is not None
        for x in args:
            if x is None or x != BOOL:
                return None
        return BOOL

    def walk_fluent_exp(self, expression: FNode, args: List[upf.typing.Type]) -> Optional[upf.typing.Type]:
        assert expression.is_fluent_exp()
        f = expression.fluent()
        if len(args) != len(f.signature()):
            return None
        for (arg, p_type) in zip(args, f.signature()):
            if arg != p_type:
                return None
        return f.type()

    def walk_param_exp(self, expression: FNode, args: List[upf.typing.Type]) -> upf.typing.Type:
        assert expression is not None
        assert len(args) == 0
        return expression.parameter().type()

    def walk_object_exp(self, expression: FNode, args: List[upf.typing.Type]) -> upf.typing.Type:
        assert expression is not None
        assert len(args) == 0
        return expression.object().type()

    @walkers.handles(op.BOOL_CONSTANT)
    def walk_identity_bool(self, expression: FNode,
                           args: List[upf.typing.Type]) -> Optional[upf.typing.Type]:
        assert expression is not None
        assert len(args) == 0
        return BOOL

    @walkers.handles(op.REAL_CONSTANT)
    def walk_identity_real(self, expression, args):
        assert expression is not None
        assert len(args) == 0
        return self.env.type_manager().RealType(expression.value(), expression.value())

    @walkers.handles(op.INT_CONSTANT)
    def walk_identity_int(self, expression, args):
        assert expression is not None
        assert len(args) == 0
        return self.env.type_manager().IntType(expression.value(), expression.value())

    @walkers.handles(op.PLUS, op.MINUS, op.TIMES, op.DIV)
    def walk_realint_to_realint(self, expression, args):
        has_real = False
        for x in args:
            if x is None or not (x.is_int_type() or x.is_real_type()):
                return None
            if x.is_real_type():
                has_real = True
        if has_real:
            return self.env.type_manager().RealType()
        else:
            return self.env.type_manager().IntType()

    @walkers.handles(op.LE, op.LT)
    def walk_math_relation(self, expression, args):
        for x in args:
            if x is None or not (x.is_int_type() or x.is_real_type()):
                return None
        return BOOL

    def walk_equals(self, expression: FNode,
                    args: List[upf.typing.Type]) -> Optional[upf.typing.Type]:
        t = args[0]
        if t is None:
            return None

        if t.is_bool_type():
            raise UPFTypeError("The expression '%s' is not well-formed."
                               "Equality operator is not supported for Boolean"
                               " terms. Use Iff instead." \
                               % str(expression))
        for x in args:
            if x is None:
                return None
            elif t.is_user_type() and t != x:
                return None
            elif (t.is_int_type() or t.is_real_type()) and not (x.is_int_type() or x.is_real_type()):
                return None
        return BOOL
