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
"""The ExpressionManager is used to create expressions.

All objects are memoized so that two syntactically equivalent expressions
are represented by the same object.
"""

import upf
import upf.operators as op
from upf.fnode import FNodeContent, FNode
from typing import Iterable


class ExpressionManager(object):
    """ExpressionManager is responsible for the creation of all expressions."""

    def __init__(self):
        self.expressions = {}
        self._next_free_id = 1

        self.true_expression = self.create_node(node_type=op.BOOL_CONSTANT,
                                                args=tuple(),
                                                payload=True)
        self.false_expression = self.create_node(node_type=op.BOOL_CONSTANT,
                                                 args=tuple(),
                                                 payload=False)
        return

    def _polymorph_args_to_tuple(self, args):
        """ Helper function to return a tuple of arguments from args.
        This function is used to allow N-ary operators to express their arguments
        both as a list of arguments or as a tuple of arguments: e.g.,
           And([a,b,c]) and And(a,b,c)
        are both valid, and they are converted into a tuple (a,b,c) """

        if len(args) == 1 and isinstance(args[0], Iterable):
            args = args[0]
        return tuple(args)

    def auto_promote(self, *args):
        tuple_args = self._polymorph_args_to_tuple(args)
        res = []
        for e in tuple_args:
            if isinstance(e, upf.Fluent):
                res.append(self.FluentExp(e))
            elif isinstance(e, upf.ActionParameter):
                res.append(self.ParameterExp(e))
            elif isinstance(e, bool):
                res.append(self.Bool(e))
            else:
                res.append(e)
        if len(res) == 1:
            return res[0]
        else:
            return res

    def create_node(self, node_type, args, payload=None):
        content = FNodeContent(node_type, args, payload)
        if content in self.expressions:
            return self.expressions[content]
        else:
            n = FNode(content, self._next_free_id)
            self._next_free_id += 1
            self.expressions[content] = n
            return n

    def And(self, *args):
        """ Returns a conjunction of terms.
        This function has polimorphic arguments:
          - And(a,b,c)
          - And([a,b,c])
        Restriction: Arguments must be boolean
        """
        new_args = self.auto_promote(args)
        tuple_args = self._polymorph_args_to_tuple(new_args)

        if len(tuple_args) == 0:
            return self.TRUE()
        elif len(tuple_args) == 1:
            return tuple_args[0]
        else:
            return self.create_node(node_type=op.AND,
                                    args=tuple_args)

    def Or(self, *args):
        """ Returns an disjunction of terms.
        This function has polimorphic n-arguments:
          - Or(a,b,c)
          - Or([a,b,c])
        Restriction: Arguments must be boolean
        """
        new_args = self.auto_promote(args)
        tuple_args = self._polymorph_args_to_tuple(new_args)

        if len(tuple_args) == 0:
            return self.FALSE()
        elif len(tuple_args) == 1:
            return tuple_args[0]
        else:
            return self.create_node(node_type=op.OR,
                                    args=tuple_args)

    def Not(self, expression):
        """ Creates an expression of the form:
                not expression
        Restriction: Expression must be of boolean type
        """
        expression = self.auto_promote(expression)
        if expression.is_not():
            return expression.arg(0)
        return self.create_node(node_type=op.NOT, args=(expression,))

    def Implies(self, left, right):
        """ Creates an expression of the form:
            left -> right
        Restriction: Left and Right must be of boolean type
        """
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=op.IMPLIES, args=(left, right))

    def Iff(self, left, right):
        """ Creates an expression of the form:
            left <-> right
        Restriction: Left and Right must be of boolean type
        """
        left, right = self.auto_promote(left, right)
        return self.create_node(node_type=op.IFF, args=(left, right))

    def TRUE(self):
        """Return the boolean constant True."""
        return self.true_expression

    def FALSE(self):
        """Return the boolean constant False."""
        return self.false_expression

    def Bool(self, value):
        if type(value) != bool:
            raise PysmtTypeError("Expecting bool, got %s" % type(value))

        if value:
            return self.true_expression
        else:
            return self.false_expression

    def FluentExp(self, fluent, params=tuple()):
        """ Creates an expression for the given fluent and parameters.
        Restriction: parameters type must be compatible with the fluent signature
        """
        assert fluent.arity() == len(params)
        params = self.auto_promote(params)
        if not isinstance(params, Iterable):
            params = (params, )
        else:
            params = tuple(params)
        return self.create_node(node_type=op.FLUENT_EXP, args=params, payload=fluent)

    def ParameterExp(self, param):
        """Returns an expression for the given action parameter"""
        param = self.auto_promote(param)
        return self.create_node(node_type=op.PARAM_EXP, args=tuple(), payload=param)


EXPR_MANAGER = ExpressionManager()


def And(*args, **kwargs):
    return EXPR_MANAGER.And(*args, **kwargs)

def Or(*args, **kwargs):
    return EXPR_MANAGER.Or(*args, **kwargs)

def Implies(*args, **kwargs):
    return EXPR_MANAGER.Implies(*args, **kwargs)

def Iff(*args, **kwargs):
    return EXPR_MANAGER.Iff(*args, **kwargs)

def TRUE(*args, **kwargs):
    return EXPR_MANAGER.TRUE(*args, **kwargs)

def FALSE(*args, **kwargs):
    return EXPR_MANAGER.FALSE(*args, **kwargs)

def Bool(*args, **kwargs):
    return EXPR_MANAGER.Bool(*args, **kwargs)

def FluentExp(*args, **kwargs):
    return EXPR_MANAGER.FluentExp(*args, **kwargs)

def ParameterExp(*args, **kwargs):
    return EXPR_MANAGER.ParameterExp(*args, **kwargs)
