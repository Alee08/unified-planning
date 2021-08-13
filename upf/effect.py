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
"""
This module defines the Effect class.
A basic Effect has a fluent and an expression.
A condition can be added to make it a conditional effect.
"""


from upf.fnode import FNode
from collections import OrderedDict
from typing import List, Union, Tuple, Optional

KINDS = list(range(0, 3))

(
    ASSIGN, INCREASE, DECREASE
) = KINDS

class Effect:
    def __init__(self, fluent: FNode, value: FNode, condition: FNode, kind: int = ASSIGN):
        self._fluent = fluent
        self._value = value
        self._condition = condition
        self._kind = kind

    def __repr__(self) -> str:
        s = []
        if self.is_conditional():
            s.append(f'if {str(self._condition)} then')
        s.append(f'{str(self._fluent)}')
        if self.is_assignment():
            s.append(':=')
        elif self.is_increase:
            s.append('+=')
        elif self.is_decrease:
            s.append('-=')
        s.append(f'{str(self._value)}')
        return ' '.join(s)

    def is_conditional(self) -> bool:
        return not self._condition.is_true()

    def fluent(self) -> FNode:
        return self._fluent

    def value(self) -> FNode:
        return self._value

    def condition(self) -> FNode:
        assert self.is_conditional()
        return self._condition

    def kind(self) -> int:
        return self._kind

    def is_assignment(self) -> bool:
        return self._kind == ASSIGN

    def is_increase(self) -> bool:
        return self._kind == INCREASE

    def is_decrease(self) -> bool:
        return self._kind == DECREASE
