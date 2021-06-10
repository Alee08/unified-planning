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
"""This module defines all the types."""

from fractions import Fraction
from typing import Optional, Dict, Tuple


class Type:
    """Basic class for representing a type."""

    def is_bool_type(self) -> bool:
        """Returns true iff is boolean type."""
        return False

    def is_user_type(self) -> bool:
        """Returns true iff is a user type."""
        return False

    def is_real_type(self) -> bool:
        """Returns true iff is real type."""
        return False

    def is_int_type(self) -> bool:
        """Returns true iff is integer type."""
        return False


class _BoolType(Type):
    """Represents the boolean type."""

    def is_bool_type(self) -> bool:
        """Returns true iff is boolean type."""
        return True


class _UserType(Type):
    """Represents the user type."""
    def __init__(self, name: str):
        Type.__init__(self)
        self._name = name

    def name(self) -> str:
        """Returns the type name."""
        return self._name

    def is_user_type(self) -> bool:
        """Returns true iff is a user type."""
        return True


class _IntType(Type):
    def __init__(self, lower_bound: int = None, upper_bound: int = None):
        Type.__init__(self)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def lower_bound(self) -> Optional[int]:
        return self._lower_bound

    def upper_bound(self) -> Optional[int]:
        return self._upper_bound

    def is_int_type(self) -> bool:
        return True


class _RealType(Type):
    def __init__(self, lower_bound: Fraction = None, upper_bound: Fraction = None):
        Type.__init__(self)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def lower_bound(self) -> Optional[Fraction]:
        return self._lower_bound

    def upper_bound(self) -> Optional[Fraction]:
        return self._upper_bound

    def is_real_type(self) -> bool:
        return True


BOOL = _BoolType()

class TypeManager:
    def __init__(self):
        self._bool = BOOL
        self._ints: Dict[Tuple[Optional[int], Optional[int]], Type] = {}
        self._reals: Dict[Tuple[Optional[Fraction], Optional[Fraction]], Type] = {}
        self._user_types: Dict[str, Type] = {}

    def BoolType(self) -> Type:
        return self._bool

    def IntType(self, lower_bound: int = None, upper_bound: int = None) -> Type:
        k = (lower_bound, upper_bound)
        if k in self._ints:
            return self._ints[k]
        else:
            it = _IntType(lower_bound, upper_bound)
            self._ints[k] = it
            return it

    def RealType(self, lower_bound: Fraction = None, upper_bound: Fraction = None) -> Type:
        k = (lower_bound, upper_bound)
        if k in self._reals:
            return self._reals[k]
        else:
            rt = _RealType(lower_bound, upper_bound)
            self._reals[k] = rt
            return rt

    def UserType(self, name: str) -> Type:
        if name in self._user_types:
            return self._user_types[name]
        else:
            ut = _UserType(name)
            self._user_types[name] = ut
            return ut
