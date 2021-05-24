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
"""This module defines the fluent class. """

from upf.environment import get_env


class Fluent:
    """Represents a fluent."""
    def __init__(self, name, typename=None, signature=[], env=None):
        self._env = get_env(env)
        self._name = name
        if typename is None:
            self._typename = self._env.type_manager.BOOL()
        else:
            self._typename = typename
        self._signature = signature

    def name(self):
        """Returns the fluent name."""
        return self._name

    def type(self):
        """Returns the fluent type."""
        return self._typename

    def signature(self):
        """Returns the fluent signature.
        The signature is the list of types of the fluent parameters.
        """
        return self._signature

    def arity(self):
        """Returns the fluent arity."""
        return len(self._signature)

    def __call__(self, *args):
        """Returns a fluent expression with the given parameters."""
        return self._env.expression_manager.FluentExp(self, args)
