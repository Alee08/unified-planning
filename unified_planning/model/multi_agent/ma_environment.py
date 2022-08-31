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

import unified_planning as up
from unified_planning.model.mixins import (
    FluentsSetMixin,
)


class MAEnvironment(
    FluentsSetMixin,
):
    """Represents a MAEnvironment."""

    def __init__(
        self,
        ma_problem: "up.model.multi_agent.ma_problem.MultiAgentProblem",
    ):
        FluentsSetMixin.__init__(
            self,
            ma_problem.env,
            ma_problem._add_user_type,
            self.has_name,
            ma_problem._initial_defaults,
        )
        self._env = ma_problem.env

    @property
    def env(self) -> "up.Environment":
        """Returns the Agent environment."""
        return self._env

    def has_name(self, name: str) -> bool:
        """Returns true if the name is in the problem."""
        return self.has_fluent(name)

    def __repr__(self) -> str:
        s = []
        s.append("fluents = [\n")
        for f in self._fluents:
            s.append(f" {str(f)}\n")
        s.append("]\n\n")
        return "".join(s)

    def __eq__(self, oth: object) -> bool:
        if not (isinstance(oth, MAEnvironment)) or self._env != oth._env:
            return False
        if set(self._fluents) != set(oth._fluents):
            return False
        return True

    def __hash__(self) -> int:
        res = 0
        for f in self._fluents:
            res += hash(f)
        return res
