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
"""This module defines the solver interface."""

import upf
import upf.model
from upf.plan import Plan
from upf.model import ProblemKind, Problem, Action, FNode
from typing import Optional, Tuple, Dict, List, Callable


class Solver:
    """Represents the solver interface."""

    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            raise

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    def is_oneshot_planner() -> bool:
        return False

    @staticmethod
    def is_plan_validator() -> bool:
        return False

    @staticmethod
    def is_grounder() -> bool:
        return False

    @staticmethod
    def supports(problem_kind: 'ProblemKind') -> bool:
        return len(problem_kind.features()) == 0

    def solve(self, problem: 'upf.model.Problem') -> Optional['upf.plan.Plan']:
        raise NotImplementedError

    def validate(self, problem: 'upf.model.Problem', plan: 'upf.plan.Plan') -> bool:
        raise NotImplementedError

    def ground(self, problem: 'upf.model.Problem') -> Tuple[Problem, Callable[[Plan], Plan]]:
        '''This function should return the tuple (grounded_problem, trace_back_map), where
        "trace_back_map" is a map from every action in the "grounded_problem" to the tuple
        (original_action, parameters). Where the grounded actions is obtained by grounding
        the "original_action" with the specific "parameters". '''
        raise NotImplementedError

    def destroy(self):
        raise NotImplementedError

    def __enter__(self):
        """Manages entering a Context (i.e., with statement)"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Manages exiting from Context (i.e., with statement)"""
        self.destroy()
