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


from typing import cast

import unified_planning as up
import unified_planning.environment
import unified_planning.engines as engines
import unified_planning.engines.mixins as mixins
import unified_planning.walkers as walkers
from unified_planning.model import AbstractProblem, Problem, ProblemKind
from unified_planning.engines.results import ValidationResult, ValidationResultStatus, LogMessage, LogLevel
from unified_planning.engines.sequential_simulator import SequentialSimulator, InstantaneousEvent
from unified_planning.plans import SequentialPlan, PlanKind


class SequentialPlanValidator(engines.engine.Engine, mixins.PlanValidatorMixin):
    """Performs plan validation."""
    def __init__(self, **options):
        self._env: 'unified_planning.environment.Environment' = unified_planning.environment.get_env(options.get('env', None))
        self.manager = self._env.expression_manager
        self._substituter = walkers.Substituter(self._env)

    @property
    def name(self):
        return 'sequential_plan_validator'

    @staticmethod
    def supports_plan(plan_kind: 'up.plans.PlanKind') -> bool:
        return plan_kind == PlanKind.SEQUENTIAL_PLAN

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind.set_problem_class('ACTION_BASED') # type: ignore
        supported_kind.set_typing('FLAT_TYPING') # type:ignore
        supported_kind.set_typing('HIERARCHICAL_TYPING') # type:ignore
        supported_kind.set_numbers('CONTINUOUS_NUMBERS') # type:ignore
        supported_kind.set_numbers('DISCRETE_NUMBERS') # type:ignore
        supported_kind.set_conditions_kind('NEGATIVE_CONDITIONS') # type:ignore
        supported_kind.set_conditions_kind('DISJUNCTIVE_CONDITIONS') # type:ignore
        supported_kind.set_conditions_kind('EQUALITY') # type:ignore
        supported_kind.set_conditions_kind('EXISTENTIAL_CONDITIONS') # type:ignore
        supported_kind.set_conditions_kind('UNIVERSAL_CONDITIONS') # type:ignore
        supported_kind.set_effects_kind('CONDITIONAL_EFFECTS') # type:ignore
        supported_kind.set_effects_kind('INCREASE_EFFECTS') # type:ignore
        supported_kind.set_effects_kind('DECREASE_EFFECTS') # type:ignore
        supported_kind.set_fluents_type('NUMERIC_FLUENTS') # type:ignore
        supported_kind.set_fluents_type('OBJECT_FLUENTS') # type:ignore
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= SequentialPlanValidator.supported_kind()

    def _validate(self, problem: 'AbstractProblem',
                  plan: 'unified_planning.plans.Plan') -> 'up.engines.results.ValidationResult':
        """Returns True if and only if the plan given in input is a valid plan for the problem given in input.
        This means that from the initial state of the problem, by following the plan, you can reach the
        problem goal. Otherwise False is returned."""
        assert isinstance(plan, SequentialPlan)
        assert isinstance(problem, Problem)
        simulator = SequentialSimulator(problem)
        current_state = up.model.UPRWState(problem.initial_values)
        count = 0 #used for better error indexing
        for ai in plan.actions:
            action = ai.action
            assert isinstance(action, unified_planning.model.InstantaneousAction)
            count = count + 1
            events = simulator.get_events(action, ai.actual_parameters)
            assert len(events) < 2, f'{str(ai)} + {len(events)}'
            for event in events:
                if not simulator.is_applicable(event, current_state):
                    error = f'Preconditions {event.conditions} of {str(count)}-th action instance {str(ai)} are not satisfied.'
                    logs = [LogMessage(LogLevel.ERROR, error)]
                    return ValidationResult(ValidationResultStatus.INVALID, self.name, logs)
                current_state = cast(up.model.UPRWState, simulator.apply_unsafe(event, current_state))
        for g in problem.goals:
            wrapper_event = InstantaneousEvent([g], [])
            if not simulator.is_applicable(wrapper_event, current_state):
                error = f'Goal {str(g)} is not reached by the plan.'
                logs = [LogMessage(LogLevel.ERROR, error)]
                return ValidationResult(ValidationResultStatus.INVALID, self.name, logs)
        return ValidationResult(ValidationResultStatus.VALID, self.name, [])
