# Copyright 2022 AIPlan4EU project
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
This module defines the Method class.
A Method has a name, a list of Parameters, a list of conditions
and a list of subtasks.
"""
from collections import OrderedDict
from typing import List, Union, Optional

import unified_planning as up
from unified_planning.environment import get_env, Environment
from unified_planning.exceptions import UPUnboundedVariablesError, UPValueError
from unified_planning.model import Timing, Parameter
from unified_planning.model.action import Action
from unified_planning.model.htn.task import Task, Subtask
from unified_planning.model.timing import Timepoint


class ParameterizedTask:
    """A task instantiated with some parameters."""
    def __init__(self, task: Task, *params: up.model.parameter.Parameter):
        self._task = task
        self._params: List[up.model.parameter.Parameter] = list(params)
        assert len(self._task.parameters) == len(self._params)

    def __repr__(self):
        return str(self._task.name) + '(' + ', '.join(map(str, self.parameters)) + ')'

    def __eq__(self, other):
        return isinstance(other, ParameterizedTask) and self._task == other._task and self._params == other._params

    def __hash__(self):
        return hash(self._task) + sum(map(hash, self._params))

    @property
    def task(self) -> Task:
        return self._task

    @property
    def parameters(self) -> List[up.model.parameter.Parameter]:
        return self._params


class Method:
    """This is the method interface."""
    def __init__(self, _name: str, _parameters: 'OrderedDict[str, up.model.types.Type]' = None,
                 _env: Environment = None, **kwargs: 'up.model.types.Type'):
        self._env = get_env(_env)
        self._task: Optional[ParameterizedTask] = None
        self._name = _name
        self._parameters: 'OrderedDict[str, up.model.parameter.Parameter]' = OrderedDict()
        self._preconditions: List[up.model.fnode.FNode] = []
        self._constraints: List[up.model.fnode.FNode] = []
        self._subtasks: List[Subtask] = []
        if _parameters is not None:
            assert len(kwargs) == 0
            for n, t in _parameters.items():
                self._parameters[n] = up.model.parameter.Parameter(n, t)
        else:
            for n, t in kwargs.items():
                self._parameters[n] = up.model.parameter.Parameter(n, t)

    def __repr__(self) -> str:
        s = []
        s.append(f'method {self.name}')
        first = True
        for p in self.parameters:
            if first:
                s.append('(')
                first = False
            else:
                s.append(', ')
            s.append(str(p))
        if not first:
            s.append(')')
        s.append(' {\n')
        s.append(f'  task = {self._task}\n')
        s.append('  preconditions = [\n')
        for c in self.preconditions:
            s.append(f'    {str(c)}\n')
        s.append('  ]\n')
        s.append('  constraints = [\n')
        for c in self.constraints:
            s.append(f'    {str(c)}\n')
        s.append('  ]\n')
        s.append('  subtasks = [\n')
        for st in self.subtasks:
            s.append(f'      {str(st)}\n')
        s.append('  ]\n')
        s.append('}')
        return ''.join(s)

    def __eq__(self, oth: object) -> bool:
        if not isinstance(oth, Method):
            return False
        return (self._env == oth._env and
                self._name == oth._name and
                self._parameters == oth._parameters and
                self._task == oth._task and
                set(self._preconditions) == set(oth._preconditions) and
                set(self._subtasks) == set(oth._subtasks) and
                set(self._constraints) == set(oth._constraints))

    def __hash__(self) -> int:
        res = hash(self._name)
        res += hash(self._task)
        res += sum(map(hash, self.parameters))
        res += sum(map(hash, self._preconditions))
        res += sum(map(hash, self._constraints))
        res += sum(map(hash, self.subtasks))
        return res

    @property
    def name(self) -> str:
        """Returns the action name."""
        return self._name

    @property
    def achieved_task(self) -> ParameterizedTask:
        """Returns the task that this method achieves."""
        assert self._task is not None
        return self._task

    def set_task(self, task: Union[Task, ParameterizedTask], *arguments: Parameter):
        """Defines the task that is method achieves."""
        assert self._task is None, f"Method {self.name} was already assigned a task"
        if isinstance(task, ParameterizedTask):
            assert all(p.name in self._parameters for p in task.parameters)
            self._task = task
        elif isinstance(task, Task) and len(arguments) == 0:
            for task_param in task.parameters:
                assert task_param.name in self._parameters, f"Missing task parameter '{task_param.name}' in method {self._name}. Please pass all parameters explicitly."
            self._task = ParameterizedTask(task, *task.parameters)
        else:
            assert all(p.name in self._parameters for p in arguments)
            self._task = ParameterizedTask(task, *arguments)

    @property
    def parameters(self) -> List['up.model.parameter.Parameter']:
        """Returns the list of the method's parameters."""
        return list(self._parameters.values())

    def parameter(self, name: str) -> 'up.model.parameter.Parameter':
        """Returns the parameter of the action with the given name."""
        for param in self.parameters:
            if param.name == name:
                return param
        raise UPValueError(f'Unknown parameter name: {name}')

    @property
    def preconditions(self) -> List['up.model.fnode.FNode']:
        """Returns the list of the method's preconditions."""
        return self._preconditions

    def add_precondition(self, precondition: Union['up.model.fnode.FNode', 'up.model.fluent.Fluent', 'up.model.parameter.Parameter', bool]):
        """Adds the given method precondition."""
        precondition_exp, = self._env.expression_manager.auto_promote(precondition)
        assert self._env.type_checker.get_type(precondition_exp).is_bool_type()
        if precondition_exp == self._env.expression_manager.TRUE():
            return
        free_vars = self._env.free_vars_oracle.get_free_variables(precondition_exp)
        if len(free_vars) != 0:
            raise UPUnboundedVariablesError(f"The precondition {str(precondition_exp)} has unbounded variables:\n{str(free_vars)}")
        if precondition_exp not in self._preconditions:
            self._preconditions.append(precondition_exp)

    @property
    def constraints(self) -> List['up.model.fnode.FNode']:
        """Returns the list of the method's constraints."""
        return self._constraints

    def add_constraint(self, constraint: Union['up.model.fnode.FNode', 'up.model.fluent.Fluent', 'up.model.parameter.Parameter', bool]):
        """Adds the given constraint to the method."""
        constraint_exp, = self._env.expression_manager.auto_promote(constraint)
        assert self._env.type_checker.get_type(constraint_exp).is_bool_type()
        if constraint_exp == self._env.expression_manager.TRUE():
            return
        free_vars = self._env.free_vars_oracle.get_free_variables(constraint_exp)
        if len(free_vars) != 0:
            raise UPUnboundedVariablesError(f"The constraint {str(constraint_exp)} has unbounded variables:\n{str(free_vars)}")
        if constraint_exp not in self._constraints:
            self._constraints.append(constraint_exp)

    @property
    def subtasks(self) -> List['Subtask']:
        """Returns the list of the method's subtasks."""
        return self._subtasks

    def add_subtask(self, subtask: Union[Subtask, Action, Task], *args, ident: Optional[str] = None) -> Subtask:
        """Add a subtask to this method, in no particular order."""
        if not isinstance(subtask, Subtask):
            subtask = Subtask(subtask, *args, ident=ident)
        assert isinstance(subtask, Subtask)
        assert all([subtask.identifier != prev.identifier for prev in self.subtasks])
        self._subtasks.append(subtask)
        return subtask

    def set_ordered(self, *subtasks: Subtask):
        """Imposes a sequentioal order between the given subtasks."""
        if len(subtasks) < 2:
            return
        prev = subtasks[0]
        for i in range(1, len(subtasks)):
            next = subtasks[i]
            self.set_strictly_before(prev, next)
            prev = next

    def set_strictly_before(self, lhs: Union[Subtask, Timepoint, Timing], rhs: Union[Subtask, Timepoint, Timing]):
        if isinstance(lhs, Subtask):
            lhs = lhs.end
        if isinstance(lhs, Timepoint):
            lhs = Timing(timepoint=lhs, delay=0)
        assert isinstance(lhs, Timing)
        if isinstance(rhs, Subtask):
            rhs = rhs.start
        if isinstance(rhs, Timepoint):
            rhs = Timing(timepoint=rhs, delay=0)
        assert isinstance(rhs, Timing)
        self.add_constraint(self._env.expression_manager.LT(lhs, rhs))



