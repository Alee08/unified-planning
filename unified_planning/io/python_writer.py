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
#​

from fractions import Fraction
import sys
import unified_planning as up
import unified_planning.walkers as walkers
from unified_planning.model.types import _UserType, _IntType, _RealType
from typing import IO, Any, Optional, cast
from io import StringIO


class ConverterToPythonString(walkers.DagWalker):
    '''Expression converter to a Python string.'''

    def __init__(self, env: 'up.environment.Environment'):
        walkers.DagWalker.__init__(self)

    def convert(self, expression):
        '''Converts the given expression to a PDDL string.'''
        if expression is None:
            return 'None'
        return self.walk(expression)

    def walk_exists(self, expression, args):
        assert len(args) == 1
        vars_string_list = [f'up.model.Variable("{v.name()}", {_print_python_type(v.type())})' for v in expression.variables()]
        return f'emgr.Exists({args[0]}, {", ".join(vars_string_list)})'

    def walk_forall(self, expression, args):
        assert len(args) == 1
        vars_string_list = [f'up.model.Variable("{v.name()}", {_print_python_type(v.type())})' for v in expression.variables()]
        return f'emgr.Forall({args[0]}, {", ".join(vars_string_list)})'

    def walk_variable_exp(self, expression, args):
        assert len(args) == 0
        v = expression.variable()
        return f'up.model.Variable("{v.name()}", {_print_python_type(v.type())})'

    def walk_and(self, expression, args):
        assert len(args) > 1
        return f'emgr.And({", ".join(args)})'

    def walk_or(self, expression, args):
        assert len(args) > 1
        return f'emgr.Or({", ".join(args)})'

    def walk_not(self, expression, args):
        assert len(args) == 1
        return f'emgr.Not({args[0]})'

    def walk_implies(self, expression, args):
        assert len(args) == 2
        return f'emgr.Implies({args[0]}, {args[1]})'

    def walk_iff(self, expression, args):
        assert len(args) == 2
        return f'emgr.Iff({args[0]}, {args[1]})'

    def walk_fluent_exp(self, expression, args):
        fluent = expression.fluent()
        if args:
            return f'fluent_{fluent.name()}({", ".join(args)})'
        return f'fluent_{fluent.name()}'

    def walk_param_exp(self, expression, args):
        assert len(args) == 0
        p = expression.parameter()
        return f'parameter_{p.name()}'

    def walk_object_exp(self, expression, args):
        assert len(args) == 0
        o = expression.object()
        return f'object_{o.name()}'

    def walk_bool_constant(self, expression, args):
        assert len(args) == 0
        if expression.bool_constant_value():
            return 'emgr.TRUE()'
        else:
            return 'emgr.FALSE()'

    def walk_real_constant(self, expression, args):
        assert len(args) == 0
        return f'emgr.Real(Fraction({str(expression.real_constant_value().numerator)}, {str(expression.real_constant_value().denominator)}))'

    def walk_int_constant(self, expression, args):
        assert len(args) == 0
        return f'emgr.Int({str(expression.constant_value())})'

    def walk_plus(self, expression, args):
        assert len(args) > 1
        return f'emgr.Plus({", ".join(args)})'

    def walk_minus(self, expression, args):
        assert len(args) == 2
        return f'emgr.Minus({args[0]}, {args[1]})'

    def walk_times(self, expression, args):
        assert len(args) > 1
        return f'emgr.Times({", ".join(args)})'

    def walk_div(self, expression, args):
        assert len(args) == 2
        return f'emgr.Div({args[0]}, {args[1]})'

    def walk_le(self, expression, args):
        assert len(args) == 2
        return f'emgr.LE({args[0]}, {args[1]})'

    def walk_lt(self, expression, args):
        assert len(args) == 2
        return f'emgr.LT({args[0]}, {args[1]})'

    def walk_equals(self, expression, args):
        assert len(args) == 2
        return f'emgr.Equals({args[0]}, {args[1]})'


class PythonWriter:
    '''This class can be used to write a Problem in PDDL.'''

    def __init__(self, problem: 'up.model.Problem'):
        self.problem = problem

    def _write_problem_code(self, out: IO[str]):

        converter = ConverterToPythonString(self.problem.env)
        out.write('from fractions import Fraction\n')
        out.write('import unified_planning as up\n')
        out.write('env = up.environment.get_env()\n')
        out.write('emgr = env.expression_manager\n')
        out.write('tm = env.type_manager\n')

        for type in self.problem.user_types(): # define user_types
            if cast(_UserType, type).father() is None:
                out.write(f'type_{cast(_UserType, type).name()} = tm.UserType("{cast(_UserType, type).name()}")\n')
            else:
                out.write(f'type_{cast(_UserType, type).name()} = tm.UserType("{cast(_UserType, type).name()}", type_{cast(_UserType, cast(_UserType, type).father()).name()})\n') 

        for f in self.problem.fluents(): # define fluents
            params = ', '.join(_print_python_type(p) for p in f.signature())
            out.write(f'fluent_{f.name()} = up.model.Fluent("{f.name()}", {_print_python_type(f.type())}, [{params}])\n')

        for o in self.problem.all_objects(): # define objects
            out.write(f'object_{o.name()} = up.model.Object("{o.name()}", type_{cast(_UserType, o.type()).name()})\n') #NOTE works if objects are only of user_type.
        
        out.write('problem_initial_defaults = {}\n') # define initial_defaults
        for type, exp in self.problem.initial_defaults().items():
            out.write(f'problem_initial_defaults[{_print_python_type(type)}] = {converter.convert(exp)}')
        out.write(f'problem = up.model.Problem("{self.problem.name}", env, initial_defaults=problem_initial_defaults)\n')

        for o in self.problem.all_objects(): # add objects to the problem
            out.write(f'problem.add_object(object_{o.name()})\n')

        for f in self.problem.fluents(): # add fluents to the problem, with their fluents_default, if they have one
            default = self.problem.fluents_defaults().get(f, None)
            out.write(f'problem.add_fluent(fluent_{f.name()}')
            if default is not None: # the fluent has a default value
                out.write(f', default_initial_value={converter.convert(default)}')
            out.write(')\n')

        for a in self.problem.actions(): # define actions and add them to the problem
            if isinstance(a, up.model.InstantaneousAction):
                out.write(f'act_{a.name} = up.model.InstantaneousAction("{a.name}"')
                for ap in a.parameters():
                    out.write(f', {ap.name()}={_print_python_type(ap.type())}')
                out.write(')\n')
                for ap in a.parameters():
                    out.write(f'parameter_{ap.name()} = act_{a.name}.parameter("{ap.name()}")\n')
                if a.cost() is not None:
                    out.write(f'act_{a.name}.set_cost({converter.convert(a.cost())})\n')
                for p in a.preconditions():
                    out.write(f'act_{a.name}.add_precondition({converter.convert(p)})\n')
                for e in a.effects():
                    if e.is_increase():
                        out.write(f'act_{a.name}.add_increase_effect(fluent={converter.convert(e.fluent())}, value={converter.convert(e.value())}, condition={converter.convert(e.condition())})\n')
                    elif e.is_decrease():
                        out.write(f'act_{a.name}.add_decrease_effect(fluent={converter.convert(e.fluent())}, value={converter.convert(e.value())}, condition={converter.convert(e.condition())})\n')
                    else:
                        out.write(f'act_{a.name}.add_effect(fluent={converter.convert(e.fluent())}, value={converter.convert(e.value())}, condition={converter.convert(e.condition())})\n')
            elif isinstance(a, up.model.DurativeAction):
                out.write(f'act_{a.name} = up.model.DurativeAction("{a.name}"')
                for ap in a.parameters():
                    out.write(f', {ap.name()}={_print_python_type(ap.type())}')
                out.write(')\n')
                for ap in a.parameters():
                    out.write(f'parameter_{ap.name()} = act_{a.name}.parameter("{ap.name()}")\n')
                if a.cost() is not None:
                    out.write(f'act_{a.name}.set_cost({converter.convert(a.cost())})\n')
                out.write(f'act_{a.name}.set_duration_constraint({_convert_interval_duration(a.duration(), converter)})\n')
                for t, cl in a.conditions().items():
                    for c in cl:
                        out.write(f'act_{a.name}.add_condition({_convert_timing(t)}, {converter.convert(c)})\n')
                for i, dcl in a.durative_conditions().items():
                    for dc in dcl:
                        out.write(f'act_{a.name}.add_durative_condition({_convert_interval(i)}, {converter.convert(dc)})\n')
                for t, el in a.effects().items():
                    for e in el:
                        if e.is_increase():
                            out.write(f'act_{a.name}.add_increase_effect(timing={_convert_timing(t)}, fluent={converter.convert(e.fluent())}, value={converter.convert(e.value())}, condition={converter.convert(e.condition())})\n')
                        elif e.is_decrease():
                            out.write(f'act_{a.name}.add_decrease_effect(timing={_convert_timing(t)}, fluent={converter.convert(e.fluent())}, value={converter.convert(e.value())}, condition={converter.convert(e.condition())})\n')
                        else:
                            out.write(f'act_{a.name}.add_effect(timing={_convert_timing(t)}, fluent={converter.convert(e.fluent())}, value={converter.convert(e.value())}, condition={converter.convert(e.condition())})\n')
            else:
                raise NotImplementedError
            out.write(f'problem.add_action(act_{a.name})\n')

        for f_exp, v_exp in self.problem._initial_values_structure().items(): # add only previously added initial values
            out.write(f'problem.set_initial_value({converter.convert(f_exp)}, {converter.convert(v_exp)})\n')

        for t, el in self.problem.timed_effects().items(): # add timed effects
            for e in el:
                if e.is_increase():
                    out.write(f'problem.add_increase_effect(timing={_convert_timing(t)}, fluent={converter.convert(e.fluent())}, value={converter.convert(e.value())}, condition={converter.convert(e.condition())})\n')
                elif e.is_decrease():
                    out.write(f'problem.add_decrease_effect(timing={_convert_timing(t)}, fluent={converter.convert(e.fluent())}, value={converter.convert(e.value())}, condition={converter.convert(e.condition())})\n')
                else:
                    out.write(f'problem.add_timed_effect(timing={_convert_timing(t)}, fluent={converter.convert(e.fluent())}, value={converter.convert(e.value())}, condition={converter.convert(e.condition())})\n')
        
        for t, gl in self.problem.timed_goals().items(): # add timed goals
            for g in gl:
                out.write(f'problem.add_timed_goal(timing={_convert_timing(t)}, goal={converter.convert(g)})\n')
        
        for i, gl in self.problem.maintain_goals().items(): # add maintain goals
            for g in gl:
                out.write(f'problem.add_maintain_goal(interval={_convert_interval(i)}, goal={converter.convert(g)})\n')
        
        for g in self.problem.goals(): # add goals
            out.write(f'problem.add_goal(goal={converter.convert(g)})\n')

        for qm in self.problem.quality_metrics(): # adding metrics
            out.write('problem.add_quality_metric(')
            if isinstance(qm, up.model.metrics.MinimizeActionCosts):
                out.write('up.model.metrics.MinimizeActionCosts()')
            elif isinstance(qm, up.model.metrics.MinimizeExpressionOnFinalState):
                out.write(f'up.model.metrics.MinimizeExpressionOnFinalState({converter.convert(qm.expression)})')
            elif isinstance(qm, up.model.metrics.MaximizeExpressionOnFinalState):
                out.write(f'up.model.metrics.MaximizeExpressionOnFinalState({converter.convert(qm.expression)})')
            else:
                raise NotImplementedError
            out.write(')\n')

    def print_domain(self): # TODO: find a better name
        '''Prints to std output the PDDL domain.'''
        self._write_problem_code(sys.stdout)

    def write_problem_code(self) -> str:
        '''Returns the PDDL domain.'''
        out = StringIO()
        self._write_problem_code(out)
        return out.getvalue()

    def write_domain(self, filename: str): # TODO: find a better name
        '''Dumps to file the PDDL domain.'''
        with open(filename, 'w') as f:
            self._write_problem_code(f)


def _print_python_type(type: 'up.model.types.Type') -> str:
    '''This method takes a type and returns how to use it from command line.
    For the user_types it assumes they have already been created and just refers to them.'''
    if type.is_user_type():
        return f'type_{cast(_UserType, type).name()}'
    elif type.is_bool_type():
        return 'tm.BoolType()'
    elif type.is_int_type():
        return f'tm.IntType({str(cast(_IntType, type).lower_bound())}, {str(cast(_IntType, type).upper_bound())})'
    elif type.is_real_type():
        return f'tm.RealType({str(cast(_RealType, type).lower_bound())}, {str(cast(_RealType, type).upper_bound())})'
    else:
        raise NotImplementedError

def _convert_timing(timing: up.model.Timing) -> str:
    bound: str = f'{str(timing.bound())}'
    if isinstance(timing.bound(), Fraction):
        bound = f'Fraction({str(timing.bound().numerator)}, {str(timing.bound().denomiantor)})'
    if timing.is_from_start():
        return f'up.model.StartTiming({bound})'
    else:
        return f'up.model.EndTiming({bound})'

def _convert_interval(interval: up.model.Interval) -> str:
    interval_feature: str = 'up.model.ClosedInterval'
    if interval.is_left_open() and interval.is_right_open():
        interval_feature = 'up.model.OpenInterval'
    elif interval.is_left_open() and not interval.is_right_open():
        interval_feature = 'up.model.LeftOpenInterval'
    elif not interval.is_left_open() and interval.is_right_open():
        interval_feature = 'up.model.RightOpenInterval'
    return f'{interval_feature}({_convert_timing(interval.lower())}, {_convert_timing(interval.upper())})'

def _convert_interval_duration(interval: up.model.IntervalDuration, converter: ConverterToPythonString) -> str:
    interval_feature: str = 'up.model.ClosedIntervalDuration'
    if interval.is_left_open() and interval.is_right_open():
        interval_feature = 'up.model.OpenIntervalDuration'
    elif interval.is_left_open() and not interval.is_right_open():
        interval_feature = 'up.model.LeftOpenIntervalDuration'
    elif not interval.is_left_open() and interval.is_right_open():
        interval_feature = 'up.model.RightOpenIntervalDuration'
    return f'{interval_feature}({converter.convert(interval.lower())}, {converter.convert(interval.upper())})'
