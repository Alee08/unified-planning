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


from fractions import Fraction
import sys
import re

from decimal import Decimal, localcontext
from warnings import warn

import unified_planning as up
import unified_planning.environment
import unified_planning.model.walkers as walkers
from unified_planning.model import InstantaneousAction, DurativeAction, Fluent, Parameter, Problem, Object
from unified_planning.model.types import _UserType as UT
from unified_planning.exceptions import UPTypeError, UPProblemDefinitionError
from unified_planning.model.types import _UserType
from typing import Callable, Dict, IO, Iterable, List, Optional, Union, cast
from io import StringIO
from functools import reduce

PDDL_KEYWORDS = {'define', 'domain', 'requirements', 'types', 'constants', 'atomic', 'predicates',
                 'problem', 'atomic', 'constraints', 'either', 'number', 'action', 'parameters',
                 'precondition', 'effect', 'and', 'forall', 'preference', 'or', 'not', 'imply',
                 'exists', 'scale-up', 'scale-down', 'increase', 'decrease', 'durative-action', 'duration',
                 'condition', 'at', 'over', 'start', #here 'objects',  'functions',
                 'when', 'increse', 'decrease', 'define',
                 'imply'}
#TODO Fill this set

# The following map is used to mangle the invalid names by their class.
INITIAL_LETTER: Dict[type, str] = {InstantaneousAction: 'a', DurativeAction: 'a', Fluent: 'f', Parameter: 'p', Problem: 'p', Object: 'o'}


class ConverterToPDDLString(walkers.DagWalker):
    '''Expression converter to a PDDL string.'''

    DECIMAL_PRECISION = 10 # Number of decimal places to print real constants

    def __init__(self, env: 'up.environment.Environment', get_mangled_name: Callable[[Union['up.model.Type', 'up.model.Action', 'up.model.Fluent', 'up.model.Object']], str]):
        walkers.DagWalker.__init__(self)
        self.get_mangled_name = get_mangled_name
        self.simplifier = env.simplifier

    def convert(self, expression):
        '''Converts the given expression to a PDDL string.'''
        self.variables = {}
        return self.walk(self.simplifier.simplify(expression))

    def walk_exists(self, expression, args):
        assert len(args) == 1
        vars_string_list = [f"?{self.get_mangled_name(v, self.variables)} - {self.get_mangled_name(v.type)}" for v in expression.variables()]
        return f'(exists ({" ".join(vars_string_list)})\n {args[0]})'

    def walk_forall(self, expression, args):
        assert len(args) == 1
        vars_string_list = [f"?{self.get_mangled_name(v, self.variables)} - {self.get_mangled_name(v.type)}" for v in expression.variables()]
        return f'(forall ({" ".join(vars_string_list)})\n {args[0]})'

    def walk_variable_exp(self, expression, args):
        assert len(args) == 0
        return f'?{self.get_mangled_name(expression.variable(), self.variables)}'

    def walk_and(self, expression, args):
        assert len(args) > 1
        return f'(and {" ".join(args)})'

    def walk_or(self, expression, args):
        assert len(args) > 1
        return f'(or {" ".join(args)})'

    def walk_not(self, expression, args):
        assert len(args) == 1
        return f'(not {args[0]})'

    def walk_implies(self, expression, args):
        assert len(args) == 2
        return f'(imply {args[0]} {args[1]})'

    def walk_iff(self, expression, args):
        assert len(args) == 2
        return f'(and (imply {args[0]} {args[1]}) (imply {args[1]} {args[0]}) )'

    def walk_fluent_exp(self, expression, args):
        fluent = expression.fluent()
        return f'({self.get_mangled_name(fluent)}{" " if len(args) > 0 else ""}{" ".join(args)})'

    def walk_param_exp(self, expression, args):
        assert len(args) == 0
        p = expression.parameter()
        return f'?{self.get_mangled_name(p)}'

    def walk_object_exp(self, expression, args):
        assert len(args) == 0
        o = expression.object()
        return f'{self.get_mangled_name(o)}'

    def walk_bool_constant(self, expression, args):
        raise up.exceptions.UPUnreachableCodeError

    def walk_real_constant(self, expression, args):
        assert len(args) == 0
        frac = expression.constant_value()

        with localcontext() as ctx:
            ctx.prec = self.DECIMAL_PRECISION
            dec = frac.numerator / Decimal(frac.denominator, ctx)

            if Fraction(dec) != frac:
                warn("The PDDL printer cannot exactly represent the real constant '%s'" % frac)
            return str(dec)

    def walk_int_constant(self, expression, args):
        assert len(args) == 0
        return str(expression.constant_value())

    def walk_plus(self, expression, args):
        assert len(args) > 1
        return reduce(lambda x, y: f'(+ {y} {x})', args)

    def walk_minus(self, expression, args):
        assert len(args) == 2
        return f'(- {args[0]} {args[1]})'

    def walk_times(self, expression, args):
        assert len(args) > 1
        return reduce(lambda x, y: f'(* {y} {x})', args)

    def walk_div(self, expression, args):
        assert len(args) == 2
        return f'(/ {args[0]} {args[1]})'

    def walk_le(self, expression, args):
        assert len(args) == 2
        return f'(<= {args[0]} {args[1]})'

    def walk_lt(self, expression, args):
        assert len(args) == 2
        return f'(< {args[0]} {args[1]})'

    def walk_equals(self, expression, args):
        assert len(args) == 2
        return f'(= {args[0]} {args[1]})'


class PDDLWriter:
    '''This class can be used to write a Problem in PDDL.'''

    def __init__(self, problem: 'up.model.Problem', needs_requirements: bool = True):
        self.problem = problem
        self.needs_requirements = needs_requirements
        # otn represents the old to new renamings
        self.otn_renamings: Dict[str, str] = {}
        # nto represents the new to old renamings
        self.nto_renamings: Dict[str, str] = {}
        # those 2 maps are "simmetrical", meaning that "(otn[k] == v) implies (nto[v] == k)"
        # current_action_parameters represent the map between a parameters name and it's name in PDDL
        self.current_parameters: Dict[str, str] = {}

    def _write_domain(self, out: IO[str]):
        problem_kind = self.problem.kind
        if problem_kind.has_intermediate_conditions_and_effects():
            raise UPProblemDefinitionError('PDDL2.1 does not support ICE.\nICE are Intermediate Conditions and Effects therefore when an Effect (or Condition) are not at StartTIming(0) or EndTIming(0).')
        if problem_kind.has_timed_effect() or problem_kind.has_timed_goals():
            raise UPProblemDefinitionError('PDDL2.1 does not support timed effects or timed goals.')
        out.write('(define ')
        if self.problem.name is None:
            name = 'pddl'
        else:
            name = _get_pddl_name(self.problem)
        out.write(f'(domain {name}-domain)\n')

        if self.needs_requirements:
            out.write(' (:requirements :strips')
            if problem_kind.has_flat_typing():
                out.write(' :typing')
            if problem_kind.has_negative_conditions():
                out.write(' :negative-preconditions')
            if problem_kind.has_disjunctive_conditions():
                out.write(' :disjunctive-preconditions')
            if problem_kind.has_equality():
                out.write(' :equality')
            if (problem_kind.has_continuous_numbers() or
                problem_kind.has_discrete_numbers()):
                out.write(' :numeric-fluents')
            if problem_kind.has_conditional_effects():
                out.write(' :conditional-effects')
            if problem_kind.has_existential_conditions():
                out.write(' :existential-preconditions')
            if problem_kind.has_universal_conditions():
                out.write(' :universal-preconditions')
            if (problem_kind.has_continuous_time() or
                problem_kind.has_discrete_time()):
                out.write(' :durative-actions')
            if problem_kind.has_duration_inequalities():
                out.write(' :duration-inequalities')
            if (self.problem.kind.has_actions_cost() or
                self.problem.kind.has_plan_length()):
                out.write(' :action-costs')
            out.write(')\n')

        if problem_kind.has_hierarchical_typing():
            object_freshname = 'object'
            while self.problem.has_type(object_freshname):
                object_freshname = object_freshname + '_'
            self.otn_renamings['object'] = object_freshname
            self.nto_renamings[object_freshname] = 'object'
            user_types_hierarchy = self.problem.user_types_hierarchy
            out.write(f' (:types\n')
            stack: List['unified_planning.model.Type'] = user_types_hierarchy[None] if None in user_types_hierarchy else []
            out.write(f'    {" ".join(self._get_mangled_name(t) for t in stack)} - object\n')
            while stack:
                current_type = stack.pop()
                direct_sons: List['unified_planning.model.Type'] = user_types_hierarchy[current_type]
                if direct_sons:
                    stack.extend(direct_sons)
                    out.write(f'    {" ".join([self._get_mangled_name(t) for t in direct_sons])} - {self._get_mangled_name(current_type)}\n')
            out.write(' )\n')
        else:
            out.write(f' (:types {" ".join([self._get_mangled_name(t) for t in self.problem.user_types])})\n' if len(self.problem.user_types) > 0 else '')

        predicates = []
        functions = []
        for f in self.problem.fluents:
            self.current_parameters = {}
            if f.type.is_bool_type():
                params = []
                i = 0
                for param in f.signature:
                    if param.type.is_user_type():
                        params.append(f' ?{self._get_mangled_name(param)} - {self._get_mangled_name(param.type)}')
                        i += 1
                    else:
                        raise UPTypeError('PDDL supports only user type parameters')
                predicates.append(f'({self._get_mangled_name(f)}{"".join(params)})')
            elif f.type.is_int_type() or f.type.is_real_type():
                params = []
                i = 0
                for param in f.signature:
                    if param.type.is_user_type():
                        params.append(f' ?{self._get_mangled_name(param)} - {self._get_mangled_name(param.type)}')
                        i += 1
                    else:
                        raise UPTypeError('PDDL supports only user type parameters')
                functions.append(f'({self._get_mangled_name(f)}{"".join(params)})')
            else:
                raise UPTypeError('PDDL supports only boolean and numerical fluents')
        if (self.problem.kind.has_actions_cost() or
            self.problem.kind.has_plan_length()):
            functions.append('(total-cost)')
        out.write(f' (:predicates {" ".join(predicates)})\n' if len(predicates) > 0 else '')
        out.write(f' (:functions {" ".join(functions)})\n' if len(functions) > 0 else '')

        converter = ConverterToPDDLString(self.problem.env, self._get_mangled_name)
        costs = {}
        metrics = self.problem.quality_metrics
        if len(metrics) == 1:
            metric = metrics[0]
            if isinstance(metric, up.model.metrics.MinimizeActionCosts):
                for a in self.problem.actions:
                    costs[a] = metric.get_action_cost(a)
            elif isinstance(metric, up.model.metrics.MinimizeSequentialPlanLength):
                for a in self.problem.actions:
                    costs[a] = self.problem.env.expression_manager.Int(1)
        elif len(metrics) > 1:
            raise up.exceptions.UPUnsupportedProblemTypeError('Only one metric is supported!')
        for a in self.problem.actions:
            self.current_parameters = {}
            if isinstance(a, up.model.InstantaneousAction):
                out.write(f' (:action {self._get_mangled_name(a)}')
                out.write(f'\n  :parameters (')
                for ap in a.parameters:
                    if ap.type.is_user_type():
                        out.write(f' ?{self._get_mangled_name(ap)} - {self._get_mangled_name(ap.type)}')
                    else:
                        raise UPTypeError('PDDL supports only user type parameters')
                out.write(')')
                if len(a.preconditions) > 0:
                    out.write(f'\n  :precondition (and {" ".join([converter.convert(p) for p in a.preconditions])})')
                if len(a.effects) > 0:
                    out.write('\n  :effect (and')
                    for e in a.effects:
                        if e.is_conditional():
                            out.write(f' (when {converter.convert(e.condition)}')
                        if e.value.is_true():
                            out.write(f' {converter.convert(e.fluent)}')
                        elif e.value.is_false():
                            out.write(f' (not {converter.convert(e.fluent)})')
                        elif e.is_increase():
                            out.write(f' (increase {converter.convert(e.fluent)} {converter.convert(e.value)})')
                        elif e.is_decrease():
                            out.write(f' (decrease {converter.convert(e.fluent)} {converter.convert(e.value)})')
                        else:
                            out.write(f' (assign {converter.convert(e.fluent)} {converter.convert(e.value)})')
                        if e.is_conditional():
                            out.write(f')')

                    if a in costs:
                        out.write(f' (increase total-cost {converter.convert(costs[a])})')
                    out.write(')')
                out.write(')\n')
            elif isinstance(a, DurativeAction):
                out.write(f' (:durative-action {self._get_mangled_name(a)}')
                out.write(f'\n  :parameters (')
                for ap in a.parameters:
                    if ap.type.is_user_type():
                        out.write(f' ?{self._get_mangled_name(ap)} - {self._get_mangled_name(ap.type)}')
                    else:
                        raise UPTypeError('PDDL supports only user type parameters')
                out.write(')')
                l, r = a.duration.lower, a.duration.upper
                if l == r:
                    out.write(f'\n  :duration (= ?duration {converter.convert(l)})')
                else:
                    out.write(f'\n  :duration (and ')
                    if a.duration.is_left_open():
                        out.write(f'(> ?duration {converter.convert(l)})')
                    else:
                        out.write(f'(>= ?duration {converter.convert(l)})')
                    if a.duration.is_right_open():
                        out.write(f'(< ?duration {converter.convert(r)})')
                    else:
                        out.write(f'(<= ?duration {converter.convert(r)})')
                    out.write(')')
                if len(a.conditions) > 0:
                    out.write(f'\n  :condition (and ')
                    for interval, cl in a.conditions.items():
                        for c in cl:
                            if interval.lower == interval.upper:
                                if interval.lower.is_from_start():
                                    out.write(f'(at start {converter.convert(c)})')
                                else:
                                    out.write(f'(at end {converter.convert(c)})')
                            else:
                                if not interval.is_left_open():
                                    out.write(f'(at start {converter.convert(c)})')
                                out.write(f'(over all {converter.convert(c)})')
                                if not interval.is_right_open():
                                    out.write(f'(at end {converter.convert(c)})')
                    out.write(')')
                if len(a.effects) > 0:
                    out.write('\n  :effect (and')
                    for t, el in a.effects.items():
                        for e in el:
                            if t.is_from_start():
                                out.write(f' (at start')
                            else:
                                out.write(f' (at end')
                            if e.is_conditional():
                                out.write(f' (when {converter.convert(e.condition)}')
                            if e.value.is_true():
                                out.write(f' {converter.convert(e.fluent)}')
                            elif e.value.is_false():
                                out.write(f' (not {converter.convert(e.fluent)})')
                            elif e.is_increase():
                                out.write(f' (increase {converter.convert(e.fluent)} {converter.convert(e.value)})')
                            elif e.is_decrease():
                                out.write(f' (decrease {converter.convert(e.fluent)} {converter.convert(e.value)})')
                            else:
                                out.write(f' (assign {converter.convert(e.fluent)} {converter.convert(e.value)})')
                            if e.is_conditional():
                                out.write(f')')
                            out.write(')')
                    if a in costs:
                        out.write(f' (at end (increase total-cost {converter.convert(costs[a])}))')
                    out.write(')')
                out.write(')\n')
            else:
                raise NotImplementedError
        out.write(')\n')

    def _write_problem(self, out: IO[str]):
        if self.problem.name is None:
            name = 'pddl'
        else:
            name = _get_pddl_name(self.problem)
        out.write(f'(define (problem {name}-problem)\n')
        out.write(f' (:domain {name}-domain)\n')
        if len(self.problem.user_types) > 0:
            out.write(' (:objects')
            for t in self.problem.user_types:
                objects = [o for o in self.problem.all_objects if o.type == t]
                if len(objects) > 0:
                    out.write(f'\n   {" ".join([self._get_mangled_name(o) for o in objects])} - {self._get_mangled_name(t)}')
            out.write('\n )\n')
        converter = ConverterToPDDLString(self.problem.env, self._get_mangled_name)
        out.write(' (:init')
        for f, v in self.problem.initial_values.items():
            if v.is_true():
                out.write(f' {converter.convert(f)}')
            elif v.is_false():
                pass
            else:
                out.write(f' (= {converter.convert(f)} {converter.convert(v)})')
        if self.problem.kind.has_actions_cost():
            out.write(f' (= total-cost 0)')
        out.write(')\n')
        out.write(f' (:goal (and {" ".join([converter.convert(p) for p in self.problem.goals])}))\n')
        metrics = self.problem.quality_metrics
        if len(metrics) == 1:
            metric = metrics[0]
            out.write(' (:metric ')
            if isinstance(metric, up.model.metrics.MinimizeExpressionOnFinalState):
                out.write(f'minimize {metric.expression}')
            elif isinstance(metric, up.model.metrics.MaximizeExpressionOnFinalState):
                out.write(f'maximize {metric.expression}')
            elif (isinstance(metric, up.model.metrics.MinimizeActionCosts) or
                  isinstance(metric, up.model.metrics.MinimizeSequentialPlanLength)):
                out.write(f'minimize total-cost')
            elif isinstance(metric, up.model.metrics.MinimizeMakespan):
                out.write(f'minimize total-time')
            else:
                raise NotImplementedError
            out.write(')\n')
        elif len(metrics) > 1:
            raise up.exceptions.UPUnsupportedProblemTypeError('Only one metric is supported!')
        out.write(')\n')

    def print_domain(self):
        '''Prints to std output the PDDL domain.'''
        self._write_domain(sys.stdout)

    def print_problem(self):
        '''Prints to std output the PDDL problem.'''
        self._write_problem(sys.stdout)

    def get_domain(self) -> str:
        '''Returns the PDDL domain.'''
        out = StringIO()
        self._write_domain(out)
        return out.getvalue()

    def get_problem(self) -> str:
        '''Returns the PDDL problem.'''
        out = StringIO()
        self._write_problem(out)
        return out.getvalue()

    def write_domain(self, filename: str):
        '''Dumps to file the PDDL domain.'''
        with open(filename, 'w') as f:
            self._write_domain(f)

    def write_problem(self, filename: str):
        '''Dumps to file the PDDL problem.'''
        with open(filename, 'w') as f:
            self._write_problem(f)

    def _get_mangled_name(self,
                          item: Union['up.model.Type', 'up.model.Action', 'up.model.Fluent', 'up.model.Object', 'up.model.Parameter', 'up.model.Variable'],
                          variables: Optional[Dict['up.model.Variable', str]] = None) -> str:
        '''This function returns a valid and unique PDDL name.'''
        if isinstance(item, up.model.Parameter):
            assert variables is None
            # name already seen
            if item.name in self.current_parameters:
                return self.current_parameters[item.name]
            else:
                # new parameter, check conflicts with the names in the problem, in the renamings or in the parameters.
                new_name = self._unique_name_checking_iterables(_get_pddl_name(item),
                                                                self.nto_renamings,
                                                                self.current_parameters.values())
                self.current_parameters[item.name] = new_name
                return new_name
        elif isinstance(item, up.model.Variable):
            assert variables is not None
            # variable already seen
            if item in variables:
                return variables[item]
            else:
                # new variable, check conflicts with the names in the problem, in the renamings, in the parameters
                # or in the variables.
                new_name = self._unique_name_checking_iterables(_get_pddl_name(item),
                                                                self.nto_renamings,
                                                                self.current_parameters.values(),
                                                                variables.values())
                variables[item] = new_name
                return new_name
        elif isinstance(item, up.model.Type):
            assert item.is_user_type()
            original_name = cast(_UserType, item).name
        else:
            original_name = item.name
        # If we already encountered this name, return it
        if original_name in self.otn_renamings:
            return self.otn_renamings[original_name]
        tmp_name = _get_pddl_name(item)
        # if the pddl valid name is the same of the original one, it can be returned
        if tmp_name == original_name:
            new_name = tmp_name
        else:
            new_name = self._unique_name_checking_iterables(tmp_name, self.nto_renamings)
        assert variables is None
        assert new_name not in self.nto_renamings and new_name not in self.otn_renamings.values()
        self.otn_renamings[original_name] = new_name
        self.nto_renamings[new_name] = original_name
        return new_name

    def _unique_name_checking_iterables(self, tmp_name: str, *iterables: Iterable[str]) -> str:
        '''This method returns a name from tmp_name that isn't in the problem or that isn't in any given iterable.'''
        count = 0
        new_name = tmp_name
        while self.problem.has_name(new_name) or any(new_name in i for i in iterables):
            new_name = f'{tmp_name}_{count}'
            count += 1
        return new_name

    def get_renamings(self) -> Dict[str, str]:
        '''
        This method returns the map in which every key is the name of the object/fluent/action/type chosen for the pddl printing
        and the associated value is the original name in the unified_planning Problem.

        :return: a map where every key is the name chosen for the pddl and every value is the original name.'''
        return self.nto_renamings

def _get_pddl_name(item: Union['up.model.Type', 'up.model.Action', 'up.model.Fluent', 'up.model.Object','up.model.Parameter', 'up.model.Variable', 'up.model.Problem']) -> str:
    '''This function returns a pddl name for the chosen item'''
    name = item.name # type: ignore
    assert name is not None
    name = name.lower()
    regex = re.compile(r'^[a-zA-Z]+.*')
    if re.match(regex, name) is None: # If the name does not start with an alphabetic char, we make it start with one.
        name = f'{INITIAL_LETTER.get(type(item), "x")}_{name}'

    name = re.sub('[^0-9a-zA-Z_]', '_', name) #Substitute non-valid elements with "_"
    while name in PDDL_KEYWORDS: # If the name is in the keywords, apply an underscore at the end until it is not a keyword anymore.
        name = f'{name}_'
    return name
