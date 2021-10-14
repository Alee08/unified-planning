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
"""This module defines the conditional effects remover class."""

from collections import OrderedDict
from upf.temporal import DurativeAction, Timing
from upf.plan import SequentialPlan, ActionInstance, TimeTriggeredPlan
from upf.problem import Problem
from upf.action import ActionInterface, Action
from upf.effect import Effect
from upf.fnode import FNode
from upf.simplifier import Simplifier
from typing import Iterable, List, Dict, Tuple, Union
from itertools import chain, combinations


class ConditionalEffectsRemover():
    '''Conditional effect remover class:
    this class requires a problem and offers the capability
    to transform a conditional problem into an unconditional one.

    This is done by substituting every conditional action with different
    actions representing every possible branch of the original action.'''
    def __init__(self, problem: Problem):
        self._problem = problem
        self._action_mapping: Dict[ActionInterface, ActionInterface] = {}
        self._env = problem.env
        self._counter: int = 0
        self._unconditional_problem = None
        self._simplifier = Simplifier(self._env)

    def powerset(self, iterable: Iterable) -> Iterable:
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def get_rewritten_problem(self) -> Problem:
        '''Creates a problem that is a copy of the original problem
        but every conditional action is removed and all the possible
        branches of the conditional action are added as non-conditional
        actions.'''
        if self._unconditional_problem is not None:
            return self._unconditional_problem
        #cycle over all the actions
        #NOTE that a different environment might be needed when multy-threading
        new_problem = self._create_problem_copy()

        for action in self._problem.conditional_actions():
            if isinstance(action, Action):
                cond_effects = action.conditional_effects()
                for p in self.powerset(range(len(cond_effects))):
                    na = self._shallow_copy_action_without_conditional_effects(action)
                    for i, e in enumerate(cond_effects):
                        if i in p:
                            # positive precondition
                            na.add_precondition(e.condition())
                            ne = Effect(e.fluent(), e.value(), self._env.expression_manager.TRUE(), e.kind())
                            na._add_effect_instance(ne)
                        else:
                            #negative precondition
                            na.add_precondition(self._env.expression_manager.Not(e.condition()))
                    #new action is created, then is checked if it has any impact and if it can be simplified
                    if len(na.effects()) > 0:
                        if self._check_and_simplify_preconditions(na):
                            self._action_mapping[na] = action
                            new_problem.add_action(na)
            elif isinstance(action, DurativeAction):
                timing_cond_effects: Dict[Timing, List[Effect]] = action.conditional_effects()
                cond_effects_timing: List[Tuple[Effect, Timing]] = [(e, t) for t, el in timing_cond_effects.items() for e in el]
                for p in self.powerset(range(len(cond_effects_timing))):
                    nda = self._shallow_copy_durative_action_without_conditional_effects(action)
                    for i, (e, t) in enumerate(cond_effects_timing):
                        if i in p:
                            # positive precondition
                            nda.add_condition(t, e.condition())
                            ne = Effect(e.fluent(), e.value(), self._env.expression_manager.TRUE(), e.kind())
                            nda._add_effect_instance(t, ne)
                        else:
                            #negative precondition
                            nda.add_condition(t, self._env.expression_manager.Not(e.condition()))
                    #new action is created, then is checked if it has any impact and if it can be simplified
                    if len(nda.effects()) > 0:
                        if self._check_and_simplify_conditions(nda):
                            self._action_mapping[nda] = action
                            new_problem.add_action(nda)
            else:
                raise NotImplementedError
        self._unconditional_problem = new_problem
        return new_problem

    def _check_and_simplify_conditions(self, action: DurativeAction) -> bool:
        '''Simplifies conditions and if it is False (a contraddiction)
        returns False, otherwise returns True.
        If the simplification is True (a tautology) removes all conditions at the given timing.
        If the simplification is still an AND rewrites back every "arg" of the AND
        in the conditions
        If the simplification is not an AND sets the simplification as the only
        condition at the given timing.'''
        #action conditions
        #tlc = timing list condition
        tlc: Dict[Timing, List[FNode]] = action.conditions()
        if len(tlc) == 0:
            return True
        # t = timing, lc = list condition
        for t, lc in tlc.copy().items():
            #conditions (as an And FNode)
            c = self._env.expression_manager.And(lc)
            #conditions simplified
            cs = self._simplifier.simplify(c)
            #new action conditions
            nac: List[FNode] = []
            if cs.is_bool_constant():
                if not cs.bool_constant_value():
                    return False
            else:
                if cs.is_and():
                    nac.extend(cs.args())
                else:
                    nac.append(cs)
            action._set_conditions(t, nac)
        return True

    def _check_and_simplify_preconditions(self, action: Action) -> bool:
        '''Simplifies preconditions and if it is False (a contraddiction)
        returns False, otherwise returns True.
        If the simplification is True (a tautology) removes all preconditions.
        If the simplification is still an AND rewrites back every "arg" of the AND
        in the preconditions
        If the simplification is not an AND sets the simplification as the only
        precondition.'''
        #action preconditions
        ap = action.preconditions()
        if len(ap) == 0:
            return True
        #preconditions (as an And FNode)
        p = self._env.expression_manager.And(ap)
        #preconditions simplified
        ps = self._simplifier.simplify(p)
        #new action preconditions
        nap: List[FNode] = []
        if ps.is_bool_constant():
            if not ps.bool_constant_value():
                return False
        else:
            if ps.is_and():
                nap.extend(ps.args())
            else:
                nap.append(ps)
        action._set_preconditions(nap)
        return True

    def _shallow_copy_action_without_conditional_effects(self, action: Action) -> Action:
        #emulates a do-while loop: searching for an available name
        while True:
            new_action_name = action.name()+ "_" +str(self._counter)
            self._counter = self._counter + 1
            if not self._problem.has_action(new_action_name):
                break
        new_parameters = OrderedDict()
        for ap in action.parameters():
            new_parameters[ap.name()] = ap.type()
        new_action = Action(new_action_name, new_parameters, self._env)
        for p in action.preconditions():
            new_action.add_precondition(p)
        for e in action.unconditional_effects():
            new_action._add_effect_instance(e)
        return new_action

    def _shallow_copy_durative_action_without_conditional_effects(self, action: DurativeAction) -> DurativeAction:
        #emulates a do-while loop: searching for an available name
        while True:
            new_action_name = action.name()+ "_" +str(self._counter)
            self._counter = self._counter + 1
            if not self._problem.has_action(new_action_name):
                break
        new_parameters = OrderedDict()
        for ap in action.parameters():
            new_parameters[ap.name()] = ap.type()
        new_action = DurativeAction(new_action_name, new_parameters, self._env)
        new_action.set_duration_constraint(action.duration())
        for t, c in action.conditions().items():
            new_action.add_condition(t, c)
        for i, dc in action.durative_conditions().items():
            new_action.add_durative_condition(i, dc)
        for t, e in action.unconditional_effects():
            new_action._add_effect_instance(t, e)
        return new_action

    def _create_problem_copy(self):
        '''Creates the shallow copy of a problem, without adding the conditional actions
        '''
        new_problem: Problem = Problem("unconditional_" + str(self._problem.name()), self._env)
        for f in self._problem.fluents().values():
            new_problem.add_fluent(f)
        for o in self._problem.all_objects():
            new_problem.add_object(o)
        for fl, v in self._problem.initial_values().items():
            new_problem.set_initial_value(fl, v)
        for t, el in self._problem.timed_effects():
            for e in el:
                if e.is_conditional():
                    raise # NOTE is it a problem if a timed_effect is conditional? It should be if this class is used!
                new_problem.add_timed_effect(t, e)
        for t, gl in self._problem.timed_goals():
            for g in gl:
                new_problem.add_timed_goal(t, g)
        for i, gl in self._problem.mantain_goals():
            for g in gl:
                new_problem.add_mantain_goal(i, g)
        for g in self._problem.goals():
            new_problem.add_goal(g)
        for ua in self._problem.unconditional_actions():
            new_problem.add_action(ua)
        return new_problem

    def rewrite_back_plan(self, plan: Union[SequentialPlan, TimeTriggeredPlan]) -> Union[SequentialPlan, TimeTriggeredPlan]:
        '''Takes the sequential plan of the non-conditional problem (created with
        the method "self.get_rewritten_problem()" and translates the plan back
        to be a plan of the original problem.'''
        if isinstance(plan, SequentialPlan):
            uncond_actions = plan.actions()
            cond_actions = []
            for ai in uncond_actions:
                if ai.action() in self._action_mapping:
                    cond_actions.append(self._new_action_instance_original_name(ai))
                else:
                    cond_actions.append(ai)
            return SequentialPlan(cond_actions)
        elif isinstance(plan, TimeTriggeredPlan):
            uncond_durative_actions = plan.actions()
            cond_durative_actions = []
            for s, ai, d in uncond_durative_actions:
                if ai.action() in self._action_mapping:
                    cond_durative_actions.append((s, self._new_action_instance_original_name(ai), d))
                else:
                    cond_durative_actions.append((s, ai, d))
            return TimeTriggeredPlan(cond_durative_actions)
        else:
            raise NotImplementedError

    def _new_action_instance_original_name(self, ai: ActionInstance) -> ActionInstance:
        #original action
        oa = self._action_mapping[ai.action()]
        return ActionInstance(oa, ai.actual_parameters())
