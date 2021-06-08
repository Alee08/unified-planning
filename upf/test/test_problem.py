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

import upf
from upf.shortcuts import *
from upf.test import TestCase, main


class TestProblem(TestCase):
    def test_basic(self):
        x = upf.Fluent('x')
        self.assertEqual(x.name(), 'x')
        self.assertEqual(x.arity(), 0)
        self.assertTrue(x.type().is_bool_type())

        a = upf.Action('a')
        a.add_precondition(Not(x))
        a.add_effect(x, True)
        self.assertEqual(a.name(), 'a')
        self.assertEqual(len(a.preconditions()), 1)
        self.assertEqual(len(a.effects()), 1)

        problem = upf.Problem('basic')
        problem.add_fluent(x)
        problem.add_action(a)
        problem.set_initial_value(x, False)
        problem.add_goal(x)
        self.assertEqual(problem.name(), 'basic')
        self.assertEqual(len(problem.fluents()), 1)
        self.assertEqual(problem.fluent('x'), x)
        self.assertEqual(len(problem.actions()), 1)
        self.assertEqual(problem.action('a'), a)
        self.assertTrue(problem.initial_value(x) is not None)
        self.assertEqual(len(problem.goals()), 1)

    def test_with_parameters(self):
        Location = UserType('Location')
        self.assertTrue(Location.is_user_type())
        self.assertEqual(Location.name(), 'Location')

        robot_at = upf.Fluent('robot_at', BoolType(), [Location])
        self.assertEqual(robot_at.name(), 'robot_at')
        self.assertEqual(robot_at.arity(), 1)
        self.assertEqual(robot_at.signature(), [Location])
        self.assertTrue(robot_at.type().is_bool_type())

        move = upf.Action('move', l_from=Location, l_to=Location)
        l_from = move.parameter('l_from')
        l_to = move.parameter('l_to')
        move.add_precondition(Not(Equals(l_from, l_to)))
        move.add_precondition(robot_at(l_from))
        move.add_precondition(Not(robot_at(l_to)))
        move.add_effect(robot_at(l_from), False)
        move.add_effect(robot_at(l_to), True)
        self.assertEqual(move.name(), 'move')
        self.assertEqual(len(move.parameters()), 2)
        self.assertEqual(l_from.name(), 'l_from')
        self.assertEqual(l_from.type(), Location)
        self.assertEqual(l_to.name(), 'l_to')
        self.assertEqual(l_to.type(), Location)
        self.assertEqual(len(move.preconditions()), 3)
        self.assertEqual(len(move.effects()), 2)

        l1 = upf.Object('l1', Location)
        l2 = upf.Object('l2', Location)
        self.assertEqual(l1.name(), 'l1')
        self.assertEqual(l1.type(), Location)
        self.assertEqual(l2.name(), 'l2')
        self.assertEqual(l2.type(), Location)

        p = upf.Problem('robot')
        p.add_fluent(robot_at)
        p.add_action(move)
        p.add_object(l1)
        p.add_object(l2)
        p.set_initial_value(robot_at(l1), True)
        p.set_initial_value(robot_at(l2), False)
        p.add_goal(robot_at(l2))
        self.assertEqual(p.name(), 'robot')
        self.assertEqual(len(p.fluents()), 1)
        self.assertEqual(p.fluent('robot_at'), robot_at)
        self.assertEqual(len(p.user_types()), 1)
        self.assertEqual(p.user_types()['Location'], Location)
        self.assertEqual(len(p.objects(Location)), 2)
        self.assertEqual(p.objects(Location), [l1, l2])
        self.assertEqual(len(p.actions()), 1)
        self.assertEqual(p.action('move'), move)
        self.assertTrue(p.initial_value(robot_at(l1)) is not None)
        self.assertTrue(p.initial_value(robot_at(l2)) is not None)
        self.assertEqual(len(p.goals()), 1)

    def test_problem_kind(self):
        problem_kind = ProblemKind()
        self.assertFalse(problem_kind.has_discrete_time())
        self.assertFalse(problem_kind.has_continuous_time())
        problem_kind.set_time('DISCRETE_TIME')
        self.assertTrue(problem_kind.has_discrete_time())
        problem_kind.set_time('CONTINUOUS_TIME')
        self.assertTrue(problem_kind.has_continuous_time())

    def test_cargo_example(self):
        Location = UserType('Location')
        self.assertTrue(Location.is_user_type())
        self.assertEqual(Location.name(), 'Location')

        robot_at = upf.Fluent('robot_at', BoolType(), [Location])
        self.assertEqual(robot_at.name(), 'robot_at')
        self.assertEqual(robot_at.arity(), 1)
        self.assertEqual(robot_at.signature(), [Location])
        self.assertTrue(robot_at.type().is_bool_type())

        cargo_at = upf.Fluent('cargo_at', BoolType(), [Location])
        self.assertEqual(cargo_at.name(), 'cargo_at')
        self.assertEqual(cargo_at.arity(), 1)
        self.assertEqual(cargo_at.signature(), [Location])
        self.assertTrue(cargo_at.type().is_bool_type())

        cargo_mounted = upf.Fluent('cargo_mounted')
        self.assertEqual(cargo_mounted.name(), 'cargo_mounted')
        self.assertEqual(cargo_mounted.arity(), 0)
        self.assertTrue(cargo_mounted.type().is_bool_type())

        move = upf.Action('move', l_from=Location, l_to=Location)
        l_from = move.parameter('l_from')
        l_to = move.parameter('l_to')
        move.add_precondition(Not(Equals(l_from, l_to)))
        move.add_precondition(robot_at(l_from))
        move.add_precondition(Not(robot_at(l_to)))
        move.add_effect(robot_at(l_from), False)
        move.add_effect(robot_at(l_to), True)
        self.assertEqual(move.name(), 'move')
        self.assertEqual(len(move.parameters()), 2)
        self.assertEqual(l_from.name(), 'l_from')
        self.assertEqual(l_from.type(), Location)
        self.assertEqual(l_to.name(), 'l_to')
        self.assertEqual(l_to.type(), Location)
        self.assertEqual(len(move.preconditions()), 3)
        self.assertEqual(len(move.effects()), 2)

        load = upf.Action('load', loc = Location)
        loc = load.parameter('loc')
        load.add_precondition(cargo_at(loc))
        load.add_precondition(robot_at(loc))
        load.add_precondition(Not(cargo_mounted))
        load.add_effect(cargo_at(loc), False)
        load.add_effect(cargo_mounted, True)
        self.assertEqual(load.name(), 'load')
        self.assertEqual(len(load.parameters()), 1)
        self.assertEqual(loc.name(), 'loc')
        self.assertEqual(loc.type(), Location)
        self.assertEqual(len(load.preconditions()), 3)
        self.assertEqual(len(load.effects()), 2)

        unload = upf.Action('unload',pos = Location)
        pos = unload.parameter('pos')
        unload.add_precondition(Not(cargo_at(pos)))
        unload.add_precondition(robot_at(pos))
        unload.add_precondition(cargo_mounted)
        unload.add_effect(cargo_at(pos), True)
        unload.add_effect(cargo_mounted, False)
        self.assertEqual(unload.name(), 'unload')
        self.assertEqual(len(unload.parameters()), 1)
        self.assertEqual(pos.name(), 'pos')
        self.assertEqual(pos.type(), Location)
        self.assertEqual(len(unload.preconditions()), 3)
        self.assertEqual(len(unload.effects()), 2)

        l1 = upf.Object('l1', Location)
        l2 = upf.Object('l2', Location)
        self.assertEqual(l1.name(), 'l1')
        self.assertEqual(l1.type(), Location)
        self.assertEqual(l2.name(), 'l2')
        self.assertEqual(l2.type(), Location)

        p = upf.Problem('robot_loader')
        p.add_fluent(robot_at)
        p.add_fluent(cargo_at)
        p.add_fluent(cargo_mounted)
        p.add_action(load)
        p.add_action(unload)
        p.add_action(move)
        p.add_object(l1)
        p.add_object(l2)
        p.set_initial_value(robot_at(l1), True)
        p.set_initial_value(robot_at(l2), False)
        p.set_initial_value(cargo_at(l1), False)
        p.set_initial_value(cargo_at(l2), True)
        p.set_initial_value(cargo_mounted, False)
        p.add_goal(cargo_at(l1))

        self.assertEqual(p.name(), 'robot_loader')
        self.assertEqual(len(p.fluents()), 3)
        self.assertEqual(p.fluent('robot_at'), robot_at)
        self.assertEqual(p.fluent('cargo_at'), cargo_at)
        self.assertEqual(p.fluent('cargo_mounted'), cargo_mounted)
        self.assertEqual(len(p.user_types()), 1)
        self.assertEqual(p.user_types()['Location'], Location)
        self.assertEqual(len(p.objects(Location)), 2)
        self.assertEqual(p.objects(Location), [l1, l2])
        self.assertEqual(len(p.actions()), 3)
        self.assertEqual(p.action('move'), move)
        self.assertEqual(p.action('load'), load)
        self.assertEqual(p.action('unload'), unload)
        self.assertTrue(p.initial_value(robot_at(l1)) is not None)
        self.assertTrue(p.initial_value(robot_at(l2)) is not None)
        self.assertTrue(p.initial_value(cargo_at(l1)) is not None)
        self.assertTrue(p.initial_value(cargo_at(l2)) is not None)
        self.assertTrue(p.initial_value(cargo_mounted) is not None)
        self.assertEqual(len(p.goals()), 1)

if __name__ == "__main__":
    main()
