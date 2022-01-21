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
from upf.solvers.upf_tarski_converter import TarskiConverter
from upf.test import TestCase
from upf.test.examples import get_example_problems
from upf.interop.tarski import convert_tarski_problem
from upf.model.problem_kind import full_classical_kind


class TestGrounder(TestCase):
    def setUp(self):
        TestCase.setUp(self)
        self.problems = get_example_problems()
        self.tc = TarskiConverter()

    def test_basic(self):
        problem = self.problems['basic'].problem
        tarski_problem = self.tc.upf_to_tarski(problem)
        new_problem = convert_tarski_problem(problem.env, tarski_problem)
        self.assertEqual(problem, new_problem)

    def test_all_non_numerical(self):
        for p in self.problems.values():
            problem = p.problem
            problem_kind = problem.kind()
            if problem_kind <= full_classical_kind:
                if problem.name == "charger_discharger":
                    continue #the charger_discharger problem has Implies, which tarski represents with Or and Not
                            #therefore the 2 problems will not be equals
                #modify the problem to have the same representation
                modified_problem = problem.clone()
                for action in modified_problem.actions():
                    if len(action.preconditions()) > 1:
                        new_precondition_as_and_of_preconditions = modified_problem.env.expression_manager.And(\
                                    action.preconditions())
                        action._set_preconditions([new_precondition_as_and_of_preconditions])
                if len(modified_problem.goals()) > 1:
                    new_goal_as_and_of_goals = modified_problem.env.expression_manager.And(\
                                    modified_problem.goals())
                    modified_problem.clear_goals()
                    modified_problem.add_goal(new_goal_as_and_of_goals)
                # print("_____ORIGINAL_PROBLEMMMM")
                # print(modified_problem)
                tarski_problem = self.tc.upf_to_tarski(modified_problem)
                new_problem = convert_tarski_problem(modified_problem.env, tarski_problem)
                if not modified_problem == new_problem:
                    print("_______ORIGINAL_PROBLEM___________")
                    print(modified_problem)
                    print("_______TARSKI_PROBLEM___________")
                    print(tarski_problem)
                    print(tarski_problem.goal)
                    print(tarski_problem.init)
                    print(tarski_problem.actions)
                    for n, a in tarski_problem.actions.items():
                        print(a)
                        print(a.precondition)
                        print(a.effects)
                    print("_______CREATED_PROBLEM___________")
                    print(new_problem)
                self.assertEqual(modified_problem, new_problem)


    # def test_basic_conditional(self):
    #     problem = self.problems['basic_conditional'].problem
    #     tarski_problem = self.tc.upf_to_tarski(problem)
    #     print(problem)
    #     print(tarski_problem)
    #     print(tarski_problem.goal)
    #     print(tarski_problem.init)
    #     assert False


    # def test_complex_conditional(self):
    #     problem = self.problems['complex_conditional'].problem
    #     tarski_problem = self.tc.upf_to_tarski(problem)
    #     print(problem)
    #     print(tarski_problem)
    #     print(tarski_problem.goal)
    #     print(tarski_problem.init)
    #     print(tarski_problem.actions)
    #     for n, a in tarski_problem.actions.items():
    #         print(a)
    #         print(a.precondition)
    #         print(a.effects)
    #     assert False


    # def test_basic_nested_conjunctions(self):
    #     problem = self.problems['basic_nested_conjunctions'].problem
    #     tarski_problem = self.tc.upf_to_tarski(problem)
    #     print(problem)
    #     print(tarski_problem)
    #     print(tarski_problem.goal)
    #     print(tarski_problem.init)
    #     print(tarski_problem.actions)
    #     for n, a in tarski_problem.actions.items():
    #         print(a)
    #         print(a.precondition)
    #         print(a.effects)
    #     assert False


    # def test_basic_exists(self):
    #     problem = self.problems['basic_exists'].problem
    #     tarski_problem = self.tc.upf_to_tarski(problem)
    #     print(problem)
    #     print(tarski_problem)
    #     print(tarski_problem.goal)
    #     print(tarski_problem.init)
    #     print(tarski_problem.actions)
    #     for n, a in tarski_problem.actions.items():
    #         print(a)
    #         print(a.precondition)
    #         print(a.effects)
    #     assert False


    # def test_basic_forall(self):
    #     problem = self.problems['basic_forall'].problem
    #     tarski_problem = self.tc.upf_to_tarski(problem)
    #     print(problem)
    #     print(tarski_problem)
    #     print(tarski_problem.goal)
    #     print(tarski_problem.init)
    #     print(tarski_problem.actions)
    #     for n, a in tarski_problem.actions.items():
    #         print(a)
    #         print(a.precondition)
    #         print(a.effects)
    #     assert False


    # def test_robot(self):
    #     problem = self.problems['robot'].problem
    #     tarski_problem = self.tc.upf_to_tarski(problem)
    #     print(problem)
    #     print(tarski_problem)
    #     print(tarski_problem.goal)
    #     print(tarski_problem.init)
    #     print(tarski_problem.actions)
    #     for n, a in tarski_problem.actions.items():
    #         print(a)
    #         print(a.precondition)
    #         print(a.effects)
    #     assert False
