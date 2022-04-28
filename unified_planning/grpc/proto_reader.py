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
import fractions
from typing import OrderedDict

import unified_planning.grpc.generated.unified_planning_pb2 as unified_planning_pb2
from unified_planning.grpc.converter import Converter, handles
import unified_planning.model
from unified_planning.model.operators import OperatorKind
from unified_planning.model.effect import EffectKind
import unified_planning.plan
from unified_planning.model import (
    Effect,
    Problem,
    Parameter,
    DurativeAction,
    InstantaneousAction,
)
from unified_planning.shortcuts import BoolType, UserType, RealType, IntType


def convert_type_str(s, env):
    if s == "bool":
        value_type = env.type_manager.BoolType()
    elif s == "int":
        value_type = env.type_manager.IntType()
    elif s == "float":
        value_type = env.type_manager.RealType()
    elif "real" in s:
        a = float(s.split("[")[1].split(",")[0])
        b = float(s.split(",")[1].split("]")[0])
        value_type = env.type_manager.RealType(a, b)
    else:
        value_type = env.type_manager.UserType(s)
    return value_type


# The operators are based on Sexpressions supported in PDDL.
def op_to_node_type(op: str) -> OperatorKind:
    if op == "+":
        return OperatorKind.PLUS
    elif op == "-":
        return OperatorKind.MINUS
    elif op == "*":
        return OperatorKind.TIMES
    elif op == "/":
        return OperatorKind.DIV
    elif op == "=":
        return OperatorKind.EQUALS
    elif op == "<=":
        return OperatorKind.LE
    elif op == "<":
        return OperatorKind.LT
    elif op == "and":
        return OperatorKind.AND
    elif op == "or":
        return OperatorKind.OR
    elif op == "not":
        return OperatorKind.NOT
    elif op == "exists":
        return OperatorKind.EXISTS
    elif op == "forall":
        return OperatorKind.FORALL
    elif op == "implies":
        return OperatorKind.IMPLIES
    elif op == "iff":
        return OperatorKind.IFF

    raise ValueError(f"Unknown operator `{op}`")


class ProtobufReader(Converter):
    current_action = None

    @handles(unified_planning_pb2.Parameter)
    def _convert_parameter(self, msg, problem):
        return Parameter(
            msg.name,
            convert_type_str(msg.type, problem.env),
        )

    @handles(unified_planning_pb2.Fluent)
    def _convert_fluent(self, msg, problem):
        value_type = convert_type_str(msg.value_type, problem.env)
        sig = []
        for p in msg.parameters:
            sig.append(self.convert(p, problem))
        fluent = unified_planning.model.Fluent(msg.name, value_type, sig, problem.env)
        return fluent

    @handles(unified_planning_pb2.ObjectDeclaration)
    def _convert_object(self, msg, problem):
        obj = unified_planning.model.Object(
            msg.name, problem.env.type_manager.UserType(msg.type)
        )
        return obj

    @handles(unified_planning_pb2.Expression)
    def _convert_expression(self, msg, problem, param_map):
        if msg.kind == unified_planning_pb2.ExpressionKind.Value("CONSTANT"):
            assert msg.atom is not None
            return self.convert(msg.atom, problem)

        elif msg.kind == unified_planning_pb2.ExpressionKind.Value("PARAMETER"):
            return problem.env.expression_manager.ParameterExp(
                param=Parameter(
                    msg.atom.symbol, problem.env.type_manager.UserType(msg.type)
                ),
            )
        elif msg.kind == unified_planning_pb2.ExpressionKind.Value("STATE_VARIABLE"):
            args = []
            payload = None

            fluent = msg.list.pop(0)
            if fluent.kind == unified_planning_pb2.ExpressionKind.Value(
                "FLUENT_SYMBOL"
            ):
                payload = self.convert(fluent.atom, problem)
            else:
                args.append(self.convert(fluent, problem, param_map))

            args.extend([self.convert(m, problem, param_map) for m in msg.list])

            return problem.env.expression_manager.create_node(
                node_type=OperatorKind.FLUENT_EXP,
                args=tuple(args),
                payload=payload,
            )
        elif msg.kind == unified_planning_pb2.ExpressionKind.Value(
            "FUNCTION_APPLICATION"
        ):
            node_type = None
            args = []

            symbol = msg.list.pop(0)
            if symbol.kind == unified_planning_pb2.ExpressionKind.Value(
                "FUNCTION_SYMBOL"
            ):
                node_type = op_to_node_type(symbol.atom.symbol)
            else:
                args.append(self.convert(symbol, problem, param_map))

            args.extend([self.convert(m, problem, param_map) for m in msg.list])

            assert node_type is not None

            return problem.env.expression_manager.create_node(
                node_type=node_type,
                args=tuple(args),
                payload=None,
            )

        raise ValueError(f"Unknown expression kind `{msg.kind}`")

    @handles(unified_planning_pb2.Atom)
    def _convert_atom(self, msg, problem):
        field = msg.WhichOneof("content")
        # No atom
        if field is None:
            return None

        value = getattr(msg, field)
        if field == "int":
            # TODO: deal with bounds
            return problem.env.expression_manager.Int(value)
        elif field == "real":
            # TODO: deal with bounds
            return problem.env.expression_manager.Real(value)
        elif field == "boolean":
            return problem.env.expression_manager.Bool(value)
        else:
            # If atom symbols, return the equivalent UP alternative
            # Note that parameters are directly handled at expression level
            if problem.has_object(value):
                return problem.env.expression_manager.ObjectExp(
                    obj=problem.object(value)
                )
            else:
                return problem.fluent(value)

    @handles(unified_planning_pb2.TypeDeclaration)
    def _convert_type_declaration(self, msg):
        if msg.type_name == "bool":
            return BoolType()
        elif msg.type_name.startswith("integer["):
            tmp = msg.type_name.split("[")[1].split("]")[0].split(", ")
            lb = None
            ub = None
            if tmp[0] != "-inf":
                lb = int(tmp[0])
            elif tmp[1] != "inf":
                ub = int(tmp[1])
            return IntType(lower_bound=lb, upper_bound=ub)
        elif msg.type_name.startswith("real["):
            tmp = msg.type_name.split("[")[1].split("]")[0].split(", ")
            lb = None
            ub = None
            if tmp[0] != "-inf":
                lb = fractions.Fraction(tmp[0])
            elif tmp[1] != "inf":
                ub = fractions.Fraction(tmp[1])
            return RealType(lower_bound=lb, upper_bound=ub)
        else:
            parent = None
            if parent != "":
                parent = UserType(msg.parent_type)
            return UserType(msg.type_name, parent)

    @handles(unified_planning_pb2.Problem)
    def _convert_problem(self, msg, problem):
        PROBLEM = Problem(name=msg.problem_name, env=problem.env)
        for obj in msg.objects:
            PROBLEM.add_object(self.convert(obj, problem))
        for f in msg.fluents:
            PROBLEM.add_fluent(self.convert(f, problem))
        for f in msg.actions:
            PROBLEM.add_action(self.convert(f, problem))
        for eff in msg.timed_effects:
            PROBLEM.add_timed_effect(self.convert(eff, problem))

        for assign in msg.initial_state:
            PROBLEM.set_initial_value(
                fluent=self.convert(assign.fluent, problem, []),
                value=self.convert(assign.value, problem, []),
            )

        for goal in msg.goals:
            timing = self.convert(goal.timing)
            goal = self.convert(goal.goal, problem, [])
            # TODO: Add timed goals
            PROBLEM.add_goal(goal)

        # TODO: add features
        for metric in msg.metrics:
            PROBLEM.add_quality_metric(self.convert(metric, problem, None))

        return PROBLEM

    @handles(unified_planning_pb2.Metric)
    def _convert_metric(self, msg, problem, param_map):
        if msg.kind == unified_planning_pb2.Metric.MINIMIZE_ACTION_COSTS:
            costs = {}
            for a, cost in msg.action_costs.items():
                costs.update(
                    {problem.action(a): self.convert(cost, problem, param_map)}
                )
            return unified_planning.model.metrics.MinimizeActionCosts(costs=costs)

    @handles(unified_planning_pb2.Action)
    def _convert_action(self, msg, problem):
        parameters = OrderedDict()
        action: unified_planning.model.Action

        for param in msg.parameters:
            parameters[param.name] = convert_type_str(param.type, problem.env)

        if msg.HasField("duration"):
            action = DurativeAction(msg.name, parameters)
            action.set_duration_constraint(self.convert(msg.duration, problem))
        else:
            action = InstantaneousAction(msg.name, parameters)

        self.current_action = action
        for cond in msg.conditions:
            exp = self.convert(cond.cond, problem, parameters)
            try:
                action.add_condition(self.convert(cond.span), exp)
            except AttributeError:
                action.add_precondition(exp)

        for eff in msg.effects:
            exp = self.convert(eff.effect, problem, parameters)
            try:
                action.add_effect(
                    timing=self.convert(eff.occurence_time),
                    fluent=exp.fluent,
                    value=exp.value,
                    condition=exp.condition,
                )
            except TypeError:
                action.add_effect(
                    fluent=exp.fluent, value=exp.value, condition=exp.condition
                )

        return action

    @handles(unified_planning_pb2.EffectExpression)
    def _convert_effect(self, msg, problem, param_map):
        # EffectKind
        kind = 0
        if msg.kind == unified_planning_pb2.EffectExpression.EffectKind.Value("ASSIGN"):
            kind = EffectKind.ASSIGN
        elif msg.kind == unified_planning_pb2.EffectExpression.EffectKind.Value(
            "INCREASE"
        ):
            kind = EffectKind.INCREASE
        elif msg.kind == unified_planning_pb2.EffectExpression.EffectKind.Value(
            "DECREASE"
        ):
            kind = EffectKind.DECREASE

        return Effect(
            fluent=self.convert(msg.fluent, problem, param_map),
            value=self.convert(msg.value, problem, param_map),
            condition=self.convert(msg.condition, problem, param_map),
            kind=kind,
        )

    @handles(unified_planning_pb2.Duration)
    def _convert_duration(self, msg, problem):
        return unified_planning.model.timing.DurationInterval(
            lower=self.convert(msg.controllable_in_bounds.lower, problem, []),
            upper=self.convert(msg.controllable_in_bounds.upper, problem, []),
            is_left_open=bool(msg.controllable_in_bounds.is_left_open),
            is_right_open=bool(msg.controllable_in_bounds.is_right_open),
        )

    @handles(unified_planning_pb2.TimeInterval)
    def _convert_timed_interval(self, msg):
        return unified_planning.model.TimeInterval(
            lower=self.convert(msg.lower),
            upper=self.convert(msg.upper),
            is_left_open=msg.is_left_open,
            is_right_open=msg.is_right_open,
        )

    @handles(unified_planning_pb2.Timing)
    def _convert_timing(self, msg):
        return unified_planning.model.Timing(
            delay=int(msg.delay), timepoint=self.convert(msg.timepoint)
        )

    @handles(unified_planning_pb2.Timepoint)
    def _convert_timepoint(self, msg):
        if msg.kind == unified_planning_pb2.Timepoint.TimepointKind.Value(
            "GLOBAL_START"
        ):
            return unified_planning.model.timing.Timepoint(
                kind=unified_planning.model.timing.TimepointKind.GLOBAL_START
            )
        elif msg.kind == unified_planning_pb2.Timepoint.TimepointKind.Value(
            "GLOBAL_END"
        ):
            return unified_planning.model.timing.Timepoint(
                kind=unified_planning.model.timing.TimepointKind.GLOBAL_END
            )
        elif msg.kind == unified_planning_pb2.Timepoint.TimepointKind.Value("START"):
            return unified_planning.model.timing.Timepoint(
                kind=unified_planning.model.timing.TimepointKind.START,
            )
        elif msg.kind == unified_planning_pb2.Timepoint.TimepointKind.Value("END"):
            return unified_planning.model.timing.Timepoint(
                kind=unified_planning.model.timing.TimepointKind.END,
            )

    @handles(unified_planning_pb2.Plan)
    def _convert_plan(self, msg, problem):
        actions = [self.convert(a, problem) for a in msg.actions]
        if isinstance(actions[0], tuple):  # TODO: Fix this
            return unified_planning.plan.TimeTriggeredPlan(actions)
        else:
            return unified_planning.plan.SequentialPlan(actions=actions)

    @handles(unified_planning_pb2.ActionInstance)
    def _convert_action_instance(self, msg, problem):
        # action instance paramaters are atoms but in UP they are FNodes
        # converting to up.model.FNode
        parameters = []
        for param in msg.parameters:
            assert param.HasField("symbol")
            assert problem.has_object(param.symbol)

            parameters.append(
                problem.env.expression_manager.create_node(
                    node_type=OperatorKind.OBJECT_EXP,
                    args=(),
                    payload=problem.object(param.symbol),
                )
            )

        action_instance = unified_planning.plan.ActionInstance(
            problem.action(msg.action_name),
            parameters,
        )
        if msg.end_time > 0.0 or msg.start_time > 0.0:
            # FIXME: This is a hack to get the Fraction from double
            start_time = fractions.Fraction(str(msg.start_time))
            end_time = fractions.Fraction(str(msg.end_time))
            return (
                start_time,  # Absolute Start Time
                action_instance,
                end_time - start_time,  # Duration
            )
        else:
            return action_instance
