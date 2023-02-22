from fractions import Fraction
from typing import Optional, Union

from unified_planning.environment import get_environment, Environment
from unified_planning.exceptions import UPProblemDefinitionError
from unified_planning.model import (
    Timepoint,
    Timing,
    TimepointKind,
    Fluent,
)
import unified_planning as up
from unified_planning.model.scheduling.chronicle import Chronicle


class Activity(Chronicle):
    """Activity is essentially an interval with start and end timing that facilitates defining constraints in the
    associated SchedulingProblem"""

    def __init__(
        self, name: str, duration: Optional[int], _env: Optional[Environment] = None
    ):
        Chronicle.__init__(self, name, _env=_env)
        start_tp = Timepoint(TimepointKind.START, container=name)
        end_tp = Timepoint(TimepointKind.END, container=name)
        self._start = Timing(0, start_tp)
        self._end = Timing(0, end_tp)

        self._duration = up.model.timing.FixedDuration(
            self._environment.expression_manager.Int(0)
        )
        if duration is not None:
            self.set_fixed_duration(duration)

    @property
    def start(self) -> Timing:
        return self._start

    @property
    def end(self) -> Timing:
        return self._end

    @property
    def duration(self) -> "up.model.timing.DurationInterval":
        """Returns the `activity` `duration interval`."""
        return self._duration

    def set_fixed_duration(self, value: Union["up.model.fnode.FNode", int, Fraction]):
        """
        Sets the `duration interval` for this `activity` as the interval `[value, value]`.

        :param value: The `value` set as both edges of this `action's duration`.
        """
        (value_exp,) = self._environment.expression_manager.auto_promote(value)
        self._set_duration_constraint(up.model.timing.FixedDuration(value_exp))

    def set_duration_bounds(
        self,
        lower: Union["up.model.fnode.FNode", int, Fraction],
        upper: Union["up.model.fnode.FNode", int, Fraction],
    ):
        """
        Sets the `duration interval` for this `activity` as the interval `[lower, upper]`.

        :param lower: The value set as the lower edge of this `action's duration`.
        :param upper: The value set as the upper edge of this `action's duration`.
        """
        lower_exp, upper_exp = self._environment.expression_manager.auto_promote(
            lower, upper
        )
        self._set_duration_constraint(
            up.model.timing.ClosedDurationInterval(lower_exp, upper_exp)
        )

    def _set_duration_constraint(self, duration: "up.model.timing.DurationInterval"):
        """
        Sets the `duration interval` for this `action`.

        :param duration: The new `duration interval` of this `action`.
        """
        lower, upper = duration.lower, duration.upper
        tlower = self._environment.type_checker.get_type(lower)
        tupper = self._environment.type_checker.get_type(upper)
        assert tlower.is_int_type() or tlower.is_real_type()
        assert tupper.is_int_type() or tupper.is_real_type()
        if (
            lower.is_constant()
            and upper.is_constant()
            and (
                upper.constant_value() < lower.constant_value()
                or (
                    upper.constant_value() == lower.constant_value()
                    and (duration.is_left_open() or duration.is_right_open())
                )
            )
        ):
            raise UPProblemDefinitionError(
                f"{duration} is an empty interval duration of action: {self.name}."
            )
        self._duration = duration

    def uses(self, resource: Fluent, amount: int = 1):
        self.add_decrease_effect(self.start, resource, amount)
        self.add_increase_effect(self.end, resource, amount)

    def set_release_date(self, date: int):
        self.add_constraint(get_environment().expression_manager.LE(date, self.start))

    def set_deadline(self, date: int):
        self.add_constraint(get_environment().expression_manager.LE(self.end, date))
