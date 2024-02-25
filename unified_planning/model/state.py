# Copyright 2021-2023 AIPlan4EU project
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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import unified_planning as up
from unified_planning.exceptions import UPUsageError, UPValueError


class State(ABC):
    """This is an abstract class representing a classical `State`"""

    @abstractmethod
    def get_value(self, value: "up.model.FNode") -> "up.model.FNode":
        """
        This method retrieves the value in the state.
        NOTE that the searched value must be set in the state.

        :param value: The value searched for in the state.
        :return: The set value.
        """
        raise NotImplementedError


class UPState(State):
    """
    unified_planning implementation of the `State` interface.
    This class has an optional field `MAX_ANCESTORS` set to 20.

    The higher this number is, the less memory the data structure will use.
    The lower this number is, the less time the data structure will need to retrieve a value.

    To set your own number just extend this class and re-define the `MAX_ANCESTORS` value. It must be `> 0`
    """

    MAX_ANCESTORS: Optional[int] = 20

    def __init__(
        self,
        values: Dict["up.model.FNode", "up.model.FNode"],
        _father: Optional["UPState"] = None,
    ):
        """
        Creates a new `UPState` where the map values represents the get_value method. The parameter `_father`
        is for internal use only.
        """
        max_ancestors = type(self).MAX_ANCESTORS
        if max_ancestors is not None and max_ancestors < 1:
            raise UPValueError(
                f"The max_ancestor field of a class extending UPState must be > 0 or None: in the class {type(self)} it is set to {type(self).MAX_ANCESTORS}"
            )
        self._father = _father
        self._values = values
        if _father is None:
            self._ancestors = 0
        else:
            self._ancestors = _father._ancestors + 1

    def __repr__(self) -> str:
        current_instance: Optional[UPState] = self
        mappings: Dict["up.model.FNode", "up.model.FNode"] = {}
        while current_instance is not None:
            for k, v in current_instance._values.items():
                mappings.setdefault(k, v)
            current_instance = current_instance._father
        return str(mappings)

    def get_value(self, fluent: "up.model.FNode") -> "up.model.FNode":
        """
        This method retrieves the value of the given fluent in the `State`.
        NOTE that the searched fluent must be set in the state otherwise an
        exception is raised.

        :params fluent: The fluent searched for in the `UPState`.
        :return: The value set for the given fluent.
        """
        current_instance: Optional[UPState] = self
        while current_instance is not None:
            #breakpoint()
            value_found = current_instance._values.get(fluent, None)
            if value_found is not None:
                return value_found
            current_instance = current_instance._father
        raise UPUsageError(
            f"The state {self} does not have a value for the value {fluent}"
        )

    def get_dot_values(self, agent: "Agent", only_true_values: Optional[bool] = False) -> "up.model.FNode":
        """
        This method retrieves the value of the given fluent in the `State`.
        NOTE that the searched fluent must be set in the state otherwise an
        exception is raised.

        :params fluent: The fluent searched for in the `UPState`.
        :return: The value set for the given fluent.
        """
        current_instance: Optional[UPState] = self
        agent_states = {}
        while current_instance is not None:
            for i, v  in current_instance._values.items():

                if i.is_dot() and i.agent()==agent.name:
                    # Extract the fluent name and its value
                    fluent_name = str(i.args[0])  # Assuming args[0] contains the fluent name
                    fluent_value = v.constant_value()
                    # Apply the filter for only_true_values if needed
                    if only_true_values and (v.is_bool_constant() and v.is_false()):
                        continue  # Skip this fluent if only_true_values is True and the fluent's value is not True
                    agent_states[(agent.name, fluent_name)] = fluent_value

            #breakpoint()
            if agent_states is not {}:
                return agent_states
            current_instance = current_instance._father

        raise UPUsageError(
            f"The state {self} does not have a value for the agent {agent.name} and the value {fluent}"
        )


    def make_child(
        self,
        updated_values: Dict["up.model.FNode", "up.model.FNode"],
    ) -> "UPState":
        """
        Returns a different `UPState` in which every value in updated_values.keys() is evaluated as his mapping
        in new the `updated_values` dict and every other value is evaluated as in `self`.

        :param updated_values: The dictionary that contains the `values` that need to be updated in the new `UPState`.
        :return: The new `UPState` created.
        """
        max_ancestors = type(self).MAX_ANCESTORS
        # When max_ancestors is None or this state has already too many ancestors, retrieve every possible
        # assignment, the action path and return a new UPState without any ancestors
        if max_ancestors is None or self._ancestors >= max_ancestors:
            current_instance: Optional[UPState] = self
            complete_values = updated_values.copy()
            while current_instance is not None:
                for k, v in current_instance._values.items():
                    complete_values.setdefault(k, v)
                current_instance = current_instance._father
            return UPState(complete_values)
        # Otherwise just return a new UPState with self as ancestor
        return UPState(updated_values, self)
