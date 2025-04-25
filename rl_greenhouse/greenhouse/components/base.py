import datetime
from abc import abstractmethod
from typing import Any, Dict

from rl_greenhouse.greenhouse.types import Action, AdversarialAction, State, Weather, Plant, Info


class Component:
    """
    Simple component of the greenhouse model. Basic interface.

    :param hyper_parameters: The hyperparameters for the entire greenhouse system
    :param parameters_used: A list mapping hyperparameter to its description
    """

    def __init__(self, hyper_parameters: dict, parameters_used: Dict[str, str]):
        self.hyper_parameters: dict = hyper_parameters
        self.parameters_used: Dict[str, (str, Any)] = parameters_used

    def step(self, **kwargs):
        """Step method to perform a single system step."""


class GreenhouseComponent(Component):
    """
    The default subsystem interface. Maps s(t) a(t)

    Its purpose is to simplify the codebase by splitting the logic into different compartments.
    """

    @abstractmethod
    def step(self, info: Info, state: State, weather: Weather, action: Action, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param delta_time: Delta time in seconds
        """

    def reset(self):
        """
        Reset this subsystem.
        """


class AdversarialComponent(Component):
    """
    Component designed to support adversarial inputs to the system.
    Takes in an AdversarialAction instead of a regular action.
    """

    @abstractmethod
    def step(self, info: Info, state: State, weather: Weather, action: AdversarialAction, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param delta_time: Delta time in seconds
        """

    def reset(self):
        """
        Reset this subsystem.
        """


class PlantComponent(Component):
    """
    The default subsystem interface. Maps s(t) a(t)

    Its purpose is to simplify the codebase by splitting the logic into different compartments.
    """

    def __init__(self, hyper_parameters, parameters_used=None):
        if parameters_used is None:
            parameters_used = {}
        super().__init__(hyper_parameters, parameters_used=parameters_used)

    @abstractmethod
    def current_selling_price(self, plant: Plant) -> float:
        """
        Calculates the current selling price of the plant
        :param plant: The current state of the plant
        """

    @abstractmethod
    def step(self, info: Info, plant: Plant, state: State, weather: Weather, delta_time: float) -> (Plant, State, dict):
        """
        Calculates the resulting plant state from the current state in the greenhouse.

        :param info: The current extra information state
        :param plant: The current state of the plant
        :param state: The current state of the system
        :param weather: The weather at this time
        :param delta_time: Delta time in seconds
        """

    @abstractmethod
    def reset(self) -> Plant:
        """
        Reset this subsystem.
        """


class WeatherComponent(Component):
    """
    The default weather interface.
    """

    def __init__(self, hyper_parameters, parameters_used=None):
        if parameters_used is None:
            parameters_used = {}
        super().__init__(hyper_parameters, parameters_used=parameters_used)

    @abstractmethod
    def step(self, info: Info) -> Weather:
        """
        Calculates the resulting weather from applying the effect of this subsystem.

        :param info: Information about the global state of the system.
        """

    @abstractmethod
    def reset(self, start_date: datetime.datetime) -> Weather:
        """
        Reset this subsystem.
        """


class EconomicComponent(Component):
    """
    The default subsystem interface. Maps s(t) a(t)

    Its purpose is to simplify the codebase by splitting the logic into different compartments.
    """

    @abstractmethod
    def step(
        self,
        info: Info,
        plant: Plant,
        state: State,
        weather: Weather,
        action: Action,
        current_selling_price: float,
        delta_time: float,
    ) -> Info:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param plant: The current state of the plant
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param current_selling_price: The current selling price per plant
        :param delta_time: Delta time in seconds
        """

    def reset(self):
        """
        Reset this subsystem.
        """
