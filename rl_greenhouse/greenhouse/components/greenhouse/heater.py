from rl_greenhouse.greenhouse.types import State, Weather, Action, Info
from rl_greenhouse.greenhouse.components.base import GreenhouseComponent


class Heater(GreenhouseComponent):
    """
    The subsystem for the heater.

    Models the heater as a heater on a heating pipe.
    - The heating pipe can be at most 100 degrees, otherwise the water boils and the excess heat is wasted.
    - The pipe transfers the heat into the surroundings directly.

    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters,
            parameters_used={
                "pipe_heat_exchange_coefficient": "Heat exchange between heating pipes and greenhouse in W/m2K",
                "greenhouse_heat_capacity": "The heat capacity of the greenhouse in J/m2 K",
            },
        )
        self.pipe_heat_exchange = hyper_parameters["pipe_heat_exchange_coefficient"]
        self.greenhouse_heat_capacity = hyper_parameters["greenhouse_heat_capacity"]
        self.heating_efficiency = hyper_parameters["heating_efficiency"]

    def step(self, info: Info, state: State, weather: Weather, action: Action, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param delta_time: Delta time in seconds
        """
        state.heating_pipe_temperature = max(
            min(state.temperature_air + action.heater_power / self.pipe_heat_exchange, 100), 0
        )

        # Actual heating is capped by the maximum pipe temperature of 100 degrees.
        temperature_difference = state.heating_pipe_temperature - state.temperature_air
        actual_heating_power = temperature_difference * self.pipe_heat_exchange

        heat_added = self.heating_efficiency * actual_heating_power * delta_time
        state.temperature_air = state.temperature_air + heat_added / self.greenhouse_heat_capacity
        return state
