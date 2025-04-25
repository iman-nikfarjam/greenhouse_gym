from rl_greenhouse.greenhouse.types import State, Weather, Action, Info
from rl_greenhouse.greenhouse.components.base import GreenhouseComponent


class WallConduction(GreenhouseComponent):
    """
    Source for wall heat transfer vs wind speed:
    RELATIONSHIP ANALYSIS OF WALL TRANSMITTANCE AND WIND SPEED WITH NUMERICAL METHOD
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters,
            parameters_used={
                "roof_heat_transfer_rate": "R value in [W/k] for the greenhouse roof",
                "screen_blackout_insulation": "R value in [W/k] for the greenhouse blackout screen if enabled",
                "screen_transparent_insulation": "R value in [W/k] for the greenhouse transparent screen if enabled",
                "greenhouse_heat_capacity": "The heat capacity of the greenhouse in J/m² K",
            },
        )
        self.roof_heat_transfer_rate = hyper_parameters["roof_heat_transfer_rate"]  # 1/0.95 for single pane glass
        self.roof_h_per_wind_speed = hyper_parameters["roof_h_wind_effect"]
        self.screen_blackout_insulation = hyper_parameters["screen_blackout_insulation"]  # 0.85
        self.screen_transparent_insulation = hyper_parameters["screen_transparent_insulation"]  # 0.85
        self.greenhouse_heat_capacity = hyper_parameters["greenhouse_heat_capacity"]

    def step(self, info: Info, state: State, weather: Weather, action: Action, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param delta_time: Delta time in seconds
        """
        r_roof = 1 / (self.roof_heat_transfer_rate + weather.wind_out * self.roof_h_per_wind_speed)

        # Quick falloff of R-value when screen is partially open. Actually this is incorrect # Fixme
        r_blackout = action.screen_blackout_position**10 * self.screen_blackout_insulation
        r_transparent = action.screen_transparent_position**10 * self.screen_transparent_insulation

        r_value = r_roof + r_blackout + r_transparent

        heat_flux = -((state.temperature_air - weather.temperature_out) / r_value)
        heat_added = heat_flux * delta_time

        # Update state
        state.temperature_air += heat_added / self.greenhouse_heat_capacity
        return state
