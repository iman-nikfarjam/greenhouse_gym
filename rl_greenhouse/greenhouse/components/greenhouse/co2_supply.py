from rl_greenhouse.greenhouse.physics import constants
from rl_greenhouse.greenhouse.types import State, Weather, Action, Info
from rl_greenhouse.greenhouse.components.base import GreenhouseComponent


class Co2Supply(GreenhouseComponent):
    """
    The subsystem for the lights.
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters,
            parameters_used={
                "greenhouse_height": "",
                "carbon_ppm_max": "",
            },
        )
        self.greenhouse_height = hyper_parameters["greenhouse_height"]
        self.carbon_ppmv_max = hyper_parameters["carbon_ppmv_max"]

    def step(self, info: Info, state: State, weather: Weather, action: Action, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param delta_time: Delta time in seconds
        """
        volume_flow_rate = action.co2_flow_rate / constants.DENSITY_CO2
        injected_co2_volume = delta_time * volume_flow_rate

        existing_co2_volume = state.carbon_ppm * self.greenhouse_height / 1e6  # m3

        # Greenhouse volume per square meter = 1 x 1 x greenhouse height. PPMV increase below:
        volume_fraction = (existing_co2_volume + injected_co2_volume) / (injected_co2_volume + self.greenhouse_height)
        state.carbon_ppm = min(1e6 * volume_fraction, self.carbon_ppmv_max)
        return state
