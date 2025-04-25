from rl_greenhouse.greenhouse.types import State, Weather, AdversarialAction, Info
from rl_greenhouse.greenhouse.components.base import AdversarialComponent
from rl_greenhouse.greenhouse.physics.humidity import humidity_rh_to_abs, humidity_abs_to_rh


class AdversarialInputComponent(AdversarialComponent):
    """
    Non-realistic Greenhouse component. Only purpose is to add forces as perturbations to the system for adversary RL.

    Important Notes:
        - This actor uses the same action space as the original
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(hyper_parameters=hyper_parameters, parameters_used={})
        self.greenhouse_heat_capacity = hyper_parameters["greenhouse_heat_capacity"]
        self.greenhouse_height = hyper_parameters["greenhouse_height"]

    def step(self, info: Info, state: State, weather: Weather, action: AdversarialAction, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The adversarial action that is applied
        :param delta_time: Delta time in seconds
        """
        heat_added = action.heat_offset * delta_time
        state.temperature_air = state.temperature_air + heat_added / self.greenhouse_heat_capacity

        absolute_humidity_added = action.humidity_absolute_offset * delta_time
        state.relative_humidity = self._update_relative_humidity(state, absolute_humidity_added)
        if state.relative_humidity < 0:
            state.relative_humidity = 0
        elif state.relative_humidity > 1:
            state.relative_humidity = 1

        carbon_added = action.carbon_offset * delta_time
        state.carbon_ppm += carbon_added
        if state.carbon_ppm < 0:
            state.carbon_ppm = 0

        light_added = action.light_offset
        state.par_light += light_added
        if state.par_light < 0:
            state.par_light = 0

        return state

    def _update_relative_humidity(self, state: State, evaporated_water: float) -> float:
        abs_vapor_density = humidity_rh_to_abs(state.temperature_air, state.relative_humidity) / 1000  # kg / m3
        abs_vapor = abs_vapor_density * self.greenhouse_height  # kg
        total_vapor = evaporated_water + abs_vapor
        total_vapor_density = 1000 * total_vapor / self.greenhouse_height  # g / m3
        return humidity_abs_to_rh(state.temperature_air, total_vapor_density)
