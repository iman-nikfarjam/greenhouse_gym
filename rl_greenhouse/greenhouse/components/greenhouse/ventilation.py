from rl_greenhouse.greenhouse.types import State, Weather, Action, Info
from rl_greenhouse.greenhouse.components.base import GreenhouseComponent
from rl_greenhouse.greenhouse.physics.humidity import humidity_rh_to_abs, humidity_abs_to_rh
from rl_greenhouse.greenhouse.physics.unit_conversion import ppm_to_gpm3, gpm3_to_ppm


class Ventilation(GreenhouseComponent):
    """
    Ventilation for the greenhouse. Models heat exchange through the window as well as gas exchange (Carbon PPM).
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters,
            parameters_used={
                "heat_exchange_open_window": "Heat exchange in W/m²K with an open window",
                "greenhouse_heat_capacity": "The heat capacity of the greenhouse in J/m² K",
            },
        )
        self.heat_exchange_open_window = hyper_parameters["heat_exchange_open_window"]
        self.greenhouse_heat_capacity = hyper_parameters["greenhouse_heat_capacity"]
        self.max_ventilation_capacity = hyper_parameters["max_ventilation_capacity"]  # m3/h
        self.greenhouse_height = hyper_parameters["greenhouse_height"]

    def step(self, info: Info, state: State, weather: Weather, action: Action, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param delta_time: Delta time in seconds
        """
        # We can skip this if the ventilation system is closed.
        if action.ventilation_position == 0:
            return state

        fraction_replaced, average_wind_speed = self._calculate_fraction_replaced(action, delta_time)

        # Update state
        state.wind_speed = average_wind_speed
        state.temperature_air = self._adjust_temperature(state, weather, action, delta_time)
        state.carbon_ppm = self._adjust_co2_concentration(state, fraction_replaced)
        state.relative_humidity = self._adjust_relative_humidity(state, weather, fraction_replaced)
        return state

    def _calculate_fraction_replaced(self, action: Action, delta_time: float):
        # Assumes a linear relation while this is far from true
        m3_air_exchanged_per_second = action.ventilation_position * self.max_ventilation_capacity / 3600
        air_per_m2_in_greenhouse = self.greenhouse_height
        fraction_replaced = min(max(delta_time * m3_air_exchanged_per_second / air_per_m2_in_greenhouse, 0), 1)
        return fraction_replaced, (m3_air_exchanged_per_second / air_per_m2_in_greenhouse)

    def _adjust_temperature(self, state: State, weather: Weather, action: Action, delta_time: float) -> float:
        temperature_difference = state.temperature_air - weather.temperature_out
        heat_flux = -temperature_difference * self.heat_exchange_open_window * action.ventilation_position**0.5
        heat_added = heat_flux * delta_time
        return state.temperature_air + heat_added / self.greenhouse_heat_capacity

    def _adjust_co2_concentration(self, state: State, fraction_replaced: float) -> float:
        new_co2_gpm3 = fraction_replaced * ppm_to_gpm3(400) + (1 - fraction_replaced) * ppm_to_gpm3(state.carbon_ppm)
        return gpm3_to_ppm(new_co2_gpm3)

    def _adjust_relative_humidity(self, state: State, weather: Weather, fraction_replaced: float) -> float:
        replaced_abs_vapor_density = humidity_rh_to_abs(weather.temperature_out, weather.relative_humidity_out)
        greenhouse_abs_vapor_density = humidity_rh_to_abs(state.temperature_air, state.relative_humidity)

        abs_vapor_density = (
            fraction_replaced * replaced_abs_vapor_density + (1 - fraction_replaced) * greenhouse_abs_vapor_density
        )

        return humidity_abs_to_rh(state.temperature_air, abs_vapor_density)
