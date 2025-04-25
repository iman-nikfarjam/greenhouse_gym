from rl_greenhouse.greenhouse.physics import constants
from rl_greenhouse.greenhouse.types import State, Weather, Action, Info
from rl_greenhouse.greenhouse.components.base import GreenhouseComponent
from rl_greenhouse.greenhouse.physics.humidity import humidity_rh_to_abs, saturated_vapor_density, humidity_abs_to_rh


class HydroponicsBasin(GreenhouseComponent):
    """
    A simulation model for evaporation and condensation of water inside the greenhouse.
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters,
            parameters_used={
                "greenhouse_heat_capacity": "The heat capacity of the greenhouse in J/m2 K",
                "water_evaporation_coefficient_base": "",
                "water_evaporation_coefficient_wind": "",
                "water_evaporation_heat": "",
                "greenhouse_height": "",
                "volume_flow_by_wall": "",
            },
        )
        # kg/m²h
        self.water_evaporation_coefficient_base = hyper_parameters["water_evaporation_coefficient_base"]
        self.water_evaporation_coefficient_wind = hyper_parameters["water_evaporation_coefficient_wind"]
        self.water_exposed_area_per_m2 = hyper_parameters["water_exposed_area_per_m2"]
        self.evaporation_heat_of_water = hyper_parameters["water_evaporation_heat"]
        self.greenhouse_heat_capacity = hyper_parameters["greenhouse_heat_capacity"]
        self.wall_temp_fraction_inside_air = hyper_parameters["wall_temp_fraction_inside_air"]
        self.greenhouse_height = hyper_parameters["greenhouse_height"]
        self.volume_flow_by_wall = hyper_parameters["volume_flow_by_wall"]

    def step(self, info: Info, state: State, weather: Weather, action: Action, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param delta_time: Delta time in seconds
        """
        # kg
        evaporated_water = self._calculate_evaporated_water(state, delta_time) - self._calculate_condensated_water(
            state, weather, delta_time
        )
        state.relative_humidity = self._update_relative_humidity(state, evaporated_water)
        state.temperature_air = self._update_temperature(state, evaporated_water)
        return state

    def _update_temperature(self, state: State, evaporated_water: float) -> float:
        heat_used = self.evaporation_heat_of_water * evaporated_water
        return state.temperature_air - heat_used / self.greenhouse_heat_capacity

    def _update_relative_humidity(self, state: State, evaporated_water: float) -> float:
        abs_vapor_density = humidity_rh_to_abs(state.temperature_air, state.relative_humidity) / 1000  # kg / m3
        abs_vapor = abs_vapor_density * self.greenhouse_height  # kg
        total_vapor = evaporated_water + abs_vapor
        total_vapor_density = 1000 * total_vapor / self.greenhouse_height  # g / m3
        return humidity_abs_to_rh(state.temperature_air, total_vapor_density)

    def _calculate_condensated_water(self, state: State, weather: Weather, delta_time: float) -> float:
        abs_vapor_density = humidity_rh_to_abs(state.temperature_air, state.relative_humidity) / 1000  # kg / m3

        # Very simple and stupid approximation
        fraction = self.wall_temp_fraction_inside_air
        near_surface_temp = fraction * state.temperature_air + (1 - fraction) * weather.temperature_out

        max_abs_vapor_density = saturated_vapor_density(near_surface_temp) / 1000  # kg / m3
        cubic_meters_that_pass_by_the_wall = self.volume_flow_by_wall * delta_time
        water_surplus = cubic_meters_that_pass_by_the_wall * max(abs_vapor_density - max_abs_vapor_density, 0)
        return water_surplus

    def _calculate_evaporated_water(self, state: State, delta_time: float) -> float:
        # kg/m²h
        evaporation_coefficient = (
            self.water_evaporation_coefficient_base + self.water_evaporation_coefficient_wind * state.wind_speed
        )

        abs_vapor_density = humidity_rh_to_abs(state.temperature_air, state.relative_humidity) / 1000  # kg / m3
        max_abs_vapor_density = saturated_vapor_density(state.temperature_air) / 1000  # kg / m3

        humidity_dry_air_ratio = abs_vapor_density / constants.DENSITY_AIR  # kg / kg
        max_humidity_dry_air_ratio = max_abs_vapor_density / constants.DENSITY_AIR  # kg / kg

        # kg / s
        water_evaporation_rate = (
            evaporation_coefficient
            * self.water_exposed_area_per_m2
            * (max_humidity_dry_air_ratio - humidity_dry_air_ratio)
            / 3600
        )
        evaporated_water = water_evaporation_rate * delta_time  # kg
        return evaporated_water


if __name__ == "__main__":
    hydroponics = HydroponicsBasin(
        {
            "water_evaporation_coefficient_base": 26,
            "water_evaporation_coefficient_wind": 19,
            "water_evaporation_heat": 2_453_500,
            "greenhouse_heat_capacity": 250_000,
            "greenhouse_height": 3,
            "water_exposed_area_per_m2": 0.20,
            "wall_temp_fraction_inside_air": 2 / 3,
            "volume_flow_by_wall": 0.01,
        }
    )

    info = Info()
    state = State(relative_humidity=0.4)
    weather = Weather()
    action = Action()
    delta_time = 60

    for i in range(60):
        state = hydroponics.step(info, state, weather, action, delta_time)
        print(state.relative_humidity)
        print(state.temperature_air)
