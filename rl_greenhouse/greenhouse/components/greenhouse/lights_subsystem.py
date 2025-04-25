from rl_greenhouse.greenhouse.physics import constants
from rl_greenhouse.greenhouse.types import State, Weather, Action, Info
from rl_greenhouse.greenhouse.components.base import GreenhouseComponent


class StandardLights(GreenhouseComponent):
    """
    The subsystem for the lights.
    """

    def __init__(self, hyper_parameters: dict):
        super().__init__(
            hyper_parameters=hyper_parameters,
            parameters_used={
                "reflectance_greenhouse": "What fraction of the sunlight is reflected by the greenhouse",
                "reflectance_blackout": "What fraction of the sunlight is reflected by the blackout screen if fully enabled",
                "reflectance_transparent": "What fraction of the sunlight is reflected by the transparent screen if fully enabled",
                "lamp_intensity": "The intensity of the lamp in par light [µmol/m² s]",
                "lamp_efficiency": "Convert intensity of lamp in par light to Watts consumed",
                "greenhouse_heat_capacity": "The heat capacity of the greenhouse in J/m² K",
                "heat_adsorption": "The amount of light that is trapped by the greenhouse system between 0 and 1",
            },
        )
        self.reflectance_greenhouse = hyper_parameters["reflectance_greenhouse"]
        self.reflectance_blackout = hyper_parameters["reflectance_blackout"]
        self.reflectance_transparent = hyper_parameters["reflectance_transparent"]
        self.lamp_intensity = hyper_parameters["lamp_intensity"]
        self.lamp_par_to_watts = hyper_parameters["lamp_par_to_watts"]
        self.greenhouse_heat_capacity = hyper_parameters["greenhouse_heat_capacity"]
        self.fraction_sun_heat_absorbed = hyper_parameters["fraction_sun_heat_absorbed"]
        self.heat_blocked_by_screens = hyper_parameters["heat_blocked_by_screens"]
        self.last_time = 0

    def step(self, info: Info, state: State, weather: Weather, action: Action, delta_time: float) -> State:
        """
        Calculates the resulting state from applying the effect of this subsystem.

        :param info: The current extra information state
        :param state: The current state of the system
        :param weather: The weather at this time
        :param action: The action that is applied
        :param delta_time: Delta time in seconds
        """
        # Incoming from the sun
        fraction_light_passes = 1 - self.reflectance_greenhouse
        fraction_light_passes *= 1 - self.reflectance_blackout * action.screen_blackout_position
        fraction_light_passes *= 1 - self.reflectance_transparent * action.screen_transparent_position
        solar_incoming = weather.illumination_out * fraction_light_passes  # W/m² that passes through

        # Incoming from the lights
        lamps_incoming = action.enable_lamps * self.lamp_intensity

        # Total PAR
        par_light = solar_incoming * constants.SOLAR_TO_PAR + lamps_incoming

        # Total heating from the sun
        heating_power_sun = (
            self.heat_blocked_by_screens * solar_incoming
            + (1 - self.heat_blocked_by_screens) * weather.illumination_out
        ) * self.fraction_sun_heat_absorbed

        power = heating_power_sun
        heat_added = power * delta_time
        state.temperature_air += heat_added / self.greenhouse_heat_capacity

        # Electricity Usage
        state.electric_power = action.enable_lamps * self.lamp_intensity * self.lamp_par_to_watts

        # Update state
        state.par_light = par_light
        return state
