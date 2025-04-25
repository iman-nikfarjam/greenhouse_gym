from rl_greenhouse.greenhouse.components.base import WeatherComponent
from rl_greenhouse.greenhouse.types import Weather, Info


class ConstantWeatherStation(WeatherComponent):
    """
    Weather station that returns the default constant weather observation
    """

    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters, parameters_used={})

    def step(self, info: Info) -> Weather:
        """
        Calculates the resulting weather from applying the effect of this subsystem.

        :param info: Information about the global state of the system.
        """
        return Weather()

    def reset(self, datetime):
        """
        Reset this subsystem.
        """
        return Weather()
