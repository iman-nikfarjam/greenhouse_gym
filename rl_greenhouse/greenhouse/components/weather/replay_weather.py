from datetime import datetime

from rl_greenhouse.greenhouse.components.base import WeatherComponent
from rl_greenhouse.greenhouse.types import Weather, Observation, Info


class ReplayWeatherStation(WeatherComponent):
    """
    Weather station that returns a given weather pattern
    """

    def __init__(self, hyper_parameters):
        super().__init__(
            hyper_parameters,
            parameters_used={
                "weather_to_replay": "The weather either in a list of full states or a list of weather types interfaces."
            },
        )
        weather = hyper_parameters["weather_to_replay"]

        if isinstance(weather[0], Observation):
            weather = [full_state.weather for full_state in weather]

        if not isinstance(weather[0], Weather):
            raise TypeError(f"Weather of type [{type(weather[0])}] is not supported.")

        self.weather = weather

    def step(self, info: Info) -> Weather:
        """
        Calculates the resulting weather from applying the effect of this subsystem.

        :param info: Information about the global state of the system.
        """
        # Weather for time + 1 otherwise par light is calculated with old weather information which makes the
        # par light value incorrect. This shift fixes this by artificially first taking the weather w(t)
        # and then going into the update loop to predict s(t) as s(t) is dependent on w(t).
        idx = round(info.time) + 1
        return self.weather[idx]

    def reset(self, start_date: datetime):
        """
        Reset this subsystem.
        """
        return self.weather[0]
