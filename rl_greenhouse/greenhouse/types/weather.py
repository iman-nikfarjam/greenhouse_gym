"""


"""
from dataclasses import dataclass

from rl_greenhouse.greenhouse.types.base import BaseDataClass


@dataclass
class Weather(BaseDataClass):
    """
    The default observation interface.

    illumination_out: The illumination outside
    rel_humidity_out: The relative humidity outside
    temperature_out: The temperature in the greenhouse in degrees Celsius
    wind_out: The wind speed outside in m/s
    """

    illumination_out: float = 0
    relative_humidity_out: float = 0
    temperature_out: float = 10
    wind_out: float = 0

    INTERVALS = {
        "illumination_out": (0, 1000),
        "relative_humidity_out": (0, 1),
        "temperature_out": (-20, 60),
        "wind_out": (0, 20),
    }
    LABELS = list(INTERVALS.keys())


if __name__ == "__main__":
    weather = Weather()
    np_weather = weather.to_numpy_2d()
    print(weather)
    print(np_weather)
    print(weather.normalize().denormalize())
